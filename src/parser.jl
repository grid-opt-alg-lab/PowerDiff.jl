using LazyArtifacts
import PowerIO

"""Normalized MATPOWER bus record."""
struct ParsedBus
    bus_i::Int
    bus_type::Int
    pd::Float64
    qd::Float64
    gs::Float64
    bs::Float64
    area::Int
    vm::Float64
    va::Float64
    base_kv::Float64
    zone::Int
    vmax::Float64
    vmin::Float64
end

"""Normalized MATPOWER generator record with quadratic cost coefficients."""
struct ParsedGen
    index::Int
    gen_bus::Int
    pg::Float64
    qg::Float64
    qmax::Float64
    qmin::Float64
    vg::Float64
    mbase::Float64
    gen_status::Int
    pmax::Float64
    pmin::Float64
    cost::NTuple{3,Float64}
end

"""Normalized MATPOWER pi-model branch record."""
struct ParsedBranch
    index::Int
    f_bus::Int
    t_bus::Int
    br_r::Float64
    br_x::Float64
    br_b::Float64
    rate_a::Float64
    rate_b::Float64
    rate_c::Float64
    tap::Float64
    shift::Float64
    br_status::Int
    angmin::Float64
    angmax::Float64
end

"""Normalized active and reactive load record."""
struct ParsedLoad
    index::Int
    load_bus::Int
    pd::Float64
    qd::Float64
    status::Int
end

"""Normalized bus shunt record."""
struct ParsedShunt
    index::Int
    shunt_bus::Int
    gs::Float64
    bs::Float64
    status::Int
end

"""
    ParsedCase

Normalized MATPOWER network data used by PowerDiff constructors. Power quantities
are stored in per-unit values. Constructing `ParsedCase` programmatically assumes
the supplied values are already normalized.
"""
struct ParsedCase
    name::String
    source_version::String
    baseMVA::Float64
    bus::Vector{ParsedBus}
    gen::Vector{ParsedGen}
    branch::Vector{ParsedBranch}
    load::Vector{ParsedLoad}
    shunt::Vector{ParsedShunt}
end

"""
    get_path(library::Symbol)

Resolve an artifact-backed library path owned by PowerDiff.
"""
function get_path(library::Symbol)
    library == :pglib && return joinpath(artifact"PGLib_opf", "pglib-opf-23.07")
    throw(ArgumentError("unsupported library $library"))
end

"""
    parse_file(io::Union{IO,String}; library=nothing, validate=true, filetype="m")

Parse a MATPOWER v2 `.m` file into a normalized `ParsedCase`.

PowerDiff intentionally supports MATPOWER files only. Convert other formats
before constructing PowerDiff types.

PowerIO is the parser and data layer. PowerDiff normalizes the PowerIO `Network`
into its own [`ParsedCase`](@ref).
"""
function parse_file(io::Union{IO,String}; library=nothing, validate=true, filetype="m", kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "unsupported parse_file keyword(s): $(join(string.(keys(kwargs)), ", "))"))
    resolved = io isa String ? _resolve_case_path(io, library) : io
    resolved_type = resolved isa String ? lowercase(splitext(resolved)[2]) : ".$(lowercase(filetype))"
    resolved_type == ".m" || throw(ArgumentError(
        "unsupported network file type $resolved_type; PowerDiff supports MATPOWER v2 .m files only"))
    return parse_matpower(resolved; validate)
end

"""
    parse_matpower(io::IO; validate=true)
    parse_matpower(file::String; library=nothing, validate=true)

Parse MATPOWER v2 data into a normalized [`ParsedCase`](@ref).
"""
function parse_matpower(io::IO; validate=true)::ParsedCase
    try
        net = PowerIO.parse_str(read(io, String), "matpower")
        return _finish_parse(_parsedcase_from_powerio(net), validate)
    catch e
        e isa ArgumentError && rethrow()
        throw(ArgumentError("PowerDiff.parse_matpower: " * sprint(showerror, e)))
    end
end

function parse_matpower(file::String; library=nothing, validate=true)::ParsedCase
    resolved = _resolve_case_path(file, library)
    isfile(resolved) || throw(ArgumentError("invalid MATPOWER file $resolved"))
    try
        return _finish_parse(_parsedcase_from_powerio(_load_powerio_network(resolved)), validate)
    catch e
        e isa ArgumentError && rethrow()
        throw(ArgumentError("PowerDiff.parse_matpower: " * sprint(showerror, e)))
    end
end

# The parser builds a raw ParsedCase from PowerIO's Network, then this applies
# PowerDiff's normalization and validation.
function _finish_parse(parsed::ParsedCase, validate::Bool)::ParsedCase
    validate || return parsed
    parsed = _normalize_parsed_case(parsed)
    _validate_parsed_case(parsed)
    return parsed
end

"""
    parse_matpower_struct(file::String; kwargs...)

Compatibility alias for [`parse_matpower`](@ref).
"""
parse_matpower_struct(file::String; kwargs...) = parse_matpower(file; kwargs...)

"""
    _load_powerio_network(path) -> PowerIO.Network

Parse `path` with the PowerIO Rust core. PowerIO infers the format from the
extension and returns a raw, lossless network (MW/MVAr, degrees, raw bus types,
out of service elements retained), which [`_parsedcase_from_powerio`](@ref) then
normalizes.
"""
_load_powerio_network(path::AbstractString) = PowerIO.parse_file(String(path))

"""
    _parsedcase_from_powerio(net) -> ParsedCase

Adapter from a PowerIO `Network` to a normalized PowerDiff [`ParsedCase`](@ref).
PowerIO emits raw, lossless data, so this reuses PowerDiff's normalization
(`_normalize_buses`, `_parse_cost_tuple`, `_normalize_angle_bounds`) before the
shared `_finish_parse` tail (`_normalize_parsed_case` + `_validate_parsed_case`)
runs in `parse_matpower`.

PowerIO keeps loads and shunts as first class records, so the adapter builds `ParsedLoad` /
`ParsedShunt` straight from those vectors (no `_build_bus_injections`), and leaves
bus injections zeroed. It still calls `_normalize_buses`, because PowerIO carries
the raw file bus type and PowerDiff infers PV/slack itself.
"""
function _parsedcase_from_powerio(net)
    isempty(PowerIO.storage(net)) || throw(ArgumentError(
        "PowerDiff does not support storage records; remove or convert storage before parsing"))
    isempty(PowerIO.hvdc(net)) || throw(ArgumentError(
        "PowerDiff does not support HVDC/dcline records; remove or convert dcline before parsing"))
    base = PowerIO.base_mva(net)
    buses = [ParsedBus(b.id, PowerIO.bus_type_code(String(b.kind)), 0.0, 0.0, 0.0, 0.0,
                       b.area, b.vm, deg2rad(b.va), b.base_kv, b.zone, b.vmax, b.vmin)
             for b in PowerIO.buses(net)]
    gens = [ParsedGen(i, g.bus, g.pg / base, g.qg / base, g.qmax / base, g.qmin / base,
                      g.vg, g.mbase, g.in_service ? 1 : 0, g.pmax / base, g.pmin / base,
                      _parse_cost_tuple(_powerio_cost_row(g.cost), base))
            for (i, g) in enumerate(PowerIO.generators(net))]
    branches = ParsedBranch[]
    for (i, br) in enumerate(PowerIO.branches(net))
        angmin, angmax = _normalize_angle_bounds(deg2rad(br.angmin), deg2rad(br.angmax))
        push!(branches, ParsedBranch(
            i, br.from, br.to, br.r, br.x, br.b, br.rate_a / base, br.rate_b / base,
            br.rate_c / base, br.tap, deg2rad(br.shift), br.in_service ? 1 : 0, angmin, angmax))
    end
    loads = [ParsedLoad(i, l.bus, l.p / base, l.q / base, l.in_service ? 1 : 0)
             for (i, l) in enumerate(PowerIO.loads(net))]
    shunts = [ParsedShunt(i, s.bus, s.g / base, s.b / base, s.in_service ? 1 : 0)
              for (i, s) in enumerate(PowerIO.shunts(net))]
    buses = _normalize_buses(buses, gens)
    return ParsedCase(PowerIO.network_name(net), "2", base, buses, gens, branches, loads, shunts)
end

# Rebuild a MATPOWER `gencost` numeric row, `[model, startup, shutdown, ncost, coeffs...]`,
# from PowerIO's GenCost so `_parse_cost_tuple` applies the same `base_mva^(n-i)` rescale and
# 3-tuple padding as the native path (PowerIO's own `quadratic()` does not rescale). PowerIO
# leaves `cost` as `nothing` for a generator with no cost row, which yields a zero cost tuple
# through `_parse_cost_tuple`'s normal path.
function _powerio_cost_row(cost)
    cost === nothing && return [2.0, 0.0, 0.0, 1.0, 0.0]
    return Float64[Float64(cost.model), Float64(cost.startup), Float64(cost.shutdown),
                   Float64(cost.ncost), (Float64(c) for c in cost.coeffs)...]
end

_resolve_case_path(path::AbstractString, ::Nothing) = String(path)
_resolve_case_path(path::AbstractString, library) = joinpath(get_path(library), path)

function _parse_cost_tuple(row::Vector{Float64}, baseMVA::Float64)
    length(row) >= 5 || throw(ArgumentError("mpc.gencost row is incomplete"))
    all(isfinite, row) || throw(ArgumentError("mpc.gencost contains a non-finite value"))
    model = Int(row[1])
    model == 2 || throw(ArgumentError("only polynomial mpc.gencost model 2 is supported"))
    n = Int(row[4])
    n >= 1 || throw(ArgumentError("mpc.gencost must declare at least one coefficient"))
    length(row) >= 4 + n || throw(ArgumentError("mpc.gencost row declares $n coefficients but contains $(length(row) - 4)"))
    coeffs = [baseMVA^(n - i) * row[4 + i] for i in 1:n]
    while length(coeffs) > 1 && iszero(first(coeffs))
        popfirst!(coeffs)
    end
    length(coeffs) <= 3 || throw(ArgumentError("only constant, linear, and quadratic generator costs are supported"))
    return length(coeffs) == 3 ? (coeffs[1], coeffs[2], coeffs[3]) :
           length(coeffs) == 2 ? (0.0, coeffs[1], coeffs[2]) :
                                 (0.0, 0.0, coeffs[1])
end

function _normalize_parsed_case(data::ParsedCase)::ParsedCase
    active_bus_ids = Set(bus.bus_i for bus in data.bus if bus.bus_type != 4)
    buses = [bus for bus in data.bus if bus.bus_i in active_bus_ids]
    gens = [gen for gen in data.gen if gen.gen_status != 0 && gen.gen_bus in active_bus_ids]
    buses = _normalize_buses(buses, gens)
    bus_by_id = Dict(bus.bus_i => bus for bus in buses)
    branches = ParsedBranch[]
    for branch in data.branch
        branch.br_status != 0 || continue
        branch.f_bus in active_bus_ids || continue
        branch.t_bus in active_bus_ids || continue
        tap = iszero(branch.tap) ? 1.0 : branch.tap
        rate_a = branch.rate_a > 0 ? branch.rate_a : _fallback_rate_a(branch, bus_by_id)
        push!(branches, ParsedBranch(
            branch.index, branch.f_bus, branch.t_bus, branch.br_r, branch.br_x,
            branch.br_b, rate_a, branch.rate_b, branch.rate_c, tap, branch.shift,
            branch.br_status, branch.angmin, branch.angmax
        ))
    end
    loads = [load for load in data.load if load.status != 0 && load.load_bus in active_bus_ids]
    shunts = [shunt for shunt in data.shunt if shunt.status != 0 && shunt.shunt_bus in active_bus_ids]
    return ParsedCase(data.name, data.source_version, data.baseMVA, buses, gens, branches, loads, shunts)
end

function _fallback_rate_a(branch::ParsedBranch, bus_by_id::Dict{Int,ParsedBus})
    theta_max = max(abs(branch.angmin), abs(branch.angmax))
    fr_vmax = bus_by_id[branch.f_bus].vmax
    to_vmax = bus_by_id[branch.t_bus].vmax
    zmag = hypot(branch.br_r, branch.br_x)
    ymag = iszero(zmag) ? 0.0 : inv(zmag)
    cmax = sqrt(fr_vmax^2 + to_vmax^2 - 2fr_vmax * to_vmax * cos(theta_max))
    return ymag * max(fr_vmax, to_vmax) * cmax
end

function _normalize_buses(buses::Vector{ParsedBus}, gens::Vector{ParsedGen})
    normalized = copy(buses)
    has_active_gen = Dict(bus.bus_i => false for bus in buses)
    biggest_gen_bus = nothing
    biggest_gen_pmax = -Inf
    for gen in gens
        has_active_gen[gen.gen_bus] = true
        if gen.pmax > biggest_gen_pmax
            biggest_gen_pmax = gen.pmax
            biggest_gen_bus = gen.gen_bus
        end
    end
    slack_found = false
    for i in eachindex(normalized)
        bus = normalized[i]
        has_gen = get(has_active_gen, bus.bus_i, false)
        bus_type = has_gen ? (bus.bus_type == 3 ? 3 : 2) : 1
        slack_found |= bus_type == 3
        normalized[i] = _with_bus_type(bus, bus_type)
    end
    if !slack_found && !isnothing(biggest_gen_bus)
        idx = findfirst(bus -> bus.bus_i == biggest_gen_bus, normalized)
        normalized[idx] = _with_bus_type(normalized[idx], 3)
    end
    return normalized
end

_with_bus_type(bus::ParsedBus, bus_type::Int) = ParsedBus(
    bus.bus_i, bus_type, bus.pd, bus.qd, bus.gs, bus.bs, bus.area,
    bus.vm, bus.va, bus.base_kv, bus.zone, bus.vmax, bus.vmin
)

function _normalize_angle_bounds(angmin::Float64, angmax::Float64)
    pad = deg2rad(60.0)
    angmin <= -pi / 2 && (angmin = -pad)
    angmax >= pi / 2 && (angmax = pad)
    iszero(angmin) && iszero(angmax) && return (-pad, pad)
    return angmin, angmax
end

function _validate_parsed_case(data::ParsedCase)
    isempty(data.bus) && throw(ArgumentError("MATPOWER file is missing mpc.bus"))
    isempty(data.gen) && throw(ArgumentError("MATPOWER file has no active generators"))
    isempty(data.branch) && throw(ArgumentError("MATPOWER file has no active branches"))
    _require_unique(getfield.(data.bus, :bus_i), "bus")
    _require_unique(getfield.(data.gen, :index), "generator")
    _require_unique(getfield.(data.branch, :index), "branch")
    bus_ids = Set(bus.bus_i for bus in data.bus)
    all(gen.gen_bus in bus_ids for gen in data.gen) || throw(ArgumentError("generator references an inactive or missing bus"))
    all(branch.f_bus in bus_ids && branch.t_bus in bus_ids for branch in data.branch) ||
        throw(ArgumentError("branch references an inactive or missing bus"))
    all(branch.rate_a > 0 for branch in data.branch) ||
        throw(ArgumentError("branches must have positive thermal limits after normalization"))
    return data
end

function _require_unique(ids, label)
    length(Set(ids)) == length(ids) || throw(ArgumentError("duplicate $label IDs are not supported"))
end
