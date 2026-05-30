using LazyArtifacts

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

const _SUPPORTED_MATPOWER_TABLES = Set(["bus", "gen", "branch", "gencost"])
const _IGNORED_MATPOWER_TABLES = Set(["areas", "bus_name"])
const _UNSUPPORTED_ELECTRICAL_TABLES = Set([
    "dcline", "dclinecost", "storage", "switch", "ne_branch",
    "branch_currents", "branch_oltc_pst",
])

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
with a format-specific package before constructing PowerDiff types.
"""
function parse_file(io::Union{IO,String}; library=nothing, validate=true, filetype="m", kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "unsupported parse_file keyword(s): $(join(string.(keys(kwargs)), ", "))"))
    resolved = io isa String ? _resolve_case_path(io, library) : io
    resolved_type = resolved isa String ? lowercase(splitext(resolved)[2]) : ".$(lowercase(filetype))"
    resolved_type == ".m" || throw(ArgumentError(
        "unsupported network file type $resolved_type; PowerDiff supports MATPOWER v2 .m files only"))
    return resolved isa String ? parse_matpower(resolved; validate) : parse_matpower(resolved; validate)
end

"""
    parse_matpower(io::IO; validate=true)
    parse_matpower(file::String; library=nothing, validate=true)

Parse MATPOWER v2 data into a normalized [`ParsedCase`](@ref).
"""
function parse_matpower(io::IO; validate=true)::ParsedCase
    parsed = _parse_matpower_typed(read(io, String))
    validate && (parsed = _normalize_parsed_case(parsed))
    validate && _validate_parsed_case(parsed)
    return parsed
end

function parse_matpower(file::String; library=nothing, validate=true)::ParsedCase
    resolved = _resolve_case_path(file, library)
    isfile(resolved) || throw(ArgumentError("invalid MATPOWER file $resolved"))
    return open(io -> parse_matpower(io; validate), resolved)
end

"""
    parse_matpower_struct(file::String; kwargs...)

Compatibility alias for [`parse_matpower`](@ref).
"""
parse_matpower_struct(file::String; kwargs...) = parse_matpower(file; kwargs...)

_resolve_case_path(path::AbstractString, ::Nothing) = String(path)
_resolve_case_path(path::AbstractString, library) = joinpath(get_path(library), path)

function _parse_matpower_typed(data_string::String)::ParsedCase
    clean = join(_strip_comment.(split(replace(data_string, "\r\n" => "\n"), '\n')), "\n")
    name = something(_capture(clean, r"function\s+[^=]+=\s*([A-Za-z_][A-Za-z0-9_]*)"), "no_name_found")
    source_version = something(_capture(clean, r"mpc\.version\s*=\s*'([^']*)'"), "0.0.0+")
    baseMVA_text = _capture(clean, r"mpc\.baseMVA\s*=\s*([^;]+)")
    isnothing(baseMVA_text) && throw(ArgumentError("MATPOWER file is missing mpc.baseMVA"))
    baseMVA = parse(Float64, strip(baseMVA_text))
    isfinite(baseMVA) && baseMVA > 0 || throw(ArgumentError("mpc.baseMVA must be finite and positive"))

    tables = Dict{String,String}()
    for assignment in eachmatch(r"mpc\.([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\[(.*?)\]\s*;"s, clean)
        key, body = assignment.captures
        key in _UNSUPPORTED_ELECTRICAL_TABLES && throw(ArgumentError(
            "MATPOWER table mpc.$key is not supported by PowerDiff"))
        key in _SUPPORTED_MATPOWER_TABLES && (tables[key] = body)
        key in _IGNORED_MATPOWER_TABLES && continue
    end

    buses = [_parse_bus_row(row, baseMVA) for row in _numeric_rows(get(tables, "bus", ""))]
    gens = [_parse_gen_row(row, i, baseMVA) for (i, row) in enumerate(_numeric_rows(get(tables, "gen", "")))]
    branches = [_parse_branch_row(row, i, baseMVA) for (i, row) in enumerate(_numeric_rows(get(tables, "branch", "")))]
    costs = [_parse_numeric_row(row) for row in _numeric_rows(get(tables, "gencost", ""))]
    gens = _apply_generator_costs(gens, costs, baseMVA)
    buses = _normalize_buses(buses, gens)
    loads, shunts = _build_bus_injections(buses)
    buses = _clear_bus_injections(buses)
    return ParsedCase(name, source_version, baseMVA, buses, gens, branches, loads, shunts)
end

_capture(text, pattern) = (m = match(pattern, text); isnothing(m) ? nothing : m.captures[1])

function _numeric_rows(body::AbstractString)
    rows = String[]
    for row in split(body, ';')
        normalized = strip(replace(row, '\n' => ' ', '\t' => ' ', ',' => ' '))
        isempty(normalized) || push!(rows, normalized)
    end
    return rows
end

_parse_numeric_row(row::AbstractString) = parse.(Float64, split(row))

function _require_columns(row, n::Int, table::String)
    values = _parse_numeric_row(row)
    length(values) >= n || throw(ArgumentError("mpc.$table row has $(length(values)) columns; expected at least $n"))
    all(isfinite, values) || throw(ArgumentError("mpc.$table contains a non-finite value"))
    return values
end

function _parse_bus_row(row::AbstractString, baseMVA::Float64)
    v = _require_columns(row, 13, "bus")
    return ParsedBus(
        Int(v[1]), Int(v[2]), v[3] / baseMVA, v[4] / baseMVA,
        v[5] / baseMVA, v[6] / baseMVA, Int(v[7]), v[8], deg2rad(v[9]),
        v[10], Int(v[11]), v[12], v[13]
    )
end

function _parse_gen_row(row::AbstractString, index::Int, baseMVA::Float64)
    v = _require_columns(row, 10, "gen")
    return ParsedGen(
        index, Int(v[1]), v[2] / baseMVA, v[3] / baseMVA,
        v[4] / baseMVA, v[5] / baseMVA, v[6], v[7], Int(v[8]),
        v[9] / baseMVA, v[10] / baseMVA, (0.0, 0.0, 0.0)
    )
end

function _parse_branch_row(row::AbstractString, index::Int, baseMVA::Float64)
    v = _require_columns(row, 13, "branch")
    angmin, angmax = _normalize_angle_bounds(deg2rad(v[12]), deg2rad(v[13]))
    return ParsedBranch(
        index, Int(v[1]), Int(v[2]), v[3], v[4], v[5],
        v[6] / baseMVA, v[7] / baseMVA, v[8] / baseMVA,
        v[9], deg2rad(v[10]), Int(v[11]), angmin, angmax
    )
end

function _apply_generator_costs(gens::Vector{ParsedGen}, rows::Vector{Vector{Float64}}, baseMVA::Float64)
    isempty(rows) && return gens
    length(rows) in (length(gens), 2length(gens)) || throw(ArgumentError(
        "mpc.gencost must contain one active-power row per generator, optionally followed by reactive-power rows"))
    out = copy(gens)
    for i in eachindex(gens)
        gen = gens[i]
        out[i] = ParsedGen(
            gen.index, gen.gen_bus, gen.pg, gen.qg, gen.qmax, gen.qmin, gen.vg,
            gen.mbase, gen.gen_status, gen.pmax, gen.pmin, _parse_cost_tuple(rows[i], baseMVA)
        )
    end
    return out
end

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

function _build_bus_injections(buses::Vector{ParsedBus})
    loads = ParsedLoad[]
    shunts = ParsedShunt[]
    for bus in buses
        status = bus.bus_type == 4 ? 0 : 1
        (iszero(bus.pd) && iszero(bus.qd)) || push!(loads, ParsedLoad(length(loads) + 1, bus.bus_i, bus.pd, bus.qd, status))
        (iszero(bus.gs) && iszero(bus.bs)) || push!(shunts, ParsedShunt(length(shunts) + 1, bus.bus_i, bus.gs, bus.bs, status))
    end
    return loads, shunts
end

function _clear_bus_injections(buses::Vector{ParsedBus})
    return [ParsedBus(
        bus.bus_i, bus.bus_type, 0.0, 0.0, 0.0, 0.0, bus.area,
        bus.vm, bus.va, bus.base_kv, bus.zone, bus.vmax, bus.vmin
    ) for bus in buses]
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

function _strip_comment(line::AbstractString)
    in_string = false
    for i in eachindex(line)
        line[i] == '\'' && (in_string = !in_string)
        line[i] == '%' && !in_string && return line[firstindex(line):prevind(line, i)]
    end
    return String(line)
end
