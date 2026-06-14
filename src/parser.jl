using LazyArtifacts
import PowerIO

# PowerIO is the parser and data layer. It parses MATPOWER/PSSE/etc. and
# `PowerIO.to_powerdata` returns normalized, per-unit, filtered data with the slack
# bus inferred, loads/shunts aggregated per bus, and polynomial costs rescaled.
# PowerDiff consumes that directly. The only logic kept here is OPF-solver modeling:
# polynomial cost interpretation, finite flow limits, default angle-difference
# bounds, and rejection of records PowerDiff does not model (storage, HVDC).

"""
    get_path(library::Symbol)

Resolve an artifact-backed library path owned by PowerDiff.
"""
function get_path(library::Symbol)
    library == :pglib && return joinpath(artifact"PGLib_opf", "pglib-opf-23.07")
    throw(ArgumentError("unsupported library $library"))
end

"""
    parse_file(io::Union{IO,String}; library=nothing, filetype="m") -> PowerIO.Network

Parse a MATPOWER v2 `.m` file into a `PowerIO.Network`.

PowerDiff intentionally supports MATPOWER files only. Convert other formats before
constructing PowerDiff types. Pass the result to [`DCNetwork`](@ref) / [`ACNetwork`](@ref).
"""
function parse_file(io::Union{IO,String}; library=nothing, filetype="m", kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "unsupported parse_file keyword(s): $(join(string.(keys(kwargs)), ", "))"))
    resolved = io isa String ? _resolve_case_path(io, library) : io
    resolved_type = resolved isa String ? lowercase(splitext(resolved)[2]) : ".$(lowercase(filetype))"
    resolved_type == ".m" || throw(ArgumentError(
        "unsupported network file type $resolved_type; PowerDiff supports MATPOWER v2 .m files only"))
    return parse_matpower(resolved)
end

"""
    parse_matpower(io::IO) -> PowerIO.Network
    parse_matpower(file::String; library=nothing) -> PowerIO.Network

Parse MATPOWER v2 data into a `PowerIO.Network`.
"""
function parse_matpower(io::IO)
    try
        return PowerIO.parse_str(read(io, String), "matpower")
    catch e
        e isa ArgumentError && rethrow()
        throw(ArgumentError("PowerDiff.parse_matpower: " * sprint(showerror, e)))
    end
end

function parse_matpower(file::String; library=nothing)
    resolved = _resolve_case_path(file, library)
    isfile(resolved) || throw(ArgumentError("invalid MATPOWER file $resolved"))
    try
        return PowerIO.parse_file(String(resolved))
    catch e
        e isa ArgumentError && rethrow()
        throw(ArgumentError("PowerDiff.parse_matpower: " * sprint(showerror, e)))
    end
end

"""
    parse_matpower_struct(file::String; kwargs...)

Compatibility alias for [`parse_matpower`](@ref).
"""
parse_matpower_struct(file::String; kwargs...) = parse_matpower(file; kwargs...)

_resolve_case_path(path::AbstractString, ::Nothing) = String(path)
_resolve_case_path(path::AbstractString, library) = joinpath(get_path(library), path)

"""
    _network_data(net::PowerIO.Network) -> NamedTuple

Normalized PowerDiff network tables built from `PowerIO.to_powerdata(net)`.

`to_powerdata` already does per-unit scaling, status/isolated filtering, slack
inference, per-bus load/shunt aggregation, and `base^(n-i)` cost rescaling, and it
returns dense file-order indices. This re-keys the bus references back to original
bus ids (so [`IDMapping`](@ref)'s sorted ordering is preserved) and applies
PowerDiff's OPF modeling: polynomial cost (rejecting PWL/quartic), a finite flow
limit fallback when `rate_a == 0`, default angle-difference bounds, and rejection of
storage / HVDC records that PowerDiff does not model.

The returned `bus`/`gen`/`branch` rows mirror the field names the network
constructors expect, with loads/shunts already folded into per-bus `pd/qd/gs/bs`.
"""
function _network_data(net)
    isempty(PowerIO.hvdc(net)) || throw(ArgumentError(
        "PowerDiff does not support HVDC/dcline records; remove or convert dcline before parsing"))
    # Surface malformed input (e.g. non-finite fields PowerIO leaves as `nothing`) as an
    # ArgumentError rather than a downstream MethodError.
    pd = try
        PowerIO.to_powerdata(net)
    catch e
        e isa ArgumentError && rethrow()
        throw(ArgumentError("PowerDiff: could not normalize the network: " * sprint(showerror, e)))
    end
    isempty(pd.storage) || throw(ArgumentError(
        "PowerDiff does not support storage records; remove or convert storage before parsing"))
    isempty(pd.bus) && throw(ArgumentError("MATPOWER file is missing mpc.bus"))
    isempty(pd.gen) && throw(ArgumentError("MATPOWER file has no active generators"))
    isempty(pd.branch) && throw(ArgumentError("MATPOWER file has no active branches"))

    orig = [Int(b.bus_i) for b in pd.bus]      # file-order index -> original bus id
    vmax = [Float64(b.vmax) for b in pd.bus]   # file-order index -> bus vmax

    buses = [(; bus_i=orig[i], bus_type=Int(b.type),
              pd=Float64(b.pd), qd=Float64(b.qd), gs=Float64(b.gs), bs=Float64(b.bs),
              vm=Float64(b.vm), va=Float64(b.va), vmin=Float64(b.vmin), vmax=Float64(b.vmax))
             for (i, b) in enumerate(pd.bus)]

    # to_powerdata's gen `c`/`n` mangles costs declared with ncost > 3 (it keeps the
    # 3 highest-degree coefficients with the wrong exponents and drops the rest), so a
    # quadratic padded to ncost=5 — e.g. MATPOWER case14 — loses its linear term. Read
    # the raw cost coefficients from PowerIO instead, aligned to to_powerdata's kept
    # gens by the same keep filter (in service, bus not isolated). Track as a PowerIO
    # enhancement: have to_powerdata expose costs losslessly.
    kept_bus = Set(Int(b.id) for b in PowerIO.buses(net) if String(b.kind) != "ISOLATED")
    raw_costs = [_cost_tuple(g.cost, Float64(pd.baseMVA))
                 for g in PowerIO.generators(net) if g.in_service && Int(g.bus) in kept_bus]
    length(raw_costs) == length(pd.gen) ||
        throw(ArgumentError("generator cost alignment mismatch ($(length(raw_costs)) vs $(length(pd.gen)))"))

    gens = [(; index=j, gen_bus=orig[g.bus],
             pg=Float64(g.pg), qg=Float64(g.qg), qmin=Float64(g.qmin), qmax=Float64(g.qmax),
             vg=Float64(g.vg), pmin=Float64(g.pmin), pmax=Float64(g.pmax), cost=raw_costs[j])
            for (j, g) in enumerate(pd.gen)]

    # PowerDiff's OPF needs a reference bus. to_powerdata only reassigns the slack when
    # an explicit REF was demoted; if the source marks none at all, promote the bus with
    # the largest generator (matching the previous parser). PowerIO issue: infer a
    # reference when the file marks none.
    if !any(b -> b.bus_type == 3, buses)
        slack_bus, best_pmax = 0, -Inf
        for g in gens
            if g.pmax > best_pmax
                best_pmax = g.pmax
                slack_bus = g.gen_bus
            end
        end
        slack_bus == 0 ||
            (buses = [b.bus_i == slack_bus ? (; b..., bus_type=3) : b for b in buses])
    end

    branches = NamedTuple[]
    for (l, br) in enumerate(pd.branch)
        angmin, angmax = _normalize_angle_bounds(Float64(br.angmin), Float64(br.angmax))
        rate_a = br.rate_a > 0 ? Float64(br.rate_a) :
                 _fallback_rate_a(Float64(br.br_r), Float64(br.br_x), angmin, angmax,
                                  vmax[br.f_bus], vmax[br.t_bus])
        push!(branches, (; index=l, f_bus=orig[br.f_bus], t_bus=orig[br.t_bus],
              br_r=Float64(br.br_r), br_x=Float64(br.br_x), br_b=Float64(br.b_fr + br.b_to),
              rate_a=rate_a, rate_b=Float64(br.rate_b), rate_c=Float64(br.rate_c),
              tap=Float64(br.tap), shift=Float64(br.shift), angmin=angmin, angmax=angmax))
    end
    all(br.rate_a > 0 for br in branches) || throw(ArgumentError(
        "branches must have positive thermal limits after normalization"))

    return (; name=PowerIO.network_name(net), baseMVA=Float64(pd.baseMVA),
            bus=buses, gen=gens, branch=branches)
end

# Convert a PowerIO raw generator cost (model, ncost, coeffs) to PowerDiff's
# (quadratic, linear, constant) tuple, in per-unit. Rescale each coefficient by
# base^(n-i), drop leading zeros (so a quadratic padded to ncost=5 collapses to a
# real quadratic), and reject piecewise-linear and higher-than-quadratic costs.
function _cost_tuple(cost, base::Float64)
    cost === nothing && return (0.0, 0.0, 0.0)
    Int(cost.model) == 2 || throw(ArgumentError("only polynomial mpc.gencost model 2 is supported"))
    n = Int(cost.ncost)
    n >= 1 || throw(ArgumentError("mpc.gencost must declare at least one coefficient"))
    cf = collect(cost.coeffs)
    coeffs = [base^(n - i) * Float64(cf[i]) for i in 1:n]
    while length(coeffs) > 1 && iszero(first(coeffs))
        popfirst!(coeffs)
    end
    length(coeffs) <= 3 || throw(ArgumentError("only constant, linear, and quadratic generator costs are supported"))
    return length(coeffs) == 3 ? (coeffs[1], coeffs[2], coeffs[3]) :
           length(coeffs) == 2 ? (0.0, coeffs[1], coeffs[2]) :
                                 (0.0, 0.0, coeffs[1])
end

# PowerDiff's OPF needs a finite thermal limit on every branch. When MATPOWER leaves
# rate_a == 0 (unlimited), synthesize one from the bus voltage limits and the branch
# impedance / angle window, matching the previous native parser.
function _fallback_rate_a(r::Float64, x::Float64, angmin::Float64, angmax::Float64,
                          fr_vmax::Float64, to_vmax::Float64)
    theta_max = max(abs(angmin), abs(angmax))
    zmag = hypot(r, x)
    ymag = iszero(zmag) ? 0.0 : inv(zmag)
    cmax = sqrt(fr_vmax^2 + to_vmax^2 - 2fr_vmax * to_vmax * cos(theta_max))
    return ymag * max(fr_vmax, to_vmax) * cmax
end

# Default angle-difference bounds (radians in, radians out). MATPOWER angmin == angmax
# == 0 means unbounded; treat ±90°-or-wider and the zero case as a ±60° window, the
# MATPOWER/PowerModels convention. PowerIO's `to_powerdata` already converts to radians.
function _normalize_angle_bounds(angmin::Float64, angmax::Float64)
    pad = deg2rad(60.0)
    angmin <= -pi / 2 && (angmin = -pad)
    angmax >= pi / 2 && (angmax = pad)
    iszero(angmin) && iszero(angmax) && return (-pad, pad)
    return angmin, angmax
end
