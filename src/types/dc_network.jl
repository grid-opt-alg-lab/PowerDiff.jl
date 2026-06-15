# Copyright 2026 Samuel Talkington and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================================================
# DCNetwork: DC Network Data Structure
# =============================================================================
#
# DC network representation for B-theta OPF formulation with susceptance-weighted
# Laplacian B = A' * Diag(-b .* sw) * A.

"""
    DCNetwork <: AbstractPowerNetwork

DC network data for B-theta OPF formulation. Uses susceptance-weighted Laplacian
`B = A' * Diagonal(-b .* sw) * A` which preserves graphical structure for
topology sensitivity analysis.

# Fields
- `n`, `m`, `k`: Number of buses, branches, and generators
- `A`: Branch-bus incidence matrix (m x n)
- `G_inc`: Generator-bus incidence matrix (n x k)
- `b`: Branch susceptances (imaginary part of 1/z)
- `sw`: Branch switching states (1 = closed, 0 = open)
- `fmax`, `gmax`, `gmin`: Flow and generation limits
- `angmax`, `angmin`: Phase angle difference limits
- `cq`, `cl`: Quadratic and linear generation cost coefficients
- `c_shed`: Load-shedding cost per bus (penalty for involuntary load curtailment)
- `ref_bus`: Reference bus index (phase angle = 0)
- `tau`: Regularization parameter for strong convexity
- `id_map`: Bidirectional mapping between original and sequential element IDs
- `demand`: Real power demand aggregated per bus
- `pg_init`: Initial real generation aggregated per bus
"""
struct DCNetwork <: AbstractPowerNetwork
    n::Int
    m::Int
    k::Int
    A::SparseMatrixCSC{Float64,Int}
    G_inc::SparseMatrixCSC{Float64,Int}
    b::Vector{Float64}
    sw::Vector{Float64}
    fmax::Vector{Float64}
    gmax::Vector{Float64}
    gmin::Vector{Float64}
    angmax::Vector{Float64}
    angmin::Vector{Float64}
    cq::Vector{Float64}
    cl::Vector{Float64}
    c_shed::Vector{Float64}
    demand::Vector{Float64}
    pg_init::Vector{Float64}
    ref_bus::Int
    tau::Float64
    id_map::IDMapping
end

# =============================================================================
# DC Power Flow and OPF State Types
# =============================================================================

"""
    DCOPFSolution <: AbstractOPFSolution

Solution container for DC OPF problem, storing both primal and dual variables.

# Fields
- `va`: Phase angles at each bus
- `pg`: Generator outputs
- `f`: Line flows
- `psh`: Load shedding at each bus
- `nu_bal`: Power balance dual variables (nodal, used for LMP computation)
- `nu_flow`: Flow definition dual variables
- `lam_ub`, `lam_lb`: Line flow upper/lower bound duals
- `rho_ub`, `rho_lb`: Generator upper/lower bound duals
- `mu_lb`, `mu_ub`: Load shedding lower/upper bound duals
- `gamma_lb`, `gamma_ub`: Phase angle difference lower/upper bound duals
- `eta_ref`: Reference bus constraint dual (`va[ref_bus] == 0`)
- `objective`: Optimal objective value
- `B_r_factor`: Cached factorization of reduced susceptance matrix `B[non_ref, non_ref]`
"""
struct DCOPFSolution{F<:Factorization{Float64}} <: AbstractOPFSolution
    va::Vector{Float64}
    pg::Vector{Float64}
    f::Vector{Float64}
    psh::Vector{Float64}
    nu_bal::Vector{Float64}
    nu_flow::Vector{Float64}
    lam_ub::Vector{Float64}
    lam_lb::Vector{Float64}
    rho_ub::Vector{Float64}
    rho_lb::Vector{Float64}
    mu_lb::Vector{Float64}
    mu_ub::Vector{Float64}
    gamma_lb::Vector{Float64}
    gamma_ub::Vector{Float64}
    eta_ref::Float64
    objective::Float64
    B_r_factor::F
end

"""
    DCPowerFlowState{F} <: AbstractPowerFlowState

DC power flow solution (phase angles from reduced-Laplacian solve, no optimization).
Supports both generation and demand for flexible sensitivity analysis.

Unlike DCOPFSolution, this represents a simple power flow solution
`θ_r = B_r \\ p_r` where `B_r` is the susceptance matrix with the reference bus row and
column deleted (invertible for a connected network), without optimal dispatch or
constraint handling.

# Fields
- `net`: DCNetwork data
- `va`: Phase angles (rad), with `va[ref_bus] = 0`
- `p`: Net injection vector (p = pg - d)
- `pg`: Generation vector
- `d`: Demand vector
- `f`: Branch flows (computed from va)
- `B_r_factor`: Factorization of `B[non_ref, non_ref]` (Cholesky for inductive networks, LU fallback)
- `non_ref`: Indices of non-reference buses
"""
struct DCPowerFlowState{F<:Factorization{Float64}} <: AbstractPowerFlowState
    net::DCNetwork
    va::Vector{Float64}
    p::Vector{Float64}
    pg::Vector{Float64}
    d::Vector{Float64}
    f::Vector{Float64}
    B_r_factor::F
    non_ref::Vector{Int}
end

# =============================================================================
# Constants
# =============================================================================

const DEFAULT_TAU = 1e-2

# Shedding cost = multiplier × peak marginal generation cost, so the solver
# only sheds when generation capacity is physically insufficient or flow
# constraints prevent delivery.
const DEFAULT_SHED_COST_MULTIPLIER = 10

# =============================================================================
# PowerIO input and network-table construction
# =============================================================================
#
# PowerIO is the parser and data layer. `PowerIO.parse_*` reads MATPOWER/PSSE/etc.
# and `PowerIO.to_powerdata` returns normalized, per-unit, status/isolated-filtered
# data with the reference bus inferred (`type == 3`), source bus ids on `bus_i`,
# loads/shunts aggregated per bus, and polynomial costs collapsed and rescaled.
# These thin wrappers return a `PowerIO.Network`, and `_network_data` turns one into
# the network tables the DCNetwork and ACNetwork constructors consume. The only
# logic beyond re-keying to source bus ids is the OPF solver modeling PowerIO leaves
# to the consumer: polynomial cost interpretation, finite flow limits, default
# angle-difference bounds, and rejection of records PowerDiff does not model.

"""
    parse_file(path::String; library=nothing, from=nothing, filetype=nothing) -> PowerIO.Network
    parse_file(io::IO; from="matpower", filetype=nothing) -> PowerIO.Network

Parse a supported PowerIO network file into a `PowerIO.Network`.

For paths, PowerIO infers the format from the extension unless `from` is given.
For streams, pass `from` (or `filetype`) because there is no extension. JSON formats
are ambiguous; use `from=:egret` or `from=:powermodels`.

Supported format tokens are PowerIO's tokens: `:matpower` / `:m`, `:psse` / `:raw`,
`:powerworld` / `:aux`, `:powermodels`, and `:egret`.
Pass the result to [`DCNetwork`](@ref) / [`ACNetwork`](@ref).
"""
function parse_file(io::Union{IO,String}; library=nothing, filetype=nothing, from=nothing, kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "unsupported parse_file keyword(s): $(join(string.(keys(kwargs)), ", "))"))
    fmt = _powerio_format_hint(from, filetype)
    if io isa String
        resolved = _resolve_case_path(io, library)
        try
            return isnothing(fmt) ? PowerIO.parse_file(resolved) : PowerIO.parse_file(resolved; from=fmt)
        catch e
            e isa ArgumentError && rethrow()
            throw(ArgumentError("PowerDiff.parse_file: " * sprint(showerror, e)))
        end
    else
        fmt = isnothing(fmt) ? "matpower" : fmt
        try
            return PowerIO.parse_file(io, fmt)
        catch e
            e isa ArgumentError && rethrow()
            throw(ArgumentError("PowerDiff.parse_file: " * sprint(showerror, e)))
        end
    end
end

"""
    parse_matpower(io::IO) -> PowerIO.Network
    parse_matpower(file::String; library=nothing) -> PowerIO.Network

Parse MATPOWER v2 data into a `PowerIO.Network`.
"""
function parse_matpower(io::IO)
    try
        return PowerIO.parse_file(io, "matpower")
    catch e
        e isa ArgumentError && rethrow()
        throw(ArgumentError("PowerDiff.parse_matpower: " * sprint(showerror, e)))
    end
end

function parse_matpower(file::String; library=nothing)
    resolved = _resolve_case_path(file, library)
    isfile(resolved) || throw(ArgumentError("invalid MATPOWER file $resolved"))
    try
        return PowerIO.parse_file(String(resolved); from="matpower")
    catch e
        e isa ArgumentError && rethrow()
        throw(ArgumentError("PowerDiff.parse_matpower: " * sprint(showerror, e)))
    end
end

"""
    parse_matpower_struct(file::String; library=nothing)

Compatibility alias for [`parse_matpower`](@ref).
"""
parse_matpower_struct(file::String; library=nothing) = parse_matpower(file; library=library)

_resolve_case_path(path::AbstractString, ::Nothing) = String(path)
_resolve_case_path(path::AbstractString, library) = joinpath(get_path(library), path)

_powerio_format_hint(::Nothing, ::Nothing) = nothing
_powerio_format_hint(from, ::Nothing) = _format_token(from)
_powerio_format_hint(::Nothing, filetype) = _format_token(filetype)
function _powerio_format_hint(from, filetype)
    f1 = _format_token(from)
    f2 = _format_token(filetype)
    f1 == f2 || throw(ArgumentError("conflicting parse format hints: from=$from and filetype=$filetype"))
    return f1
end

function _format_token(x)
    s = lowercase(String(x))
    startswith(s, ".") && (s = s[2:end])
    s == "json" && throw(ArgumentError(
        "JSON input is ambiguous; pass from=:egret or from=:powermodels"))
    s in ("m", "matpower") && return "matpower"
    s in ("raw", "psse") && return "psse"
    s in ("aux", "powerworld") && return "powerworld"
    s in ("pm", "powermodels", "powermodels-json") && return "powermodels-json"
    s in ("egret", "egret-json") && return "egret-json"
    throw(ArgumentError(
        "unsupported network format $x (expected matpower, psse/raw, powerworld/aux, powermodels-json, or egret-json)"))
end

"""
    _network_data(net::PowerIO.Network) -> NamedTuple

Build PowerDiff network tables from `PowerIO.to_powerdata(net)`.

`to_powerdata` does per-unit scaling, status/isolated filtering, per-bus
load/shunt aggregation, reference-bus inference (`type == 3`), source bus ids on
`bus_i`, and polynomial cost collapse/rescaling, returning dense file-order rows.
This adapter keys bus references back to source bus ids (so [`IDMapping`](@ref)'s
sorted ordering is preserved) and applies the OPF modeling PowerIO leaves to the
consumer: polynomial cost interpretation (rejecting PWL and higher-than-quadratic),
a finite flow-limit fallback when `rate_a == 0`, default angle-difference bounds,
and rejection of storage / HVDC records that PowerDiff does not model.

The returned `bus`/`gen`/`branch` rows mirror the field names the network
constructors expect, with loads/shunts already folded into per-bus `pd/qd/gs/bs`.
`shunt` re-exposes those bus shunts as a table (one `(; index, shunt_bus, gs, bs)`
record per bus with a nonzero shunt admittance) for callers that want shunt records.

Bus rows carry the source bus id on `bus_i`, so [`IDMapping`](@ref)`.bus_ids`
(and any bus-indexed sensitivity `row_to_id`) map back to the input network.
Generator and branch `index` values are source row numbers among the unfiltered
PowerIO rows, so out-of-service rows leave gaps instead of renumbering active rows.
"""
function _network_data(net)
    # Reject records PowerDiff does not model. Both guards read the raw network so
    # they stay consistent: to_powerdata's filtered output drops out-of-service
    # records, which would silently accept a file that declares them.
    isempty(PowerIO.hvdc(net)) || throw(ArgumentError(
        "PowerDiff does not support HVDC/dcline records; remove or convert dcline before parsing"))
    isempty(PowerIO.storage(net)) || throw(ArgumentError(
        "PowerDiff does not support storage records; remove or convert storage before parsing"))
    pd = PowerIO.to_powerdata(net)
    isempty(pd.bus) && throw(ArgumentError("network has no active buses"))
    isempty(pd.gen) && throw(ArgumentError("network has no active generators"))
    isempty(pd.branch) && throw(ArgumentError("network has no active branches"))

    orig = [Int(b.bus_i) for b in pd.bus]   # dense file-order index -> source bus id
    gen_source_rows, branch_source_rows = _active_source_rows(net, pd)

    buses = [(; bus_i=orig[i], bus_type=Int(b.type),
              pd=Float64(b.pd), qd=Float64(b.qd), gs=Float64(b.gs), bs=Float64(b.bs),
              vm=Float64(b.vm), va=Float64(b.va), vmin=Float64(b.vmin), vmax=Float64(b.vmax))
             for (i, b) in enumerate(pd.bus)]

    # Costs come straight from to_powerdata's gen rows (already per-unit and
    # right-aligned). Map dense `gen.bus` to the source bus id via `orig`.
    gens = [(; index=gen_source_rows[j], gen_bus=orig[g.bus],
             pg=Float64(g.pg), qg=Float64(g.qg), qmin=Float64(g.qmin), qmax=Float64(g.qmax),
             vg=Float64(g.vg), pmin=Float64(g.pmin), pmax=Float64(g.pmax), cost=_poly_cost(g))
            for (j, g) in enumerate(pd.gen)]

    branches = [_branch_row(branch_source_rows[l], br, orig, buses) for (l, br) in enumerate(pd.branch)]
    all(br.rate_a > 0 for br in branches) || throw(ArgumentError(
        "branches must have positive thermal limits after normalization"))

    # to_powerdata folds shunts into per-bus gs/bs (which the constructors consume).
    # Re-expose them as a table, one record per bus with a nonzero shunt admittance,
    # for callers that want shunt records back.
    shunt_buses = [b for b in buses if b.gs != 0.0 || b.bs != 0.0]
    shunts = [(; index=i, shunt_bus=b.bus_i, gs=b.gs, bs=b.bs) for (i, b) in enumerate(shunt_buses)]

    return (; name=PowerIO.network_name(net), baseMVA=Float64(pd.baseMVA),
            bus=buses, gen=gens, branch=branches, shunt=shunts)
end

function _active_source_rows(net, pd)
    raw = PowerIO.to_powerdata(net; filtered=false)
    kept_bus_ids = Set(Int(b.bus_i) for b in pd.bus)
    raw_bus_id = Dict(Int(b.i) => Int(b.bus_i) for b in raw.bus)

    gen_rows = Int[]
    for (row, gen) in enumerate(raw.gen)
        status = hasproperty(gen, :status) ? Int(gen.status) != 0 : true
        bus_id = get(raw_bus_id, Int(gen.bus), nothing)
        status && bus_id in kept_bus_ids && push!(gen_rows, row)
    end

    branch_rows = Int[]
    for (row, br) in enumerate(raw.branch)
        status = hasproperty(br, :status) ? Int(br.status) != 0 : true
        f_id = get(raw_bus_id, Int(br.f_bus), nothing)
        t_id = get(raw_bus_id, Int(br.t_bus), nothing)
        status && f_id in kept_bus_ids && t_id in kept_bus_ids && push!(branch_rows, row)
    end

    length(gen_rows) == length(pd.gen) || throw(ArgumentError(
        "PowerDiff could not map active generators back to source rows"))
    length(branch_rows) == length(pd.branch) || throw(ArgumentError(
        "PowerDiff could not map active branches back to source rows"))

    return gen_rows, branch_rows
end

# Build one PowerDiff branch row from a to_powerdata branch: map dense f_bus/t_bus to
# source ids, default the angle window, and synthesize a finite rate_a when the
# source leaves it at 0 (unlimited), using the endpoint buses' vmax limits.
function _branch_row(l, br, orig, buses)
    angmin, angmax = _normalize_angle_bounds(Float64(br.angmin), Float64(br.angmax))
    rate_a = br.rate_a > 0 ? Float64(br.rate_a) :
             _fallback_rate_a(Float64(br.br_r), Float64(br.br_x), angmin, angmax,
                              buses[br.f_bus].vmax, buses[br.t_bus].vmax)
    return (; index=l, f_bus=orig[br.f_bus], t_bus=orig[br.t_bus],
            br_r=Float64(br.br_r), br_x=Float64(br.br_x), br_b=Float64(br.b_fr + br.b_to),
            rate_a=rate_a, rate_b=Float64(br.rate_b), rate_c=Float64(br.rate_c),
            tap=Float64(br.tap), shift=Float64(br.shift), angmin=angmin, angmax=angmax)
end

# Interpret a PowerIO gen row's polynomial cost as PowerDiff's (quadratic, linear,
# constant) tuple. to_powerdata returns polynomial (model 2) costs as a right-aligned,
# per-unit (cq, cl, cc) triple and rejects higher-than-quadratic itself. A generator
# with no gencost row comes back as `model_poly == false` with `n == 0` (cost-free);
# piecewise-linear (model 1) is `model_poly == false` with `n > 0` and is unsupported.
function _poly_cost(g)
    if !g.model_poly
        Int(g.n) == 0 && return (0.0, 0.0, 0.0)
        throw(ArgumentError(
            "piecewise linear generator costs are not supported; convert model 1 costs to polynomial model 2 before parsing"))
    end
    # to_powerdata right-aligns the (quadratic, linear, constant) triple, but guard the
    # indexing so a model-2 cost shorter than 3 terms (purely linear/constant) zero-pads
    # the missing leading coefficients instead of throwing a BoundsError.
    c = g.c
    cq = length(c) >= 3 ? Float64(c[end-2]) : 0.0
    cl = length(c) >= 2 ? Float64(c[end-1]) : 0.0
    cc = length(c) >= 1 ? Float64(c[end]) : 0.0
    return (cq, cl, cc)
end

# PowerDiff's OPF needs a finite thermal limit on every branch. When the source
# leaves rate_a == 0 (unlimited), synthesize one from the bus voltage limits and
# the branch impedance / angle window, matching the previous native parser.
function _fallback_rate_a(r::Float64, x::Float64, angmin::Float64, angmax::Float64,
                          fr_vmax::Float64, to_vmax::Float64)
    theta_max = max(abs(angmin), abs(angmax))
    zmag = hypot(r, x)
    ymag = iszero(zmag) ? 0.0 : inv(zmag)
    cmax = sqrt(fr_vmax^2 + to_vmax^2 - 2fr_vmax * to_vmax * cos(theta_max))
    return ymag * max(fr_vmax, to_vmax) * cmax
end

# Default angle-difference bounds (radians in, radians out). MATPOWER angmin == angmax
# == 0 means unbounded; treat ±90° or wider and the zero case as a ±60° window, the
# MATPOWER/PowerModels convention. PowerIO's `to_powerdata` already converts to radians.
function _normalize_angle_bounds(angmin::Float64, angmax::Float64)
    pad = deg2rad(60.0)
    angmin <= -pi / 2 && (angmin = -pad)
    angmax >= pi / 2 && (angmax = pad)
    iszero(angmin) && iszero(angmax) && return (-pad, pad)
    return angmin, angmax
end

# =============================================================================
# DCNetwork Constructors
# =============================================================================

"""
    DCNetwork(net::Dict; kwargs...)

Reject the removed dictionary API with a migration hint.
"""
function DCNetwork(net::Dict{String,<:Any}; kwargs...)
    throw(ArgumentError("dictionary constructors were removed; parse a network file with PowerDiff.parse_file"))
end

"""
    DCNetwork(net::PowerIO.Network; tau=DEFAULT_TAU, ref_bus=nothing)

Construct a DCNetwork from a parsed PowerIO network.

# Example
```julia
net = parse_file("case14.m")
dc_net = DCNetwork(net)
```
"""
DCNetwork(net::PowerIO.Network; tau::Float64=DEFAULT_TAU, ref_bus::Union{Nothing,Int}=nothing) =
    DCNetwork(_network_data(net); tau=tau, ref_bus=ref_bus)

# Build from PowerDiff network tables (see `_network_data`). The `PowerIO.Network`
# method runs PowerDiff's modeling deltas; this assumes the tables are already
# normalized, so programmatic callers can supply ready values directly.
function DCNetwork(data::NamedTuple; tau::Float64=DEFAULT_TAU, ref_bus::Union{Nothing,Int}=nothing)
    id_map = IDMapping(data)

    n = length(id_map.bus_ids)
    m = length(id_map.branch_ids)
    k = length(id_map.gen_ids)
    bus_tbl = Dict(bus.bus_i => bus for bus in data.bus)
    branch_tbl = Dict(branch.index => branch for branch in data.branch)
    gen_tbl = Dict(gen.index => gen for gen in data.gen)

    # Incidence matrix A (m × n) from active branches using id_map translation
    A = spzeros(m, n)
    for orig_id in id_map.branch_ids
        br = branch_tbl[orig_id]
        row = id_map.branch_to_idx[orig_id]
        f_col = id_map.bus_to_idx[br.f_bus]
        t_col = id_map.bus_to_idx[br.t_bus]
        A[row, f_col] = 1.0
        A[row, t_col] = -1.0
    end

    # Generator-bus incidence matrix G_inc (n × k)
    G_inc = spzeros(n, k)
    for orig_id in id_map.gen_ids
        gen = gen_tbl[orig_id]
        col = id_map.gen_to_idx[orig_id]
        row = id_map.bus_to_idx[gen.gen_bus]
        G_inc[row, col] = 1.0
    end

    # Branch susceptances: b = imag(1/z)
    b = zeros(m)
    for orig_id in id_map.branch_ids
        br = branch_tbl[orig_id]
        idx = id_map.branch_to_idx[orig_id]
        r = br.br_r
        x = br.br_x
        z2 = r^2 + x^2
        if z2 > 1e-10
            b[idx] = -x / z2
        else
            _SILENCE_WARNINGS[] || @warn "Branch $(orig_id) has near-zero impedance (|z|² = $(z2)); treating as open (zero admittance)."
        end
    end

    # All branches initially active
    sw = ones(m)

    # Limits (iterate in sequential order via sorted IDs)
    fmax = [branch_tbl[id_map.branch_ids[i]].rate_a for i in 1:m]
    gmax = [gen_tbl[id_map.gen_ids[i]].pmax for i in 1:k]
    gmin = [gen_tbl[id_map.gen_ids[i]].pmin for i in 1:k]

    # Phase angle difference limits
    angmax = [branch_tbl[id_map.branch_ids[i]].angmax for i in 1:m]
    angmin = [branch_tbl[id_map.branch_ids[i]].angmin for i in 1:m]

    # Cost coefficients (assumes polynomial cost with at least 2 terms)
    cq = [gen_tbl[id_map.gen_ids[i]].cost[1] for i in 1:k]
    cl = [gen_tbl[id_map.gen_ids[i]].cost[2] for i in 1:k]
    demand = calc_demand_vector(data, id_map)
    pg_init = _calc_generation_vector(data, id_map)

    # Load-shedding cost: high penalty to discourage shedding when feasible.
    # Guard the reduction so a generator-free network (valid for pure DC power flow
    # built via the NamedTuple constructor) falls back to a unit marginal cost
    # instead of `maximum` throwing on an empty collection.
    marginal_cost_ub = k == 0 ? 1.0 : max(maximum(2cq .* gmax .+ cl), 1.0)
    c_shed = fill(DEFAULT_SHED_COST_MULTIPLIER * marginal_cost_ub, n)

    # Reference bus (translate original ID to sequential index)
    if isnothing(ref_bus)
        ref_candidates = [id for id in id_map.bus_ids if bus_tbl[id].bus_type == 3]
        if isempty(ref_candidates)
            _SILENCE_WARNINGS[] || @warn "No reference bus (type 3) in the network; defaulting to bus $(id_map.bus_ids[1]) as slack. Pass `ref_bus` to choose explicitly."
            orig_ref = id_map.bus_ids[1]
        else
            orig_ref = ref_candidates[1]
        end
        ref_bus = id_map.bus_to_idx[orig_ref]
    else
        # If user provided an original bus ID, translate it; validate the result
        if haskey(id_map.bus_to_idx, ref_bus)
            ref_bus = id_map.bus_to_idx[ref_bus]
        elseif !(1 <= ref_bus <= n)
            throw(ArgumentError(
                "ref_bus=$ref_bus is not a valid bus ID ($(id_map.bus_ids)) or index (1:$n)"))
        end
    end

    return DCNetwork(n, m, k, A, G_inc, b, sw, fmax, gmax, gmin, angmax, angmin,
                     cq, cl, c_shed, demand, pg_init, ref_bus, tau, id_map)
end

"""
    DCNetwork(n, m, k, A, G_inc, b; kwargs...)

Direct constructor for DCNetwork with matrices and vectors.
Useful for building networks programmatically.
"""
function DCNetwork(
    n::Int, m::Int, k::Int,
    A::AbstractMatrix, G_inc::AbstractMatrix, b::AbstractVector;
    sw::AbstractVector=ones(m),
    fmax::AbstractVector=fill(Inf, m),
    gmax::AbstractVector=fill(Inf, k),
    gmin::AbstractVector=zeros(k),
    angmax::AbstractVector=fill(π, m),
    angmin::AbstractVector=fill(-π, m),
    cq::AbstractVector=zeros(k),
    cl::AbstractVector=zeros(k),
    c_shed::AbstractVector=fill(1e4, n),
    demand::AbstractVector=zeros(n),
    pg_init::AbstractVector=zeros(n),
    ref_bus::Int=1,
    tau::Float64=DEFAULT_TAU
)
    length(c_shed) == n || throw(DimensionMismatch("c_shed length $(length(c_shed)) must match number of buses $n"))
    length(demand) == n || throw(DimensionMismatch("demand length $(length(demand)) must match number of buses $n"))
    length(pg_init) == n || throw(DimensionMismatch("pg_init length $(length(pg_init)) must match number of buses $n"))
    all(c_shed .> 0) || throw(ArgumentError("c_shed must be strictly positive at all buses"))
    return DCNetwork(
        n, m, k,
        sparse(Float64.(A)), sparse(Float64.(G_inc)),
        Float64.(b), Float64.(sw),
        Float64.(fmax), Float64.(gmax), Float64.(gmin),
        Float64.(angmax), Float64.(angmin),
        Float64.(cq), Float64.(cl),
        Float64.(c_shed),
        Float64.(demand), Float64.(pg_init),
        ref_bus, tau,
        IDMapping(n, m, k)
    )
end

# =============================================================================
# DCNetwork Helper Functions
# =============================================================================

"""
    calc_demand_vector(network::DCNetwork)

Extract demand vector from a DCNetwork.
"""
function calc_demand_vector(network::DCNetwork)
    return copy(network.demand)
end

calc_demand_vector(net::PowerIO.Network) = calc_demand_vector(_network_data(net))
calc_demand_vector(data::NamedTuple) = calc_demand_vector(data, IDMapping(data))

function calc_demand_vector(data::NamedTuple, id_map::IDMapping)
    # to_powerdata already aggregates loads into per-bus demand (per-unit). Index by
    # the sorted IDMapping so demand aligns even when original bus IDs are unsorted.
    d = zeros(length(id_map.bus_ids))
    for bus in data.bus
        d[id_map.bus_to_idx[bus.bus_i]] += bus.pd
    end
    return d
end

"""
    calc_susceptance_matrix(network::DCNetwork)

Compute the susceptance-weighted Laplacian: B = A' * Diagonal(-b .* sw) * A.

Sign convention: `b` stores Im(1/z) which is negative for inductive branches.
The negation `-b` produces positive edge weights, making B positive semidefinite.
This is the negative of PowerModels' `calc_susceptance_matrix` (which uses
the standard bus susceptance matrix convention with negative diagonal).

DC power flow: B * θ = p (net injection).
Branch flows: f = Diag(-b .* sw) * A * θ.
"""
function calc_susceptance_matrix(network::DCNetwork)
    W = Diagonal(-network.b .* network.sw)
    return sparse(network.A' * W * network.A)
end

"""
    _factorize_B_r(net::DCNetwork) → (factor, non_ref)

Factorize the reduced susceptance matrix `B[non_ref, non_ref]`.

Uses Cholesky for standard inductive networks (~2x faster), with LU fallback
for edge cases (capacitive branches or disconnected networks) where B_r is not
positive definite. Follows the approach of AcceleratedDCPowerFlows.jl.
"""
function _factorize_B_r(net::DCNetwork)
    B = calc_susceptance_matrix(net)
    non_ref = setdiff(1:net.n, net.ref_bus)
    B_r = B[non_ref, non_ref]
    factor = try
        cholesky(Symmetric(B_r))
    catch e
        e isa PosDefException || rethrow()
        _SILENCE_WARNINGS[] || @warn "Reduced susceptance matrix B_r is not positive definite (e.g., capacitive branches or disconnected subnetwork); falling back to LU factorization. Results remain correct."
        lu(B_r)
    end
    return factor, non_ref
end

"""
Aggregate generation to bus-level vector.
"""
function _calc_generation_vector(data::NamedTuple, id_map::IDMapping)
    n = length(id_map.bus_ids)
    g = zeros(n)
    for gen in data.gen
        g[id_map.bus_to_idx[gen.gen_bus]] += gen.pg
    end
    return g
end

# =============================================================================
# DCPowerFlowState Constructors
# =============================================================================

"""
    DCPowerFlowState(net::DCNetwork, g::AbstractVector, d::AbstractVector)

Solve DC power flow for given generation and demand.

Computes phase angles θ by solving the reduced system:
    B_r * θ_r = p_r
where B_r is the susceptance-weighted Laplacian with the reference bus row and
column deleted (invertible for a connected network), and p_r is the net injection
with the reference entry removed. The reference bus angle is zero by construction.

# Arguments
- `net`: DCNetwork containing topology and parameters
- `g`: Generation vector (length n, aggregated at each bus)
- `d`: Demand vector (length n)

# Returns
DCPowerFlowState containing angles, injections, and flows.

# Example
```julia
net = DCNetwork(pm_data)
d = calc_demand_vector(net)
g = zeros(net.n)  # Or specify generation at each bus
state = DCPowerFlowState(net, g, d)
```
"""
function DCPowerFlowState(net::DCNetwork, g::AbstractVector{<:Real}, d::AbstractVector{<:Real})
    n, m = net.n, net.m
    length(g) == n || throw(DimensionMismatch("Generation vector length $(length(g)) must match number of buses $n"))
    length(d) == n || throw(DimensionMismatch("Demand vector length $(length(d)) must match number of buses $n"))

    # Net injection
    p = Float64.(g .- d)

    # Factorize reduced susceptance matrix (Cholesky with LU fallback)
    F, non_ref = _factorize_B_r(net)

    # Solve reduced system: θ[non_ref] = B_r \ p[non_ref], θ[ref] = 0
    θ = zeros(n)
    θ[non_ref] = F \ p[non_ref]

    if any(!isfinite, θ)
        error("DC power flow produced non-finite angles. " *
              "The network may be disconnected or have isolated buses.")
    end

    # Compute flows: f = W * A * θ where W = Diag(-b ⊙ sw)
    W = Diagonal(-net.b .* net.sw)
    f = W * net.A * θ

    if any(!isfinite, f)
        error("DC power flow produced non-finite branch flows. " *
              "Check branch impedances for extreme values.")
    end

    return DCPowerFlowState(net, θ, p, convert(Vector{Float64}, g), convert(Vector{Float64}, d), f, F, non_ref)
end

"""
    DCPowerFlowState(net::DCNetwork, d::AbstractVector)

Solve DC power flow with zero generation (pure load flow).

# Arguments
- `net`: DCNetwork containing topology and parameters
- `d`: Demand vector (length n)

# Returns
DCPowerFlowState with generation set to zeros.
"""
function DCPowerFlowState(net::DCNetwork, d::AbstractVector{<:Real})
    g = zeros(net.n)
    return DCPowerFlowState(net, g, d)
end

"""
    DCPowerFlowState(net::Dict; kwargs...)

Reject the removed dictionary API with a migration hint.
"""
function DCPowerFlowState(net::Dict{String,<:Any}; kwargs...)
    throw(ArgumentError("dictionary constructors were removed; construct DCPowerFlowState(DCNetwork(data), g, d)"))
end

"""
    DCPowerFlowState(net::PowerIO.Network; g=nothing, d=nothing)

Construct DCPowerFlowState from a parsed PowerIO network.
If `d` is not provided, extracts demand from the network.
If `g` is not provided, aggregates generation from gen data to buses.
"""
function DCPowerFlowState(net::PowerIO.Network; g::Union{Nothing,AbstractVector}=nothing, d::Union{Nothing,AbstractVector}=nothing)
    net = DCNetwork(net)

    if isnothing(d)
        d = net.demand
    end

    # Aggregate generation to buses if not provided
    if isnothing(g)
        g = net.pg_init
    end

    return DCPowerFlowState(net, g, d)
end
