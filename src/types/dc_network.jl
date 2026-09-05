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
Internal cache for the energized DC topology. The incidence matrix structure is
fixed after construction, while `b` and `sw` may change in place.

The cache is prewarmed by constructors and refreshed by topology readers when
`b` or `sw` changes. It is not a synchronization primitive: callers sharing a
`DCNetwork` across threads must treat topology fields as read only, or serialize
mutations and the first topology read after each mutation.
"""
mutable struct _DCTopologyCache
    # Branch endpoints as sequential bus indices, cached from the incidence matrix:
    # from_bus[e]/to_bus[e] are the columns of the +1/-1 entries in row `e` of `A`.
    # This is the same information `A` already encodes; it is materialized as dense
    # Int vectors (once, since `A`'s structure is fixed) so the connectivity refresh
    # can read each branch's endpoints in O(1) instead of rescanning sparse rows.
    from_bus::Vector{Int}
    to_bus::Vector{Int}
    energized::BitVector    # energized[e] == (b[e] * sw[e] != 0)
    refs::Vector{Int}       # one reference bus per energized island (sorted)
    non_ref::Vector{Int}    # all buses except `refs`
    initialized::Bool
end

_DCTopologyCache() = _DCTopologyCache(Int[], Int[], BitVector(), Int[], Int[], false)

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
- `c_shed`: Load shedding cost per bus (penalty for involuntary load curtailment)
- `ref_bus`: Preferred reference bus index (phase angle = 0)
- `tau`: Regularization parameter for strong convexity
- `id_map`: Bidirectional mapping between original and sequential element IDs
- `demand`: Real power demand aggregated per bus
- `pg_init`: Initial real generation aggregated per bus
- `topology_cache`: Internal energized island cache. This cache is mutable even
  though `DCNetwork` is an immutable struct; concurrent direct mutations of
  `b`/`sw` are unsupported.
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
    topology_cache::_DCTopologyCache
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
- `eta_ref`: Reference bus constraint duals (`va[reference_buses(net)] == 0`)
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
    eta_ref::Vector{Float64}
    objective::Float64
    B_r_factor::F
end

"""
    DCPowerFlowState{F} <: AbstractPowerFlowState

DC power flow solution (phase angles from reduced-Laplacian solve, no optimization).
Supports both generation and demand for flexible sensitivity analysis.

Unlike DCOPFSolution, this represents a simple power flow solution
`θ_r = B_r \\ p_r` where `B_r` is the susceptance matrix with one reference row and
column deleted per energized island, without optimal dispatch or
constraint handling.

# Fields
- `net`: DCNetwork data
- `va`: Phase angles (rad), with `va[reference_buses(net)] = 0`
- `p`: Net injection vector (p = pg - d)
- `pg`: Generation vector
- `d`: Demand vector
- `f`: Branch flows (computed from va)
- `B_r_factor`: Factorization of `B[non_ref, non_ref]` (Cholesky for inductive networks, LU fallback)
- `non_ref`: Indices excluding one reference bus per energized island
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
# PowerIO is the parser and data layer, and the only one: PowerDiff has no parser
# backend switch and no second representation of a case. `PowerIO.parse` reads every
# format the linked library ships and returns a `PowerIO.PioModule{T}` carrying the
# parsed value, the reader's diagnostics, and the source record.
#
# `PowerIO.to_powerdata` states a balanced network in per unit, with radian branch
# angles, per-bus load and shunt aggregation, rescaled polynomial costs, a `status`
# flag and a source row number `i` on every row, and the four terminal admittance
# coefficients `c1..c8` per branch. It is unfiltered: out-of-service rows and
# isolated buses reach the caller.
#
# `_network_data` selects what PowerDiff models -- in-service elements on buses that
# are not isolated -- and adds the OPF modeling PowerIO leaves to its consumers:
# polynomial cost interpretation, a finite thermal limit when the source states
# none, default angle-difference bounds, and refusal of records PowerDiff has no
# model for. It derives no electrical quantity: the series admittance and the
# terminal charging terms are read off PowerIO's coefficients.

"""
    parse_file(path::String; library=nothing, from=nothing, filetype=nothing) -> PowerIO.PioModule
    parse_file(io::IO; from="matpower", filetype=nothing) -> PowerIO.PioModule

Parse a network into a `PowerIO.PioModule{PowerIO.BalancedNetwork}`, in any
transmission format the linked PowerIO library reads.

For paths, PowerIO infers the format from the extension unless `from` is given. For
streams, pass `from` (or `filetype`), because a stream has no extension; MATPOWER is
assumed when neither is given. A bare `json` names a container rather than a reader,
so name the reader (`from=:powermodels`, `:egret`, `:pandapower`, `:goc3`, `:surge`,
`:opfdata`).

`from` takes PowerIO's own format tokens, and the vocabulary is PowerIO's rather
than a copy of it: an unrecognized token goes to PowerIO, which answers with what
the linked library actually reads, so a reader PowerIO gains is usable here at once.
PowerDiff's historical short spellings (`:m`, `:raw`, `:aux`, `:pm`, `:powermodels`,
`:egret`) resolve to the tokens they always meant.

A source that parses to anything other than a balanced transmission network -- a
distribution case, a time series, a scenario set, a calculation instance -- is
refused by naming what it holds.

The returned module carries more than the network: `m.value` is the case,
`m.diagnostics` is what the reader retained or had to assume, `m.sources[1].format`
is the reader that ran, and `PowerIO.emit(m, format, path)` writes the case out
again. Pass the module to [`DCNetwork`](@ref) or [`ACNetwork`](@ref).
"""
function parse_file(source::Union{IO,AbstractString}; library=nothing, filetype=nothing,
                    from=nothing, kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "unsupported parse_file keyword(s): $(join(string.(keys(kwargs)), ", "))"))
    fmt = _powerio_format_hint(from, filetype)
    m = if source isa AbstractString
        PowerIO.parse(_resolve_case_path(source, library); format=fmt)
    else
        PowerIO.parse(source; format=something(fmt, "matpower"))
    end
    return _require_balanced(m)
end

"""
    parse_matpower(io::IO) -> PowerIO.PioModule
    parse_matpower(file::String; library=nothing) -> PowerIO.PioModule

Parse MATPOWER v2 data into a `PowerIO.PioModule{PowerIO.BalancedNetwork}`.
"""
parse_matpower(io::IO) = _require_balanced(PowerIO.parse(io; format="matpower"))

parse_matpower(file::String; library=nothing) =
    _require_balanced(PowerIO.parse(_resolve_case_path(file, library); format="matpower"))

# PowerIO returns one of twenty value kinds. Only a balanced transmission network
# has the buses, branches and generators PowerDiff differentiates, so the type check
# is the refusal: it covers distribution cases, series carriers and calculation
# instances alike, without a list here that trails what PowerIO parses.
_require_balanced(m::PowerIO.PioModule{PowerIO.BalancedNetwork}) = m
_require_balanced(m::PowerIO.PioModule) = throw(ArgumentError(
    "PowerDiff models balanced transmission networks; this source parsed to a " *
    "$(nameof(typeof(m.value)))"))

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

# The short spellings PowerDiff has always accepted, mapped to PowerIO's own token.
# This is a courtesy layer, not a gate: a token absent from it passes through
# untouched, and PowerIO answers for it.
const _FORMAT_ALIASES = Dict(
    "m" => "matpower",
    "raw" => "psse",
    "aux" => "powerworld",
    "pm" => "powermodels-json",
    "powermodels" => "powermodels-json",
    "egret" => "egret-json",
)

function _format_token(x)
    s = lowercase(String(x))
    startswith(s, ".") && (s = s[2:end])
    isempty(s) && throw(ArgumentError("network format hint is empty"))
    # A bare `json` names a container, not a reader, and PowerIO has several. Say so
    # here: routing it would pick one by guess.
    s == "json" && throw(ArgumentError(
        "JSON input is ambiguous; name the reader (from=:powermodels, from=:egret, " *
        "from=:pandapower, from=:goc3, from=:surge, from=:opfdata)"))
    return get(_FORMAT_ALIASES, s, s)
end

"""
    _network_data(m::PowerIO.PioModule) -> NamedTuple
    _network_data(net::PowerIO.BalancedNetwork) -> NamedTuple

Build PowerDiff network tables from a parsed PowerIO network.

`PowerIO.to_powerdata` supplies every value: per-unit powers and ratings, radian
branch angles and angle bounds, per-bus aggregated load and shunt, right-aligned
per-unit polynomial cost coefficients, and the four terminal admittance
coefficients from which the series conductance and susceptance are read. It is
unfiltered, so this selects what PowerDiff models: rows whose `status` is set, on
buses that are not isolated (`type == 4`). Its row field `i` is the source row
number, which becomes the [`IDMapping`](@ref) index, so out-of-service rows leave
gaps rather than renumbering the rows that remain.

On top of that selection sit the four modeling decisions PowerIO leaves to its
consumer: polynomial cost interpretation (rejecting piecewise linear), a
synthesized thermal limit when the source states none, default angle-difference
bounds, and refusal of storage and HVDC records.

Bus rows carry the source bus id on `bus_i`, so [`IDMapping`](@ref)`.bus_ids` and
any bus-indexed sensitivity `row_to_id` map back to the input network.
"""
# A parsed case reaches PowerDiff either as the module `parse_file` returns or as the
# network inside it. Both name the same case, so both construct the same networks.
const PowerIOSource = Union{PowerIO.PioModule{PowerIO.BalancedNetwork},PowerIO.BalancedNetwork}

_network_data(m::PowerIO.PioModule{PowerIO.BalancedNetwork}) = _network_data(m.value)

function _network_data(net::PowerIO.BalancedNetwork)
    isempty(net.hvdc) || throw(ArgumentError(
        "PowerDiff does not support HVDC/dcline records; remove or convert dcline before parsing"))
    isempty(net.storage) || throw(ArgumentError(
        "PowerDiff does not support storage records; remove or convert storage before parsing"))

    pd = PowerIO.to_powerdata(net)
    isempty(pd.bus) && throw(ArgumentError("network has no buses"))

    # `to_powerdata` states the cost coefficients but not which cost model produced
    # them, so the model comes off the element table. One pass, read once.
    gens = collect(net.generators)

    # An isolated bus is out of service, and so is every element on it.
    live = [Int(b.type) != 4 for b in pd.bus]
    keep_bus = findall(live)
    isempty(keep_bus) && throw(ArgumentError("network has no in-service buses"))

    orig = [Int(b.bus_i) for b in pd.bus]   # table position -> source bus id
    buses = [_bus_row(orig[i], pd.bus[i]) for i in keep_bus]
    vmax_by_id = Dict(b.bus_i => b.vmax for b in buses)

    kept_gen = [Int(g.i) for g in pd.gen if Int(g.status) != 0 && live[g.bus]]
    isempty(kept_gen) && throw(ArgumentError("network has no in-service generators"))
    gen_rows = [_gen_row(pd.gen[j], orig[pd.gen[j].bus], gens[j]) for j in kept_gen]

    kept_branch = [Int(br.i) for br in pd.branch
                   if Int(br.status) != 0 && live[br.f_bus] && live[br.t_bus]]
    isempty(kept_branch) && throw(ArgumentError("network has no in-service branches"))
    branches = [_branch_row(pd.branch[l], orig, vmax_by_id) for l in kept_branch]

    return (; name = net.name,
            baseMVA = Float64(pd.baseMVA),
            bus = buses, gen = gen_rows, branch = branches)
end

# =============================================================================
# Absent numeric bounds
# =============================================================================
#
# PowerIO passes an absent bound through as `±Inf` instead of refusing the case:
# `Inf` is how MATPOWER, PowerModels, pandapower and PyPSA all spell "no limit", and
# stock pglib cases carry it (case9241pegase leaves the reactive limits off seven
# generators). PowerDiff's KKT layout is fixed, with one complementarity row per
# bound, so `Inf` cannot simply flow in: `ρ * (qg - qmin)` with `qmin == -Inf` is
# `0 * Inf`, a `NaN` in the residual and an `Inf` in the Jacobian.
#
# Reactive generator limits model absence properly -- the bound is left off the
# solver model and its complementarity row reads `ρ = 0`, the multiplier of a
# constraint that is not there (see `_lb_complementarity` and its derivatives in
# `prob/kkt_ac_opf.jl`). A branch that states no rating takes a synthesized one.
# Everywhere else PowerIO rejects a non-finite value itself, naming the element and
# the field, so nothing here repeats that check; `_require_finite_bounds` covers the
# caller-built tables PowerIO never sees.

"""
    _absent_bound(v) -> Bool

Whether a variable bound is absent. PowerIO spells an absent bound `±Inf`.
"""
_absent_bound(v::Real) = !isfinite(v)

_bus_row(bus_id, b) = (;
    bus_i = bus_id, bus_type = Int(b.type),
    pd = Float64(b.pd), qd = Float64(b.qd),
    gs = Float64(b.gs), bs = Float64(b.bs),
    vm = Float64(b.vm), va = Float64(b.va),
    vmin = Float64(b.vmin), vmax = Float64(b.vmax),
)

_gen_row(g, bus_id, element) = (;
    index = Int(g.i), gen_bus = bus_id,
    pg = Float64(g.pg), qg = Float64(g.qg),
    qmin = Float64(g.qmin), qmax = Float64(g.qmax),
    vg = Float64(g.vg),
    pmin = Float64(g.pmin), pmax = Float64(g.pmax),
    cost = _poly_cost(g, element),
)

# One PowerDiff branch row from a `to_powerdata` branch row: default the angle
# window, read the series admittance off PowerIO's terminal coefficients, and
# synthesize a finite rate_a when the source states no thermal limit.
#
# "States no limit" is `rate_a == 0`, MATPOWER's spelling, or a non-finite `rate_a`,
# which is how PowerIO carries an unbounded rating out of the formats that write
# one. Both mean the same thing and both take the same synthesized limit, the
# largest flow the endpoint voltage limits and the angle window physically admit --
# a bound by construction, not an invented rating that could bind.
function _branch_row(br, orig, vmax_by_id)
    angmin, angmax = _normalize_angle_bounds(Float64(br.angmin), Float64(br.angmax))
    f_bus = orig[br.f_bus]
    t_bus = orig[br.t_bus]
    g_to = Float64(br.g_to)
    b_to = Float64(br.b_to)
    # `to_powerdata`'s eighth and seventh coefficients are the to-side terminal
    # admittance added to the series admittance, so the series part reads back
    # exactly. Nothing here inverts an impedance.
    g = Float64(br.c7) - g_to
    b = Float64(br.c8) - b_to
    raw_rate_a = Float64(br.rate_a)
    rate_a = (isfinite(raw_rate_a) && raw_rate_a > 0) ? raw_rate_a :
             _fallback_rate_a(Float64(br.br_r), Float64(br.br_x), angmin, angmax,
                              vmax_by_id[f_bus], vmax_by_id[t_bus])
    return (; index = Int(br.i), f_bus = f_bus, t_bus = t_bus,
            br_r = Float64(br.br_r), br_x = Float64(br.br_x),
            g = g, b = b,
            g_fr = Float64(br.g_fr), b_fr = Float64(br.b_fr),
            g_to = g_to, b_to = b_to,
            rate_a = rate_a,
            rate_b = _unlimited_as_zero(br.rate_b), rate_c = _unlimited_as_zero(br.rate_c),
            tap = Float64(br.tap), shift = Float64(br.shift),
            angmin = angmin, angmax = angmax)
end

# `rate_b` / `rate_c` are carried through untouched and unused by either formulation.
# Keep them numeric so a caller reading the tables never meets an `Inf`: an unbounded
# rating reads as `0`, which is the same "no limit" spelling `rate_a` arrives in.
_unlimited_as_zero(x) = (v = Float64(x); isfinite(v) ? v : 0.0)

# Interpret a generator's polynomial cost as PowerDiff's (quadratic, linear,
# constant) tuple. `to_powerdata` returns a model 2 cost as a right-aligned per-unit
# triple and rejects higher-than-quadratic itself; the cost model comes off the
# element, which states it verbatim. A generator with no cost record is cost-free.
function _poly_cost(g, element)
    cost = element.cost
    cost === nothing && return (0.0, 0.0, 0.0)
    Int(cost.model) == 2 || throw(ArgumentError(
        "generator $(Int(g.i)) states cost model $(Int(cost.model)); PowerDiff models " *
        "polynomial (model 2) costs. Convert piecewise linear costs before parsing"))
    c = g.c
    return (Float64(c[1]), Float64(c[2]), Float64(c[3]))
end

# PowerDiff's OPF needs a finite thermal limit on every branch. When the source
# states none, synthesize one from the bus voltage limits and the branch impedance
# and angle window.
function _fallback_rate_a(r::Float64, x::Float64, angmin::Float64, angmax::Float64,
                          fr_vmax::Float64, to_vmax::Float64)
    theta_max = max(abs(angmin), abs(angmax))
    zmag = hypot(r, x)
    ymag = iszero(zmag) ? 0.0 : inv(zmag)
    cmax = sqrt(fr_vmax^2 + to_vmax^2 - 2fr_vmax * to_vmax * cos(theta_max))
    return ymag * max(fr_vmax, to_vmax) * cmax
end

# Default angle difference bounds (radians in, radians out). MATPOWER angmin == angmax
# == 0 means unbounded; treat ±90° or wider and the zero case as a ±60° window, the
# MATPOWER/PowerModels convention. PowerIO's `to_powerdata` already converts to radians.
function _normalize_angle_bounds(angmin::Float64, angmax::Float64)
    pad = deg2rad(60.0)
    angmin <= -pi / 2 && (angmin = -pad)
    angmax >= pi / 2 && (angmax = pad)
    iszero(angmin) && iszero(angmax) && return (-pad, pad)
    return angmin, angmax
end

# The branch-by-bus incidence matrix over PowerDiff's own index space: one row per
# branch in `id_map.branch_ids` order, one column per bus in sorted source-id order,
# `+1` at the from bus and `-1` at the to bus. Both network types read it from here,
# so the two describe the same graph by construction.
#
# PowerIO's `calc_incidence_matrix` covers in-service branches in table order against
# all buses and returns a bare sparse matrix with no branch or bus index map
# (eigenergy/PowerIO.jl#2 in the issues this port filed), so relabeling it into this
# space costs more than stating it.
function _incidence_matrix(branch_tbl, id_map::IDMapping)
    m = length(id_map.branch_ids)
    A = spzeros(Float64, m, length(id_map.bus_ids))
    for orig_id in id_map.branch_ids
        br = branch_tbl[orig_id]
        row = id_map.branch_to_idx[orig_id]
        A[row, id_map.bus_to_idx[br.f_bus]] = 1.0
        A[row, id_map.bus_to_idx[br.t_bus]] = -1.0
    end
    return A
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
    DCNetwork(net::PowerIO.BalancedNetwork; tau=DEFAULT_TAU, ref_bus=nothing)

Construct a DCNetwork from a parsed PowerIO network.

# Example
```julia
net = parse_file("case14.m")
dc_net = DCNetwork(net)
```
"""
DCNetwork(net::PowerIOSource; tau::Float64=DEFAULT_TAU, ref_bus::Union{Nothing,Int}=nothing) =
    DCNetwork(_network_data(net); tau=tau, ref_bus=ref_bus)

# Build from PowerDiff network tables (see `_network_data`). The `PowerIO.BalancedNetwork`
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

    A = _incidence_matrix(branch_tbl, id_map)

    # Generator-bus incidence matrix G_inc (n × k)
    G_inc = spzeros(n, k)
    for orig_id in id_map.gen_ids
        gen = gen_tbl[orig_id]
        col = id_map.gen_to_idx[orig_id]
        row = id_map.bus_to_idx[gen.gen_bus]
        G_inc[row, col] = 1.0
    end

    # Series susceptance per branch, as PowerIO states it: `imag(1/z)`, negative for
    # an inductive branch, so `W = -b .* sw` weights the Laplacian positively.
    b = [Float64(branch_tbl[id_map.branch_ids[i]].b) for i in 1:m]

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

    # Load shedding cost: high penalty to discourage shedding when feasible.
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

    net = DCNetwork(n, m, k, A, G_inc, b, sw, fmax, gmax, gmin, angmax, angmin,
                    cq, cl, c_shed, demand, pg_init, ref_bus, tau, id_map,
                    _DCTopologyCache())
    _refresh_topology_cache!(net)
    return net
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
    net = DCNetwork(
        n, m, k,
        sparse(Float64.(A)), sparse(Float64.(G_inc)),
        Float64.(b), Float64.(sw),
        Float64.(fmax), Float64.(gmax), Float64.(gmin),
        Float64.(angmax), Float64.(angmin),
        Float64.(cq), Float64.(cl),
        Float64.(c_shed),
        Float64.(demand), Float64.(pg_init),
        ref_bus, tau,
        IDMapping(n, m, k),
        _DCTopologyCache()
    )
    _refresh_topology_cache!(net)
    return net
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

calc_demand_vector(net::PowerIOSource) = calc_demand_vector(_network_data(net))
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

@inline _is_energized(net::DCNetwork, e::Int) =
    !iszero(getfield(net, :b)[e] * getfield(net, :sw)[e])

# Phase-angle bounds use the switching value on energized branches and vanish
# everywhere else. This preserves the existing continuous switching scaling while
# making the bound topology agree with the energized-island predicate above.
@inline _angle_difference_gate(net::DCNetwork, e::Int) =
    _is_energized(net, e) ? getfield(net, :sw)[e] : zero(getfield(net, :sw)[e])

function _angle_difference_gates(net::DCNetwork)
    gates = similar(getfield(net, :sw))
    @inbounds for e in eachindex(gates)
        gates[e] = _angle_difference_gate(net, e)
    end
    return gates
end

# Within a fixed energized regime, the gate is `sw` when b != 0 and zero when
# b == 0. Crossing b == 0 is a nonsmooth topology boundary, so its local
# susceptance derivative follows the same zero convention used at other topology
# boundaries.
@inline _angle_difference_gate_dsw(net::DCNetwork, e::Int) =
    iszero(getfield(net, :b)[e]) ? zero(getfield(net, :sw)[e]) : one(getfield(net, :sw)[e])

function _topology_cache_valid(net::DCNetwork)
    cache = getfield(net, :topology_cache)
    cache.initialized || return false
    energized = cache.energized
    m = getfield(net, :m)
    length(energized) == m || return false
    @inbounds for e in 1:m
        energized[e] == _is_energized(net, e) || return false
    end
    return true
end

# Recompute the energized-island partition and its reference buses.
#
# Energized branches (b[e]*sw[e] != 0) define the connectivity graph; buses joined
# by them form one island. We run union-find over those branches, then pick one
# reference bus per island: the lowest bus index in each, except the configured
# `ref_bus`, which is forced to be the reference for its own island. Buses with no
# energized branch are singleton islands that reference themselves.
function _refresh_topology_cache!(net::DCNetwork)
    cache = net.topology_cache

    # Union-find over buses. `parent[i]` points toward i's component root; a bus is
    # its own parent until merged. find_root uses path halving, union_roots! keeps
    # the lower index as root so the partition (and thus refs) is deterministic.
    parent = collect(1:net.n)

    function find_root(i::Int)
        while parent[i] != i
            parent[i] = parent[parent[i]]
            i = parent[i]
        end
        return i
    end

    function union_roots!(i::Int, j::Int)
        root_i = find_root(i)
        root_j = find_root(j)
        root_i == root_j && return nothing
        if root_i < root_j
            parent[root_j] = root_i
        else
            parent[root_i] = root_j
        end
        return nothing
    end

    # Cache the branch endpoints from `A` once. `A`'s structure is fixed after
    # construction, so this only runs the first time (or if `m` somehow changed).
    if length(cache.from_bus) != net.m
        cache.from_bus = zeros(Int, net.m)
        cache.to_bus = zeros(Int, net.m)
        rows, cols, nz_values = findnz(net.A)
        @inbounds for p in eachindex(rows)
            if nz_values[p] > 0
                cache.from_bus[rows[p]] = cols[p]   # +1 entry -> from bus
            else
                cache.to_bus[rows[p]] = cols[p]     # -1 entry -> to bus
            end
        end
    end

    # Recompute the energized flags (these track `b`/`sw`) and union the endpoints
    # of every energized branch so each island collapses to a single root.
    resize!(cache.energized, net.m)
    @inbounds for e in 1:net.m
        cache.energized[e] = _is_energized(net, e)
        cache.energized[e] || continue
        cache.from_bus[e] > 0 && cache.to_bus[e] > 0 ||
            error("Incidence matrix row $e must have exactly two nonzero entries")
        union_roots!(cache.from_bus[e], cache.to_bus[e])
    end

    # One reference per island: start with the lowest bus index in each component...
    refs_by_root = Dict{Int,Int}()
    @inbounds for bus in 1:net.n
        root = find_root(bus)
        refs_by_root[root] = min(get(refs_by_root, root, bus), bus)
    end
    # ...then force the configured ref_bus to be the reference for its island.
    refs_by_root[find_root(net.ref_bus)] = net.ref_bus
    cache.refs = sort!(collect(values(refs_by_root)))
    cache.non_ref = setdiff(1:net.n, cache.refs)
    cache.initialized = true
    return cache
end

function _topology_cache(net::DCNetwork)
    # Refresh mutates `net.topology_cache`. Constructors prewarm this cache, so
    # normal read only sharing across threads does not first touch it. If callers
    # mutate `b` or `sw` directly, they must serialize that mutation and the next
    # topology read; the exposed vectors themselves are not thread safe.
    _topology_cache_valid(net) || _refresh_topology_cache!(net)
    return getfield(net, :topology_cache)
end

_reference_buses(net::DCNetwork) = _topology_cache(net).refs

"""
    reference_buses(net::DCNetwork) → Vector{Int}

Return one deterministic reference bus for each energized island.

The configured `net.ref_bus` is preserved for its island. Every other island,
including an isolated bus, uses its lowest sequential bus index. A branch is
energized when `b[e] * sw[e] != 0`.
"""
reference_buses(net::DCNetwork) = copy(_reference_buses(net))

"""Return bus indices after removing one reference bus per energized island."""
_non_reference_buses(net::DCNetwork) = copy(_topology_cache(net).non_ref)

"""
    _factorize_B_r(net::DCNetwork) → (factor, non_ref)

Factorize the reduced susceptance matrix `B[non_ref, non_ref]`.

Uses Cholesky for standard inductive networks (~2x faster), with LU fallback
for edge cases such as capacitive branches where B_r is not positive definite.
One reference row and column is removed per energized island, so disconnected
networks and isolated buses remain well-defined.
"""
function _factorize_B_r(net::DCNetwork)
    B = calc_susceptance_matrix(net)
    non_ref = _non_reference_buses(net)
    B_r = B[non_ref, non_ref]
    factor = try
        cholesky(Symmetric(B_r))
    catch e
        e isa PosDefException || rethrow()
        _SILENCE_WARNINGS[] || @warn "Reduced susceptance matrix B_r is not positive definite (e.g., capacitive branches); falling back to LU factorization. Results remain correct."
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
where B_r is the susceptance-weighted Laplacian with one reference bus row and
column deleted per energized island, and p_r is the net injection with those
reference entries removed. Reference bus angles are zero by construction.

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

    # Solve reduced system: θ[non_ref] = B_r \ p[non_ref], θ[refs] = 0
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
    DCPowerFlowState(net::PowerIO.BalancedNetwork; g=nothing, d=nothing)

Construct DCPowerFlowState from a parsed PowerIO network.
If `d` is not provided, extracts demand from the network.
If `g` is not provided, aggregates generation from gen data to buses.
"""
function DCPowerFlowState(net::PowerIOSource; g::Union{Nothing,AbstractVector}=nothing, d::Union{Nothing,AbstractVector}=nothing)
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
