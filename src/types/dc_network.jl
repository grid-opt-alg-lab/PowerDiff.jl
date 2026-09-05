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
# backend switch and no second representation of a case. `PowerIO.parse_*` reads every
# transmission format the linked library ships; `PowerIO.to_normalized` puts a case in
# per unit with out-of-service and isolated elements dropped, the reference bus
# inferred, source bus ids preserved, loads and shunts aggregated per bus, and
# polynomial costs collapsed and rescaled.
#
# These thin wrappers return a `PowerIO.BalancedNetwork`, and `_network_data` turns one
# into the network tables the DCNetwork and ACNetwork constructors consume — once per
# network, memoized, so the two constructors cannot describe different cases. The only
# logic beyond re-keying to source bus ids is the OPF solver modeling PowerIO leaves to
# the consumer: polynomial cost interpretation, finite flow limits, default angle
# difference bounds, absent reactive limits, and rejection of records PowerDiff does
# not model. Everything PowerIO reports about the case travels with it, as data, under
# `network_findings`.

"""
    parse_file(path::String; library=nothing, from=nothing, filetype=nothing) -> PowerIO.BalancedNetwork
    parse_file(io::IO; from="matpower", filetype=nothing) -> PowerIO.BalancedNetwork

Parse a network file into a `PowerIO.BalancedNetwork`, in any transmission format
the linked PowerIO library reads.

For paths, PowerIO infers the format from the extension unless `from` is given. For
streams, pass `from` (or `filetype`), because a stream has no extension; MATPOWER is
assumed when neither is given. A bare `json` names a container rather than a reader,
so name the reader (`from=:powermodels`, `:egret`, `:pandapower`, `:goc3`, `:surge`,
`:opfdata`).

`from` takes PowerIO's own format tokens — `:matpower`/`:m`, `:psse`/`:raw`,
`:psse34`, `:psse35`, `:powerworld`/`:aux`, `:powermodels`, `:egret`, `:pandapower`,
`:pypsa`, `:pslf`/`:epc`, `:pwb`, `:gridfm`, `:goc3`, `:surge`, `:opfdata` — and the
list is PowerIO's, not a copy of it, so a reader PowerIO gains is usable here at
once. An unrecognized token is refused by PowerIO with the set its build actually
reads. Distribution formats (`:dss`, `:pmd`, `:bmopf`) are refused here: they parse
to a `PowerIO.MulticonductorNetwork`, which PowerDiff does not model.

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
    parse_matpower(io::IO) -> PowerIO.BalancedNetwork
    parse_matpower(file::String; library=nothing) -> PowerIO.BalancedNetwork

Parse MATPOWER v2 data into a `PowerIO.BalancedNetwork`.
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

# Distribution tokens parse to a `PowerIO.MulticonductorNetwork`, which is not a
# balanced transmission network and carries nothing PowerDiff differentiates. They
# are named here so the refusal says what to do rather than surfacing a type error
# from three frames down.
const _DISTRIBUTION_FORMAT_TOKENS = (
    "dss", "opendss",
    "pmd", "pmd-json", "pmdjson", "engineering",
    "bmopf", "bmopf-json", "bmopfjson",
)

# The short spellings PowerDiff has always accepted, mapped to PowerIO's own token.
# This is a courtesy layer, not a gate: a token absent from it passes through
# untouched. PowerIO owns the format vocabulary, so a reader it gains is usable
# from `parse_file` the day it ships, and an unknown token is answered by PowerIO's
# `REQUEST.FORMAT.UNKNOWN` (which lists what the linked library actually reads)
# rather than by a list here that silently trails it.
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
    s in _DISTRIBUTION_FORMAT_TOKENS && throw(ArgumentError(
        "$x is a distribution format; PowerDiff models balanced transmission networks. " *
        "Parse it with PowerIO, then lower it: PowerIO.to_package, " *
        "PowerIO.lower_multiconductor_to_balanced, PowerIO.from_package"))
    return get(_FORMAT_ALIASES, s, s)
end

"""
    _network_data(net::PowerIO.BalancedNetwork) -> NamedTuple

Build PowerDiff network tables from a parsed PowerIO network.

PowerDiff runs PowerIO's normalize pass itself (`PowerIO.to_normalized`) and reads
`PowerIO.to_powerdata` off the normalized network. That is one pass, not two:
`to_powerdata` recognizes an already-normalized input and skips its own. It also
puts the pass's fidelity findings in PowerDiff's hands as data — see
[`network_findings`](@ref) — instead of a `@warn` burst PowerDiff could neither
label nor deduplicate.

Normalization does per-unit scaling, status/isolated filtering, per-bus load/shunt
aggregation, reference-bus inference (`type == 3`), source bus ids on `bus_i`, and
polynomial cost collapse/rescaling, returning dense file-order rows. This adapter
keys bus references back to source bus ids (so [`IDMapping`](@ref)'s sorted ordering
is preserved) and applies the OPF modeling PowerIO leaves to the consumer:
polynomial cost interpretation (rejecting PWL and higher-than-quadratic), a finite
flow limit fallback when the source states no thermal limit, default angle
difference bounds, and rejection of storage / HVDC records that PowerDiff does not
model.

The returned `bus`/`gen`/`branch` rows mirror the field names the network
constructors expect, with loads/shunts already folded into per-bus `pd/qd/gs/bs`.
`shunt` re-exposes those bus shunts as a table (one `(; index, shunt_bus, gs, bs)`
record per bus with a nonzero shunt admittance) for callers that want shunt records.

Bus rows carry the source bus id on `bus_i`, so [`IDMapping`](@ref)`.bus_ids`
(and any bus-indexed sensitivity `row_to_id`) map back to the input network.
Generator and branch `index` values are source row numbers among the unfiltered
PowerIO rows, so out-of-service rows leave gaps instead of renumbering active rows.

The tables are memoized per parsed network, so building a [`DCNetwork`](@ref) and an
[`ACNetwork`](@ref) from one `net` normalizes once, materializes the JSON payload
once, and reports each finding once. The two therefore cannot disagree about the
case they describe.
"""
_network_data(net::PowerIO.BalancedNetwork) = _powerio_ingest(net).tables

"""
    network_findings(net::PowerIO.BalancedNetwork) -> NamedTuple

What PowerIO reported about `net`, as data rather than log output.

- `reader` — the fidelity findings the parser retained: what the source format could
  not represent, or what the reader had to assume. Also reachable as
  `PowerIO.warnings(net)`.
- `normalize` — the findings of the normalize pass PowerDiff builds its tables from,
  such as `CANONICALIZE.NORMALIZE.GEN_COST_ABSENT` (the case states no generator cost
  data, so any cost objective built from it is identically zero) or
  `CANONICALIZE.NORMALIZE.REFERENCE_DESIGNATED` (the case named no reference bus, so
  one was chosen).

Every line reads `CODE: message`. Split at the first `": "` and branch on the code;
the prose carries no stability promise. PowerDiff reports the `normalize` findings
once per network through its own warning channel (silenced by [`silence`](@ref));
the `reader` findings are returned here and never logged, matching PowerIO, which
leaves them on the parsed network for the consumer to read.

```julia
net = parse_file("case14.m")
findings = network_findings(net)
any(startswith("CANONICALIZE.NORMALIZE.GEN_COST_ABSENT"), findings.normalize)
```
"""
function network_findings(net::PowerIO.BalancedNetwork)
    ingest = _powerio_ingest(net)
    return (; reader = PowerIO.warnings(net), normalize = ingest.findings)
end

# Ingesting a network materializes its JSON payload, runs the normalize pass and
# walks every row, and PowerDiff's own constructors call it once per network type.
# Memoizing on the network object keeps `DCNetwork(net)` and `ACNetwork(net)` to one
# pass between them, which is also what keeps their tables identical by construction.
# Weak keys so a parsed network that goes out of scope is still collectable.
const _INGEST_CACHE = WeakKeyDict{PowerIO.BalancedNetwork,Any}()

# An ingest is only valid for the live Rust handle it was read through, and a
# precompiled image must not carry one across a process boundary. Cleared on load.
_reset_ingest_cache!() = (empty!(_INGEST_CACHE); nothing)

function _powerio_ingest(net::PowerIO.BalancedNetwork)
    cached = get(_INGEST_CACHE, net, nothing)
    cached === nothing || return cached

    # Reject records PowerDiff does not model. Both guards read the raw network so
    # they stay consistent: normalized output drops out-of-service records, which
    # would silently accept a file that declares them.
    isempty(PowerIO.hvdc(net)) || throw(ArgumentError(
        "PowerDiff does not support HVDC/dcline records; remove or convert dcline before parsing"))
    isempty(PowerIO.storage(net)) || throw(ArgumentError(
        "PowerDiff does not support storage records; remove or convert storage before parsing"))

    # One normalize pass, owned here. `to_powerdata` on a network already flagged
    # `"normalized"` reads it straight through, so this is not an extra pass; it is
    # the same pass with its findings returned instead of logged.
    normalized = PowerIO.source_format(net) == "normalized" ? net : PowerIO.to_normalized(net)
    findings = PowerIO.warnings(normalized)
    _report_findings(findings)

    ingest = (; tables = _build_network_tables(net, PowerIO.to_powerdata(normalized)),
              findings = findings)
    _INGEST_CACHE[net] = ingest
    return ingest
end

# PowerIO 0.9 raises the normalize findings as one `@warn` per distinct code from
# inside `to_powerdata`. PowerDiff takes that pass over, so it owes the user the same
# information: same one-per-code rule, said once for the network rather than once per
# constructor, and under PowerDiff's own silence switch.
function _report_findings(findings)
    (_SILENCE_WARNINGS[] || isempty(findings)) && return nothing
    seen = Set{SubString{String}}()
    for line in findings
        code = first(split(line, ": "; limit=2))
        code in seen && continue
        push!(seen, code)
        @warn "PowerIO normalize: $line"
    end
    return nothing
end

function _build_network_tables(net, pd)
    isempty(pd.bus) && throw(ArgumentError("network has no active buses"))
    isempty(pd.gen) && throw(ArgumentError("network has no active generators"))
    isempty(pd.branch) && throw(ArgumentError("network has no active branches"))

    orig = [Int(b.bus_i) for b in pd.bus]   # dense file-order index -> source bus id
    gen_source_rows, branch_source_rows = _active_source_rows(net, pd)

    buses = [_bus_row(orig[i], b) for (i, b) in enumerate(pd.bus)]

    # Costs come straight from the normalized gen rows (already per-unit and
    # right-aligned). Map dense `gen.bus` to the source bus id via `orig`.
    gens = [_gen_row(gen_source_rows[j], orig[g.bus], g) for (j, g) in enumerate(pd.gen)]

    branches = [_branch_row(branch_source_rows[l], br, orig, buses) for (l, br) in enumerate(pd.branch)]
    all(br.rate_a > 0 for br in branches) || throw(ArgumentError(
        "branches must have positive thermal limits after normalization"))

    # Normalization folds shunts into per-bus gs/bs (which the constructors consume).
    # Re-expose them as a table, one record per bus with a nonzero shunt admittance,
    # for callers that want shunt records back.
    shunt_buses = [b for b in buses if b.gs != 0.0 || b.bs != 0.0]
    shunts = [(; index=i, shunt_bus=b.bus_i, gs=b.gs, bs=b.bs) for (i, b) in enumerate(shunt_buses)]

    return (; name=PowerIO.network_name(net),
            baseMVA=_finite(pd.baseMVA, "network", :baseMVA),
            bus=buses, gen=gens, branch=branches, shunt=shunts)
end

# =============================================================================
# Absent numeric bounds
# =============================================================================
#
# PowerIO 0.9 passes an absent bound through as `±Inf` instead of refusing the case:
# `Inf` is how MATPOWER, PowerModels, pandapower and PyPSA all spell "no limit", and
# stock pglib cases carry it (case9241pegase leaves the reactive limits off seven
# generators). PowerDiff's KKT layout is fixed, with one complementarity row per
# bound, so `Inf` cannot simply flow in: `ρ * (qg - qmin)` with `qmin == -Inf` is
# `0 * Inf`, a `NaN` in the residual and an `Inf` in the Jacobian.
#
# Reactive generator limits model absence properly — the bound is left off the solver
# model and its complementarity row reads `ρ = 0`, the multiplier of a constraint that
# is not there (see `_lb_complementarity` and its derivatives in
# `prob/kkt_ac_opf.jl`). Everywhere else a non-finite value is a modeling error, and
# PowerDiff names the element and field rather than letting it reach a factorization.

"""
    _absent_bound(v) -> Bool

Whether a variable bound is absent. PowerIO spells an absent bound `±Inf`.
"""
_absent_bound(v::Real) = !isfinite(v)

function _finite(x, element::AbstractString, field::Symbol)
    v = Float64(x)
    isfinite(v) || throw(ArgumentError(
        "PowerDiff: $element has non-finite `$field` ($v). PowerDiff needs a finite " *
        "value here; only generator reactive limits may be left unbounded"))
    return v
end

# A bound may be absent; a `NaN` never is. `to_powerdata` already rejects `NaN`, so
# this only has to keep the two spellings apart for a caller-built table.
function _optional_bound(x, element::AbstractString, field::Symbol)
    v = Float64(x)
    isnan(v) && throw(ArgumentError("PowerDiff: $element has NaN `$field`"))
    return v
end

_bus_row(bus_id, b) = (;
    bus_i = bus_id, bus_type = Int(b.type),
    pd = _finite(b.pd, "bus $bus_id", :pd), qd = _finite(b.qd, "bus $bus_id", :qd),
    gs = _finite(b.gs, "bus $bus_id", :gs), bs = _finite(b.bs, "bus $bus_id", :bs),
    vm = _finite(b.vm, "bus $bus_id", :vm), va = _finite(b.va, "bus $bus_id", :va),
    vmin = _finite(b.vmin, "bus $bus_id", :vmin), vmax = _finite(b.vmax, "bus $bus_id", :vmax),
)

_gen_row(row, bus_id, g) = (;
    index = row, gen_bus = bus_id,
    pg = _finite(g.pg, "generator $row", :pg), qg = _finite(g.qg, "generator $row", :qg),
    qmin = _optional_bound(g.qmin, "generator $row", :qmin),
    qmax = _optional_bound(g.qmax, "generator $row", :qmax),
    vg = _finite(g.vg, "generator $row", :vg),
    pmin = _finite(g.pmin, "generator $row", :pmin),
    pmax = _finite(g.pmax, "generator $row", :pmax),
    cost = _poly_cost(g),
)

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

# Build one PowerDiff branch row from a normalized branch: map dense f_bus/t_bus to
# source ids, default the angle window, and synthesize a finite rate_a when the source
# states no thermal limit, using the endpoint buses' vmax limits.
#
# "States no limit" is `rate_a == 0`, MATPOWER's spelling, or a non-finite `rate_a`,
# which is how PowerIO 0.9 carries an unbounded rating out of the formats that write
# one. Both mean the same thing and both take the same synthesized limit, which is
# the largest flow the endpoint voltage limits and the angle window physically admit
# — a bound by construction, not an invented rating that could bind.
function _branch_row(l, br, orig, buses)
    what = "branch $l"
    angmin, angmax = _normalize_angle_bounds(_finite(br.angmin, what, :angmin),
                                             _finite(br.angmax, what, :angmax))
    br_r = _finite(br.br_r, what, :br_r)
    br_x = _finite(br.br_x, what, :br_x)
    raw_rate_a = Float64(br.rate_a)
    rate_a = (isfinite(raw_rate_a) && raw_rate_a > 0) ? raw_rate_a :
             _fallback_rate_a(br_r, br_x, angmin, angmax,
                              buses[br.f_bus].vmax, buses[br.t_bus].vmax)
    return (; index=l, f_bus=orig[br.f_bus], t_bus=orig[br.t_bus],
            br_r=br_r, br_x=br_x,
            br_b=_finite(br.b_fr, what, :b_fr) + _finite(br.b_to, what, :b_to),
            rate_a=rate_a,
            rate_b=_unlimited_as_zero(br.rate_b), rate_c=_unlimited_as_zero(br.rate_c),
            tap=_finite(br.tap, what, :tap), shift=_finite(br.shift, what, :shift),
            angmin=angmin, angmax=angmax)
end

# `rate_b` / `rate_c` are carried through untouched and unused by either formulation.
# Keep them numeric so a caller reading the tables never meets an `Inf`: an unbounded
# rating reads as `0`, which is the same "no limit" spelling `rate_a` arrives in.
_unlimited_as_zero(x) = (v = Float64(x); isfinite(v) ? v : 0.0)

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
DCNetwork(net::PowerIO.BalancedNetwork; tau::Float64=DEFAULT_TAU, ref_bus::Union{Nothing,Int}=nothing) =
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

calc_demand_vector(net::PowerIO.BalancedNetwork) = calc_demand_vector(_network_data(net))
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
function DCPowerFlowState(net::PowerIO.BalancedNetwork; g::Union{Nothing,AbstractVector}=nothing, d::Union{Nothing,AbstractVector}=nothing)
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
