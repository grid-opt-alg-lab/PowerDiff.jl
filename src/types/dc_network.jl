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
# DCNetwork Constructors
# =============================================================================

"""
    DCNetwork(net::Dict; tau=DEFAULT_TAU, ref_bus=nothing)

Construct a DCNetwork from a PowerModels network dictionary.

Accepts both basic and non-basic networks. Non-basic networks (with arbitrary
bus/branch/gen IDs) are automatically translated to sequential indices internally.
The original IDs are preserved in `id_map` for result interpretation.

# Example
```julia
raw = PowerModels.parse_file("case14.m")
dc_net = DCNetwork(raw)  # non-basic OK
# or
net = PowerModels.make_basic_network(raw)
dc_net = DCNetwork(net)  # basic also OK
```
"""
function DCNetwork(net::Dict; tau::Float64=DEFAULT_TAU, ref_bus::Union{Nothing,Int}=nothing)
    pm_data, id_map = _prepare_network_data(net)

    n = length(id_map.bus_ids)
    m = length(id_map.branch_ids)
    k = length(id_map.gen_ids)
    branch_tbl = pm_data["branch"]
    gen_tbl = pm_data["gen"]

    # Incidence matrix A (m × n) from active branches using id_map translation
    A = spzeros(m, n)
    for orig_id in id_map.branch_ids
        br = branch_tbl[string(orig_id)]
        row = id_map.branch_to_idx[orig_id]
        f_col = id_map.bus_to_idx[br["f_bus"]]
        t_col = id_map.bus_to_idx[br["t_bus"]]
        A[row, f_col] = 1.0
        A[row, t_col] = -1.0
    end

    # Generator-bus incidence matrix G_inc (n × k)
    G_inc = spzeros(n, k)
    for orig_id in id_map.gen_ids
        gen = gen_tbl[string(orig_id)]
        col = id_map.gen_to_idx[orig_id]
        row = id_map.bus_to_idx[gen["gen_bus"]]
        G_inc[row, col] = 1.0
    end

    # Branch susceptances: b = imag(1/z)
    b = zeros(m)
    for orig_id in id_map.branch_ids
        br = branch_tbl[string(orig_id)]
        idx = id_map.branch_to_idx[orig_id]
        r = br["br_r"]
        x = br["br_x"]
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
    fmax = [branch_tbl[string(id_map.branch_ids[i])]["rate_a"] for i in 1:m]
    gmax = [gen_tbl[string(id_map.gen_ids[i])]["pmax"] for i in 1:k]
    gmin = [gen_tbl[string(id_map.gen_ids[i])]["pmin"] for i in 1:k]

    # Phase angle difference limits
    angmax = [branch_tbl[string(id_map.branch_ids[i])]["angmax"] for i in 1:m]
    angmin = [branch_tbl[string(id_map.branch_ids[i])]["angmin"] for i in 1:m]

    # Cost coefficients (assumes polynomial cost with at least 2 terms)
    cq = [gen_tbl[string(id_map.gen_ids[i])]["cost"][1] for i in 1:k]
    cl = [gen_tbl[string(id_map.gen_ids[i])]["cost"][2] for i in 1:k]
    demand = _calc_demand_vector(pm_data, id_map)
    pg_init = _calc_generation_vector(pm_data, id_map)

    # Load-shedding cost: high penalty to discourage shedding when feasible
    marginal_cost_ub = max(maximum(2cq .* gmax .+ cl), 1.0)
    c_shed = fill(DEFAULT_SHED_COST_MULTIPLIER * marginal_cost_ub, n)

    # Reference bus (translate original ID to sequential index)
    if isnothing(ref_bus)
        ref_candidates = [id for id in id_map.bus_ids if get(pm_data["bus"][string(id)], "bus_type", 1) == 3]
        orig_ref = isempty(ref_candidates) ? id_map.bus_ids[1] : ref_candidates[1]
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

function DCNetwork(data::ParsedCase; tau::Float64=DEFAULT_TAU, ref_bus::Union{Nothing,Int}=nothing)
    return DCNetwork(_parsedcase_to_pm_data(data); tau=tau, ref_bus=ref_bus)
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
        IDMapping(n, m, k, 0)
    )
end

# =============================================================================
# DCNetwork Helper Functions
# =============================================================================

"""
    calc_demand_vector(net::Dict)

Extract demand vector from PowerModels network dictionary.

Works with both basic and non-basic networks.
"""
function calc_demand_vector(net::Dict)
    pm_data, id_map = _prepare_network_data(net)
    return _calc_demand_vector(pm_data, id_map)
end

"""
    calc_demand_vector(network::DCNetwork)

Extract demand vector from a DCNetwork.
"""
function calc_demand_vector(network::DCNetwork)
    return copy(network.demand)
end

calc_demand_vector(data::ParsedCase) = begin
    bus_to_idx = Dict(b.bus_i => i for (i, b) in enumerate(data.bus))
    d = zeros(length(data.bus))
    for load in data.load
        load.status != 0 || continue
        d[bus_to_idx[load.load_bus]] += load.pd
    end
    d
end

"""
Internal demand vector extraction from standardized network data.
"""
function _calc_demand_vector(pm_data::Dict, id_map::IDMapping)
    n = length(id_map.bus_ids)
    d = zeros(n)
    if haskey(pm_data, "load")
        for load_orig_id in id_map.load_ids
            load = pm_data["load"][string(load_orig_id)]
            bus_idx = id_map.bus_to_idx[load["load_bus"]]
            d[bus_idx] += get(load, "pd", 0.0)
        end
    else
        for (i, bus_orig_id) in enumerate(id_map.bus_ids)
            d[i] = get(pm_data["bus"][string(bus_orig_id)], "pd", 0.0)
        end
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
function _calc_generation_vector(pm_data::Dict, id_map::IDMapping)
    n = length(id_map.bus_ids)
    g = zeros(n)
    for gen_orig_id in id_map.gen_ids
        gen_data = pm_data["gen"][string(gen_orig_id)]
        bus_idx = id_map.bus_to_idx[gen_data["gen_bus"]]
        pg = get(gen_data, "pg", (gen_data["pmin"] + gen_data["pmax"]) / 2)
        g[bus_idx] += pg
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
    DCPowerFlowState(net::Dict; g=nothing, d=nothing)

Construct DCPowerFlowState from PowerModels network dictionary.

Accepts both basic and non-basic networks.
If `d` is not provided, extracts demand from the network.
If `g` is not provided, aggregates generation from gen data to buses.
"""
function DCPowerFlowState(pm_net::Dict; g::Union{Nothing,AbstractVector}=nothing, d::Union{Nothing,AbstractVector}=nothing)
    net = DCNetwork(pm_net)

    if isnothing(d)
        d = net.demand
    end

    # Aggregate generation to buses if not provided
    if isnothing(g)
        g = net.pg_init
    end

    return DCPowerFlowState(net, g, d)
end
