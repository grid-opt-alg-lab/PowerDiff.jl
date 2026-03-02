# =============================================================================
# DCNetwork: DC Network Data Structure
# =============================================================================
#
# DC network representation for B-theta OPF formulation with susceptance-weighted
# Laplacian L = A' * Diag(-b .* sw) * A.

"""
    DCNetwork <: AbstractPowerNetwork

DC network data for B-theta OPF formulation. Uses susceptance-weighted Laplacian
`B = A' * Diagonal(-b .* sw) * A` which preserves graphical structure for
topology sensitivity analysis and integration with RandomizedSwitching tools.

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
    ref_bus::Int
    tau::Float64
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
- `objective`: Optimal objective value
"""
struct DCOPFSolution <: AbstractOPFSolution
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
    objective::Float64
end

"""
    DCPowerFlowState{F} <: AbstractPowerFlowState

DC power flow solution (phase angles from reduced-Laplacian solve, no optimization).
Supports both generation and demand for flexible sensitivity analysis.

Unlike DCOPFSolution, this represents a simple power flow solution
`θ_r = L_r \\ p_r` where `L_r` is the Laplacian with the reference bus row and
column deleted (invertible for a connected network), without optimal dispatch or
constraint handling.

# Fields
- `net`: DCNetwork data
- `va`: Phase angles (rad), with `va[ref_bus] = 0`
- `p`: Net injection vector (p = pg - d)
- `pg`: Generation vector
- `d`: Demand vector
- `f`: Branch flows (computed from va)
- `L_r_factor`: LU factorization of `L[non_ref, non_ref]`
- `non_ref`: Indices of non-reference buses
"""
struct DCPowerFlowState{F<:Factorization{Float64}} <: AbstractPowerFlowState
    net::DCNetwork
    va::Vector{Float64}
    p::Vector{Float64}
    pg::Vector{Float64}
    d::Vector{Float64}
    f::Vector{Float64}
    L_r_factor::F
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

The network must be a basic network (`net["basic_network"] == true`).
If `ref_bus` is not specified, uses the reference bus from the network data.

# Example
```julia
raw = PowerModels.parse_file("case14.m")
net = PowerModels.make_basic_network(raw)
dc_net = DCNetwork(net)
```
"""
function DCNetwork(net::Dict; tau::Float64=DEFAULT_TAU, ref_bus::Union{Nothing,Int}=nothing)
    @assert haskey(net, "basic_network") && net["basic_network"] == true "Network must be a basic network (use make_basic_network)"

    gen = net["gen"]
    load = net["load"]
    branch = net["branch"]

    # Dimensions
    n = length(net["bus"])
    m = length(branch)
    k = length(gen)

    # Incidence matrix (m × n)
    A = Float64.(PM.calc_basic_incidence_matrix(net))

    # Generator-bus incidence matrix (n × k)
    G_inc = _calc_gen_incidence_matrix(gen, n, k)

    # Branch susceptances: b = imag(1/z)
    z_branch = PM.calc_basic_branch_series_impedance(net)
    b = imag.(inv.(z_branch))

    # All branches initially active
    sw = ones(m)

    # Limits
    fmax = [branch[string(i)]["rate_a"] for i in 1:m]
    gmax = [gen[string(i)]["pmax"] for i in 1:k]
    gmin = [gen[string(i)]["pmin"] for i in 1:k]

    # Phase angle difference limits
    angmax = [branch[string(i)]["angmax"] for i in 1:m]
    angmin = [branch[string(i)]["angmin"] for i in 1:m]

    # Cost coefficients (assumes polynomial cost with at least 2 terms)
    cq = [gen[string(i)]["cost"][1] for i in 1:k]
    cl = [gen[string(i)]["cost"][2] for i in 1:k]

    # Load-shedding cost: high penalty to discourage shedding when feasible
    marginal_cost_ub = max(maximum(2cq .* gmax .+ cl), 1.0)
    c_shed = fill(DEFAULT_SHED_COST_MULTIPLIER * marginal_cost_ub, n)

    # Reference bus
    if isnothing(ref_bus)
        ref_bus = _find_reference_bus(net)
    end

    return DCNetwork(n, m, k, A, G_inc, b, sw, fmax, gmax, gmin, angmax, angmin, cq, cl, c_shed, ref_bus, tau)
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
    ref_bus::Int=1,
    tau::Float64=DEFAULT_TAU
)
    @assert length(c_shed) == n "c_shed length ($(length(c_shed))) must match number of buses ($n)"
    @assert all(c_shed .> 0) "c_shed must be strictly positive at all buses"
    return DCNetwork(
        n, m, k,
        sparse(Float64.(A)), sparse(Float64.(G_inc)),
        Float64.(b), Float64.(sw),
        Float64.(fmax), Float64.(gmax), Float64.(gmin),
        Float64.(angmax), Float64.(angmin),
        Float64.(cq), Float64.(cl),
        Float64.(c_shed),
        ref_bus, tau
    )
end

# =============================================================================
# DCNetwork Helper Functions
# =============================================================================

"""
Generator-bus incidence matrix (n × k).
"""
function _calc_gen_incidence_matrix(gen::Dict, n::Int, k::Int)
    G_inc = spzeros(n, k)
    for i in 1:k
        bus_idx = gen[string(i)]["gen_bus"]
        G_inc[bus_idx, i] = 1.0
    end
    return G_inc
end

"""
Find reference bus from network data.
"""
function _find_reference_bus(net::Dict)
    for (_, bus) in net["bus"]
        if bus["bus_type"] == 3  # Reference bus type in MATPOWER
            return bus["bus_i"]
        end
    end
    # Default to bus 1 if no reference found
    return 1
end

"""
    calc_demand_vector(net::Dict)

Extract demand vector from PowerModels network dictionary.
"""
function calc_demand_vector(net::Dict)
    n = length(net["bus"])
    load = net["load"]
    n_load = length(load)

    d = zeros(n)
    for i in 1:n_load
        load_data = load[string(i)]
        bus_idx = load_data["load_bus"]
        d[bus_idx] = load_data["pd"]
    end

    return d
end

"""
    calc_susceptance_matrix(network::DCNetwork)

Compute the susceptance matrix B = A' * Diagonal(-b .* sw) * A.
"""
function calc_susceptance_matrix(network::DCNetwork)
    W = Diagonal(-network.b .* network.sw)
    return sparse(network.A' * W * network.A)
end

"""
Aggregate generation to bus-level vector.
"""
function _calc_generation_vector(net::Dict, n::Int)
    gen = net["gen"]
    k = length(gen)

    g = zeros(n)
    for i in 1:k
        gen_data = gen[string(i)]
        bus_idx = gen_data["gen_bus"]
        # Use pg if available (from solved power flow), otherwise use midpoint of limits
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
    L_r * θ_r = p_r
where L_r is the susceptance-weighted Laplacian with the reference bus row and
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
d = calc_demand_vector(pm_data)
g = zeros(net.n)  # Or specify generation at each bus
state = DCPowerFlowState(net, g, d)
```
"""
function DCPowerFlowState(net::DCNetwork, g::AbstractVector{<:Real}, d::AbstractVector{<:Real})
    n, m = net.n, net.m
    @assert length(g) == n "Generation vector length must match number of buses"
    @assert length(d) == n "Demand vector length must match number of buses"

    # Net injection
    p = Float64.(g) - Float64.(d)

    # Build reduced Laplacian (delete reference bus row/col) and factorize
    L = calc_susceptance_matrix(net)
    non_ref = setdiff(1:n, net.ref_bus)
    F = lu(L[non_ref, non_ref])   # sparse LU (UmfpackLU)

    # Solve reduced system: θ[non_ref] = L_r \ p[non_ref], θ[ref] = 0
    θ = zeros(n)
    θ[non_ref] = F \ p[non_ref]

    # Compute flows: f = W * A * θ where W = Diag(-b ⊙ sw)
    W = Diagonal(-net.b .* net.sw)
    f = W * net.A * θ

    return DCPowerFlowState(net, θ, p, Float64.(g), Float64.(d), f, F, non_ref)
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

If `d` is not provided, extracts demand from the network.
If `g` is not provided, aggregates generation from gen data to buses.
"""
function DCPowerFlowState(pm_net::Dict; g::Union{Nothing,AbstractVector}=nothing, d::Union{Nothing,AbstractVector}=nothing)
    net = DCNetwork(pm_net)

    # Extract demand if not provided
    if isnothing(d)
        d = calc_demand_vector(pm_net)
    end

    # Aggregate generation to buses if not provided
    if isnothing(g)
        g = _calc_generation_vector(pm_net, net.n)
    end

    return DCPowerFlowState(net, g, d)
end
