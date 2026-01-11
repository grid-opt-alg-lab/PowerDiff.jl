# DC OPF Problem Formulation (B-θ formulation)
# Uses susceptance-weighted Laplacian to preserve graphical structure

using JuMP
using SparseArrays

const DEFAULT_TAU = 1e-2

# =============================================================================
# DCNetwork Constructors
# =============================================================================

"""
    DCNetwork(net::Dict; τ=DEFAULT_TAU, ref_bus=nothing)

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
function DCNetwork(net::Dict; τ::Float64=DEFAULT_TAU, ref_bus::Union{Nothing,Int}=nothing)
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
    z = ones(m)

    # Limits
    fmax = [branch[string(i)]["rate_a"] for i in 1:m]
    gmax = [gen[string(i)]["pmax"] for i in 1:k]
    gmin = [gen[string(i)]["pmin"] for i in 1:k]

    # Phase angle difference limits
    Δθ_max = [branch[string(i)]["angmax"] for i in 1:m]
    Δθ_min = [branch[string(i)]["angmin"] for i in 1:m]

    # Cost coefficients (assumes polynomial cost with at least 2 terms)
    cq = [gen[string(i)]["cost"][1] for i in 1:k]
    cl = [gen[string(i)]["cost"][2] for i in 1:k]

    # Reference bus
    if isnothing(ref_bus)
        ref_bus = _find_reference_bus(net)
    end

    return DCNetwork(n, m, k, A, G_inc, b, z, fmax, gmax, gmin, Δθ_max, Δθ_min, cq, cl, ref_bus, τ)
end

"""
    DCNetwork(n, m, k, A, G_inc, b; kwargs...)

Direct constructor for DCNetwork with matrices and vectors.
Useful for building networks programmatically.
"""
function DCNetwork(
    n::Int, m::Int, k::Int,
    A::AbstractMatrix, G_inc::AbstractMatrix, b::AbstractVector;
    z::AbstractVector=ones(m),
    fmax::AbstractVector=fill(Inf, m),
    gmax::AbstractVector=fill(Inf, k),
    gmin::AbstractVector=zeros(k),
    Δθ_max::AbstractVector=fill(π, m),
    Δθ_min::AbstractVector=fill(-π, m),
    cq::AbstractVector=zeros(k),
    cl::AbstractVector=zeros(k),
    ref_bus::Int=1,
    τ::Float64=DEFAULT_TAU
)
    return DCNetwork(
        n, m, k,
        sparse(Float64.(A)), sparse(Float64.(G_inc)),
        Float64.(b), Float64.(z),
        Float64.(fmax), Float64.(gmax), Float64.(gmin),
        Float64.(Δθ_max), Float64.(Δθ_min),
        Float64.(cq), Float64.(cl),
        ref_bus, τ
    )
end

# =============================================================================
# DCOPFProblem
# =============================================================================

"""
    DCOPFProblem

B-θ formulation of DC OPF wrapped around a JuMP model.

# Fields
- `model`: JuMP JuMP.Model
- `network`: DCNetwork data
- `θ`, `g`, `f`: Variable references for phase angles, generation, flows
- `d`: Demand parameter (can be updated for sensitivity analysis)
- `cons`: Named tuple of constraint references
"""
mutable struct DCOPFProblem
    model::JuMP.Model
    network::DCNetwork
    θ::Vector{VariableRef}
    g::Vector{VariableRef}
    f::Vector{VariableRef}
    d::Vector{Float64}
    cons::NamedTuple
end

"""
    DCOPFProblem(network::DCNetwork, d::AbstractVector; optimizer=Clarabel.Optimizer, silent=true)

Build a B-θ DC OPF problem for the given network and demand.

# Arguments
- `network`: DCNetwork containing topology and parameters
- `d`: Demand vector (length n)
- `optimizer`: JuMP-compatible optimizer (default: Clarabel, supports Ipopt/HiGHS/Gurobi)
- `silent`: Suppress solver output (default: true)

# Example
```julia
dc_net = DCNetwork(net)
d = calc_demand_vector(net)
prob = DCOPFProblem(dc_net, d)
solve!(prob)
```
"""
function DCOPFProblem(network::DCNetwork, d::AbstractVector; optimizer=Clarabel.Optimizer, silent::Bool=true)
    n, m, k = network.n, network.m, network.k
    @assert length(d) == n "Demand vector length must match number of buses"

    # Build susceptance matrix B = A' * W * A
    W = Diagonal(-network.b .* network.z)
    B_mat = sparse(network.A' * W * network.A)

    # Create model
    model = JuMP.Model(optimizer)
    silent && set_silent(model)

    # Variables
    @variable(model, θ[1:n])
    @variable(model, g[1:k])
    @variable(model, f[1:m])

    # Objective: quadratic generation cost + regularization on flows
    @objective(model, Min,
        sum(network.cq[i] * g[i]^2 + network.cl[i] * g[i] for i in 1:k) +
        (1/2) * network.τ^2 * sum(f[i]^2 for i in 1:m)
    )

    # Constraints
    # Power balance: G_inc * g - d = B * θ
    power_bal = @constraint(model, network.G_inc * g .- d .== B_mat * θ)

    # Flow definition: f = W * A * θ
    flow_def = @constraint(model, f .== W * network.A * θ)

    # Flow limits
    line_lb = @constraint(model, f .>= -network.fmax)
    line_ub = @constraint(model, f .<= network.fmax)

    # Generation limits
    gen_lb = @constraint(model, g .>= network.gmin)
    gen_ub = @constraint(model, g .<= network.gmax)

    # Reference bus
    ref_con = @constraint(model, θ[network.ref_bus] == 0.0)

    # Phase angle difference limits
    phase_diff = @constraint(model, network.Δθ_min .<= network.A * θ .<= network.Δθ_max)

    cons = (
        power_bal = power_bal,
        flow_def = flow_def,
        line_lb = line_lb,
        line_ub = line_ub,
        gen_lb = gen_lb,
        gen_ub = gen_ub,
        ref = ref_con,
        phase_diff = phase_diff
    )

    return DCOPFProblem(model, network, θ, g, f, Float64.(d), cons)
end

"""
    DCOPFProblem(net::Dict; d=nothing, kwargs...)

Convenience constructor: build DCOPFProblem directly from PowerModels dict.

If `d` is not provided, extracts demand from the network data.
"""
function DCOPFProblem(net::Dict; d::Union{Nothing,AbstractVector}=nothing, τ::Float64=DEFAULT_TAU, kwargs...)
    network = DCNetwork(net; τ=τ)
    if isnothing(d)
        d = calc_demand_vector(net)
    end
    return DCOPFProblem(network, d; kwargs...)
end

# =============================================================================
# Solving and Solution Extraction
# =============================================================================

"""
    solve!(prob::DCOPFProblem)

Solve the DC OPF problem and return a DCOPFSolution.

# Returns
DCOPFSolution containing optimal primal and dual variables.

# Throws
Error if optimization does not converge to optimal/locally optimal solution.
"""
function solve!(prob::DCOPFProblem)::DCOPFSolution
    optimize!(prob.model)

    status = termination_status(prob.model)
    @assert status ∈ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED] "Optimization failed with status: $status"

    # Extract primal variables
    θ_val = value.(prob.θ)
    g_val = value.(prob.g)
    f_val = value.(prob.f)

    # Extract dual variables (using dual() for ≤ constraints, shadow_price for equality)
    ν_bal = dual.(prob.cons.power_bal)
    λ_ub = dual.(prob.cons.line_ub)
    λ_lb = dual.(prob.cons.line_lb)
    ρ_ub = dual.(prob.cons.gen_ub)
    ρ_lb = dual.(prob.cons.gen_lb)

    obj = objective_value(prob.model)

    return DCOPFSolution(θ_val, g_val, f_val, ν_bal, λ_ub, λ_lb, ρ_ub, ρ_lb, obj)
end

"""
    update_demand!(prob::DCOPFProblem, d::AbstractVector)

Update the demand parameter in the DC OPF problem.

This modifies the RHS of power balance constraints for re-solving with new demand.
"""
function update_demand!(prob::DCOPFProblem, d::AbstractVector)
    n = prob.network.n
    @assert length(d) == n "Demand vector length must match number of buses"

    # Update stored demand
    prob.d .= d

    # Update constraint RHS: G_inc * g - d = B * θ  →  G_inc * g - B * θ = d
    # The constraint is stored as: G_inc * g - B * θ - d = 0
    # We need to update the constant term
    W = Diagonal(-prob.network.b .* prob.network.z)
    B_mat = sparse(prob.network.A' * W * prob.network.A)

    for i in 1:n
        set_normalized_rhs(prob.cons.power_bal[i], d[i])
    end

    return prob
end

# =============================================================================
# Helper Functions
# =============================================================================

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

Compute the susceptance matrix B = A' * Diagonal(-b .* z) * A.
"""
function calc_susceptance_matrix(network::DCNetwork)
    W = Diagonal(-network.b .* network.z)
    return sparse(network.A' * W * network.A)
end

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
