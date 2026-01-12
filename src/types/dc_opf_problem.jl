# =============================================================================
# DCOPFProblem: DC OPF Problem Type and Constructors
# =============================================================================
#
# B-θ formulation of DC OPF wrapped around a JuMP model.

"""
    DCOPFProblem <: AbstractOPFProblem

B-θ formulation of DC OPF wrapped around a JuMP model.

# Fields
- `model`: JuMP Model
- `network`: DCNetwork data
- `θ`, `g`, `f`: Variable references for phase angles, generation, flows
- `d`: Demand parameter (can be updated for sensitivity analysis)
- `cons`: Named tuple of constraint references
"""
mutable struct DCOPFProblem <: AbstractOPFProblem
    model::JuMP.Model
    network::DCNetwork
    θ::Vector{VariableRef}
    g::Vector{VariableRef}
    f::Vector{VariableRef}
    d::Vector{Float64}
    cons::NamedTuple
end

# =============================================================================
# DCOPFProblem Constructors
# =============================================================================

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
