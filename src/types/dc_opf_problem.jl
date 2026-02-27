# =============================================================================
# DCOPFProblem: DC OPF Problem Type and Constructors
# =============================================================================
#
# B-θ formulation of DC OPF wrapped around a JuMP model.

# =============================================================================
# Sensitivity Cache
# =============================================================================

"""
    DCSensitivityCache

Mutable cache for DC OPF sensitivity data to avoid redundant KKT solves.

DC OPF supports 6 parameter types (`:d`, `:sw`, `:cl`, `:cq`, `:fmax`, `:b`),
each producing a separate `dz_d*` full-derivative matrix. All share one KKT LU
factorization (`kkt_factor`), so after the first parameter type is queried the
factorization is reused for subsequent parameters. Different operand queries
(e.g. `:va` vs `:pg` vs `:lmp`) for the *same* parameter type all extract
rows from the same cached `dz_d*` matrix — no recomputation needed.

By contrast, `ACSensitivityCache` only needs 2 fields because AC OPF currently
supports only switching (`:sw`) as a parameter, and `dz_dsw` already contains all
operand rows. Power flow states (`DCPowerFlowState`, `ACPowerFlowState`) have no
cache at all because their sensitivities are cheap direct algebra (pseudoinverse
or Jacobian factorization is precomputed at construction time).

# Fields
- `solution`: Cached DCOPFSolution (or nothing if not yet solved)
- `kkt_factor`: Cached LU factorization of KKT Jacobian (or nothing)
- `dz_dd`: Full KKT derivative w.r.t. demand (or nothing)
- `dz_dcl`: Full KKT derivative w.r.t. linear cost (or nothing)
- `dz_dcq`: Full KKT derivative w.r.t. quadratic cost (or nothing)
- `dz_dsw`: Full KKT derivative w.r.t. switching (or nothing)
- `dz_dfmax`: Full KKT derivative w.r.t. flow limits (or nothing)
- `dz_db`: Full KKT derivative w.r.t. susceptances (or nothing)
"""
mutable struct DCSensitivityCache
    solution::Union{Nothing,DCOPFSolution}
    kkt_factor::Union{Nothing,LinearAlgebra.LU}
    dz_dd::Union{Nothing,Matrix{Float64}}
    dz_dcl::Union{Nothing,Matrix{Float64}}
    dz_dcq::Union{Nothing,Matrix{Float64}}
    dz_dsw::Union{Nothing,Matrix{Float64}}
    dz_dfmax::Union{Nothing,Matrix{Float64}}
    dz_db::Union{Nothing,Matrix{Float64}}
end

# Deprecation alias
function SensitivityCache(args...)
    Base.depwarn("`SensitivityCache` is deprecated, use `DCSensitivityCache` instead.", :SensitivityCache)
    return DCSensitivityCache(args...)
end

"""
    DCSensitivityCache()

Create an empty sensitivity cache with all fields set to nothing.
"""
function DCSensitivityCache()
    return DCSensitivityCache(nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
end

"""
    invalidate!(cache::DCSensitivityCache)

Clear all cached sensitivity data. Called when problem parameters change.
"""
function invalidate!(cache::DCSensitivityCache)
    cache.solution = nothing
    cache.kkt_factor = nothing
    cache.dz_dd = nothing
    cache.dz_dcl = nothing
    cache.dz_dcq = nothing
    cache.dz_dsw = nothing
    cache.dz_dfmax = nothing
    cache.dz_db = nothing
    return nothing
end

# =============================================================================
# DCOPFProblem
# =============================================================================

"""
    DCOPFProblem <: AbstractOPFProblem

B-θ formulation of DC OPF wrapped around a JuMP model.

# Fields
- `model`: JuMP Model
- `network`: DCNetwork data
- `θ`, `g`, `f`: Variable references for phase angles, generation, flows
- `d`: Demand parameter (can be updated for sensitivity analysis)
- `cons`: Named tuple of constraint references
- `cache`: Mutable sensitivity cache for avoiding redundant KKT solves
- `_optimizer`: Optimizer factory for model rebuilds (internal)
- `_silent`: Whether to suppress solver output (internal)
"""
mutable struct DCOPFProblem <: AbstractOPFProblem
    model::JuMP.Model
    network::DCNetwork
    θ::Vector{VariableRef}
    g::Vector{VariableRef}
    f::Vector{VariableRef}
    d::Vector{Float64}
    cons::NamedTuple
    cache::DCSensitivityCache
    _optimizer::Any
    _silent::Bool
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
    @assert length(d) == network.n "Demand vector length must match number of buses"

    prob = DCOPFProblem(
        JuMP.Model(), network, VariableRef[], VariableRef[], VariableRef[],
        Float64.(d), (;), DCSensitivityCache(), optimizer, silent
    )
    _rebuild_jump_model!(prob)
    return prob
end

"""
    _rebuild_jump_model!(prob::DCOPFProblem)

Build (or rebuild) the JuMP model from current network parameters.
Called by the constructor and by `update_switching!` after mutating `network.sw`.
"""
function _rebuild_jump_model!(prob::DCOPFProblem)
    network = prob.network
    n, m, k = network.n, network.m, network.k
    d = prob.d

    # Build susceptance matrix B = A' * W * A
    W = Diagonal(-network.b .* network.sw)
    B_mat = sparse(network.A' * W * network.A)

    # Create model
    model = JuMP.Model(prob._optimizer)
    prob._silent && set_silent(model)

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

    prob.model = model
    prob.θ = θ
    prob.g = g
    prob.f = f
    prob.cons = (
        power_bal = power_bal,
        flow_def = flow_def,
        line_lb = line_lb,
        line_ub = line_ub,
        gen_lb = gen_lb,
        gen_ub = gen_ub,
        ref = ref_con,
        phase_diff = phase_diff
    )

    return nothing
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
