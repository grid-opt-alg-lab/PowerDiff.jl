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

`ACSensitivityCache` follows the same design: one KKT factorization shared across
6 parameter types (`:sw`, `:d`, `:qd`, `:cq`, `:cl`, `:fmax`), each producing a
separate cached `dz_d*` matrix. Power flow states (`DCPowerFlowState`,
`ACPowerFlowState`) have no cache at all because their sensitivities are cheap
direct algebra (reduced-Laplacian factorization or Jacobian factorization is
precomputed at construction time).

# Fields
- `solution`: Cached DCOPFSolution (or nothing if not yet solved)
- `kkt_factor`: Cached LU factorization of KKT Jacobian (or nothing)
- `dz_dd`: Full KKT derivative w.r.t. demand (or nothing)
- `dz_dcl`: Full KKT derivative w.r.t. linear cost (or nothing)
- `dz_dcq`: Full KKT derivative w.r.t. quadratic cost (or nothing)
- `dz_dsw`: Full KKT derivative w.r.t. switching (or nothing)
- `dz_dfmax`: Full KKT derivative w.r.t. flow limits (or nothing)
- `dz_db`: Full KKT derivative w.r.t. susceptances (or nothing)
- `b_r_factor`: Cached reduced susceptance factorization (topology-dependent, survives demand changes)
"""
mutable struct DCSensitivityCache
    solution::Union{Nothing,DCOPFSolution}
    kkt_factor::Union{Nothing,Factorization}
    dz_dd::Union{Nothing,Matrix{Float64}}
    dz_dcl::Union{Nothing,Matrix{Float64}}
    dz_dcq::Union{Nothing,Matrix{Float64}}
    dz_dsw::Union{Nothing,Matrix{Float64}}
    dz_dfmax::Union{Nothing,Matrix{Float64}}
    dz_db::Union{Nothing,Matrix{Float64}}
    b_r_factor::Union{Nothing,Factorization{Float64}}
end

"""
    DCSensitivityCache()

Create an empty sensitivity cache with all fields set to nothing.
"""
function DCSensitivityCache()
    return DCSensitivityCache(nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
end

"""
    invalidate!(cache::DCSensitivityCache)

Clear cached sensitivity data that depends on the current solution.
The `b_r_factor` field is preserved because it depends only on network topology
(susceptances and switching), not on demand or the optimization solution.
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

"""
    invalidate_topology!(cache::DCSensitivityCache)

Clear all cached data including topology-dependent `b_r_factor`.
Called when network topology changes (switching, susceptances).
"""
function invalidate_topology!(cache::DCSensitivityCache)
    invalidate!(cache)
    cache.b_r_factor = nothing
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
- `va`, `pg`, `f`, `psh`: Variable references for phase angles, generation, flows, load shedding
- `d`: Demand parameter (can be updated for sensitivity analysis)
- `cons`: Named tuple of constraint references
- `cache`: Mutable sensitivity cache for avoiding redundant KKT solves
- `_optimizer`: Optimizer factory for model rebuilds (internal)
- `_silent`: Whether to suppress solver output (internal)
"""
mutable struct DCOPFProblem <: AbstractOPFProblem
    model::JuMP.Model
    network::DCNetwork
    va::Vector{VariableRef}
    pg::Vector{VariableRef}
    f::Vector{VariableRef}
    psh::Vector{VariableRef}
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
    length(d) == network.n || throw(DimensionMismatch("Demand vector length $(length(d)) must match number of buses $(network.n)"))

    _warn_negative_demand(d)

    prob = DCOPFProblem(
        JuMP.Model(), network, VariableRef[], VariableRef[], VariableRef[], VariableRef[],
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
    @variable(model, va[1:n])
    @variable(model, pg[1:k])
    @variable(model, f[1:m])
    @variable(model, psh[1:n])

    # Objective: quadratic generation cost + regularization on flows + shedding penalty
    @objective(model, Min,
        sum(network.cq[i] * pg[i]^2 + network.cl[i] * pg[i] for i in 1:k) +
        (1/2) * network.tau^2 * sum(f[i]^2 for i in 1:m) +
        sum(network.c_shed[i] * psh[i] for i in 1:n)
    )

    # Constraints
    # Power balance: G_inc * pg + psh - d = B * va
    power_bal = @constraint(model, network.G_inc * pg .+ psh .- d .== B_mat * va)

    # Flow definition: f = W * A * va
    flow_def = @constraint(model, f .== W * network.A * va)

    # Flow limits
    line_lb = @constraint(model, f .>= -network.fmax)
    line_ub = @constraint(model, f .<= network.fmax)

    # Generation limits
    gen_lb = @constraint(model, pg .>= network.gmin)
    gen_ub = @constraint(model, pg .<= network.gmax)

    # Load shedding bounds: 0 ≤ psh ≤ d
    shed_lb = @constraint(model, psh .>= 0)
    shed_ub = @constraint(model, psh .<= d)

    # Reference bus
    ref_con = @constraint(model, va[network.ref_bus] == 0.0)

    # Phase angle difference limits
    phase_diff_lb = @constraint(model, network.A * va .>= network.angmin)
    phase_diff_ub = @constraint(model, network.A * va .<= network.angmax)

    prob.model = model
    prob.va = va
    prob.pg = pg
    prob.f = f
    prob.psh = psh
    prob.cons = (
        power_bal = power_bal,
        flow_def = flow_def,
        line_lb = line_lb,
        line_ub = line_ub,
        gen_lb = gen_lb,
        gen_ub = gen_ub,
        shed_lb = shed_lb,
        shed_ub = shed_ub,
        ref = ref_con,
        phase_diff_lb = phase_diff_lb,
        phase_diff_ub = phase_diff_ub,
    )

    return nothing
end

"""
    DCOPFProblem(network::DCNetwork; d=nothing, optimizer=Clarabel.Optimizer, silent=true)

Build a B-θ DC OPF problem from a DCNetwork.

If `d` is not provided and the network was constructed from a PowerModels dict,
demand is extracted from the stored `ref`.

# Example
```julia
net = DCNetwork(pm_data)
prob = DCOPFProblem(net)       # demand extracted from stored ref
prob = DCOPFProblem(net; d=d)  # explicit demand
```
"""
function DCOPFProblem(network::DCNetwork; d::Union{Nothing,AbstractVector}=nothing, kwargs...)
    if isnothing(d)
        d = calc_demand_vector(network)
    end
    return DCOPFProblem(network, d; kwargs...)
end

"""
    DCOPFProblem(net::Dict; d=nothing, kwargs...)

Convenience constructor: build DCOPFProblem directly from PowerModels dict.

Accepts both basic and non-basic networks.
If `d` is not provided, extracts demand from the network data.
"""
function DCOPFProblem(net::Dict; d::Union{Nothing,AbstractVector}=nothing, tau::Float64=DEFAULT_TAU, kwargs...)
    network = DCNetwork(net; tau=tau)
    if isnothing(d)
        d = _calc_demand_vector(network.ref, network.id_map)
    end
    return DCOPFProblem(network, d; kwargs...)
end
