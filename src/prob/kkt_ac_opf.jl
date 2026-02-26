# =============================================================================
# KKT System for AC OPF
# =============================================================================
#
# Implements KKT conditions for implicit differentiation of AC OPF solutions.
# Design mirrors kkt_dc_opf.jl for consistency.
#
# Note: Uses reduced-space formulation where branch flows p_fr, q_fr, p_to, q_to
# are functions of voltage state (va, vm), not separate primal variables.
# This matches the PowerModels formulation used in ac_opf_problem.jl.

# =============================================================================
# Dimension Calculations
# =============================================================================

"""
    ac_kkt_dims(prob::ACOPFProblem)

Compute the dimension of the flattened KKT variable vector for AC OPF.

The KKT system includes:
- Primal: va (n), vm (n), pg (k), qg (k)
- Dual (equality): ν_p_bal (n), ν_q_bal (n)
- Dual (inequality): λ_thermal_fr (m), λ_thermal_to (m), μ_vm_lb (n), μ_vm_ub (n),
                     ρ_pg_lb (k), ρ_pg_ub (k), ρ_qg_lb (k), ρ_qg_ub (k)

Total: 2n + 2k + 2n + 2m + 2n + 4k = 6n + 2m + 6k
"""
function ac_kkt_dims(prob::ACOPFProblem)
    n, m, k = prob.network.n, prob.network.m, prob.n_gen
    # primal: va(n) + vm(n) + pg(k) + qg(k) = 2n + 2k
    # dual equality: ν_p_bal(n) + ν_q_bal(n) = 2n
    # dual inequality: λ_thermal_fr(m) + λ_thermal_to(m) + μ_vm_lb(n) + μ_vm_ub(n) +
    #                  ρ_pg_lb(k) + ρ_pg_ub(k) + ρ_qg_lb(k) + ρ_qg_ub(k) = 2m + 2n + 4k
    return 6n + 2m + 6k
end

"""
    ac_kkt_indices(n, m, k) → NamedTuple

Compute all KKT variable indices from problem dimensions.
Single source of truth for index calculations.

# Variable ordering
[va(n), vm(n), pg(k), qg(k),
 ν_p_bal(n), ν_q_bal(n),
 λ_thermal_fr(m), λ_thermal_to(m), μ_vm_lb(n), μ_vm_ub(n),
 ρ_pg_lb(k), ρ_pg_ub(k), ρ_qg_lb(k), ρ_qg_ub(k)]

# Returns
NamedTuple with index ranges for each variable block.
"""
function ac_kkt_indices(n::Int, m::Int, k::Int)
    i = 0
    # Primal
    idx_va = (i+1):(i+n); i += n
    idx_vm = (i+1):(i+n); i += n
    idx_pg = (i+1):(i+k); i += k
    idx_qg = (i+1):(i+k); i += k

    # Dual (equality)
    idx_ν_p_bal = (i+1):(i+n); i += n
    idx_ν_q_bal = (i+1):(i+n); i += n

    # Dual (inequality)
    idx_λ_thermal_fr = (i+1):(i+m); i += m
    idx_λ_thermal_to = (i+1):(i+m); i += m
    idx_μ_vm_lb = (i+1):(i+n); i += n
    idx_μ_vm_ub = (i+1):(i+n); i += n
    idx_ρ_pg_lb = (i+1):(i+k); i += k
    idx_ρ_pg_ub = (i+1):(i+k); i += k
    idx_ρ_qg_lb = (i+1):(i+k); i += k
    idx_ρ_qg_ub = (i+1):(i+k); i += k

    return (
        va = idx_va, vm = idx_vm, pg = idx_pg, qg = idx_qg,
        ν_p_bal = idx_ν_p_bal, ν_q_bal = idx_ν_q_bal,
        λ_thermal_fr = idx_λ_thermal_fr, λ_thermal_to = idx_λ_thermal_to,
        μ_vm_lb = idx_μ_vm_lb, μ_vm_ub = idx_μ_vm_ub,
        ρ_pg_lb = idx_ρ_pg_lb, ρ_pg_ub = idx_ρ_pg_ub,
        ρ_qg_lb = idx_ρ_qg_lb, ρ_qg_ub = idx_ρ_qg_ub
    )
end

ac_kkt_indices(prob::ACOPFProblem) = ac_kkt_indices(prob.network.n, prob.network.m, prob.n_gen)

# =============================================================================
# Variable Flattening/Unflattening
# =============================================================================

"""
    ac_flatten_variables(sol::ACOPFSolution, prob::ACOPFProblem)

Flatten solution primal and dual variables into a single vector for KKT evaluation.

# Variable ordering
[va; vm; pg; qg; ν_p_bal; ν_q_bal;
 λ_thermal_fr; λ_thermal_to; μ_vm_lb; μ_vm_ub; ρ_pg_lb; ρ_pg_ub; ρ_qg_lb; ρ_qg_ub]
"""
function ac_flatten_variables(sol::ACOPFSolution, prob::ACOPFProblem)
    return vcat(
        sol.va,
        sol.vm,
        sol.pg,
        sol.qg,
        sol.ν_p_bal,
        sol.ν_q_bal,
        sol.λ_thermal_fr,
        sol.λ_thermal_to,
        sol.μ_vm_lb,
        sol.μ_vm_ub,
        sol.ρ_pg_lb,
        sol.ρ_pg_ub,
        sol.ρ_qg_lb,
        sol.ρ_qg_ub
    )
end

"""
    ac_unflatten_variables(z::AbstractVector, prob::ACOPFProblem)

Unflatten KKT variable vector into named components.

# Returns
NamedTuple with fields for all primal and dual variables.
"""
function ac_unflatten_variables(z::AbstractVector, prob::ACOPFProblem)
    n, m, k = prob.network.n, prob.network.m, prob.n_gen
    idx = ac_kkt_indices(n, m, k)

    return (
        va = z[idx.va],
        vm = z[idx.vm],
        pg = z[idx.pg],
        qg = z[idx.qg],
        ν_p_bal = z[idx.ν_p_bal],
        ν_q_bal = z[idx.ν_q_bal],
        λ_thermal_fr = z[idx.λ_thermal_fr],
        λ_thermal_to = z[idx.λ_thermal_to],
        μ_vm_lb = z[idx.μ_vm_lb],
        μ_vm_ub = z[idx.μ_vm_ub],
        ρ_pg_lb = z[idx.ρ_pg_lb],
        ρ_pg_ub = z[idx.ρ_pg_ub],
        ρ_qg_lb = z[idx.ρ_qg_lb],
        ρ_qg_ub = z[idx.ρ_qg_ub]
    )
end

# =============================================================================
# KKT Jacobian via ForwardDiff
# =============================================================================

"""
    calc_ac_kkt_jacobian(prob::ACOPFProblem; sol=nothing)

Compute the Jacobian of the KKT operator using ForwardDiff.

For AC OPF, the KKT system is nonlinear (contains trig functions and products),
so we use automatic differentiation rather than analytical derivatives.

# Arguments
- `prob`: ACOPFProblem
- `sol`: Optional pre-computed solution. If not provided, calls solve!(prob).

# Returns
Matrix ∂K/∂z where z is the flattened variable vector.
"""
function calc_ac_kkt_jacobian(prob::ACOPFProblem; sol::Union{ACOPFSolution,Nothing}=nothing)
    if isnothing(sol)
        sol = solve!(prob)
    end

    # Flatten solution to get the point at which to compute Jacobian
    z0 = ac_flatten_variables(sol, prob)

    # Use ForwardDiff to compute Jacobian
    # Pass current switching state as fixed parameter
    z_switch = prob.network.z
    J = ForwardDiff.jacobian(z -> ac_kkt(z, prob, z_switch), z0)

    return J
end

"""
    ac_kkt(z::AbstractVector, prob::ACOPFProblem, z_switch::AbstractVector)

Evaluate the KKT conditions for AC OPF at the given variable vector.

The switching state z_switch is passed as a separate parameter to enable
differentiation with respect to switching using ForwardDiff.

Returns a vector of KKT residuals (should be zero at optimum).

# KKT Conditions (reduced-space formulation)
1. Stationarity w.r.t. va, vm, pg, qg
2. Primal feasibility: power balance
3. Complementary slackness for thermal limits, voltage bounds, generation bounds

Note: Branch flows are computed as functions of voltages, not separate variables.
"""
function ac_kkt(z::AbstractVector, prob::ACOPFProblem, z_switch::AbstractVector)
    vars = ac_unflatten_variables(z, prob)
    net = prob.network
    ref = prob.ref
    n, m, k = net.n, net.m, prob.n_gen

    va, vm = vars.va, vars.vm
    pg, qg = vars.pg, vars.qg
    ν_p_bal, ν_q_bal = vars.ν_p_bal, vars.ν_q_bal
    λ_thermal_fr, λ_thermal_to = vars.λ_thermal_fr, vars.λ_thermal_to
    μ_vm_lb, μ_vm_ub = vars.μ_vm_lb, vars.μ_vm_ub
    ρ_pg_lb, ρ_pg_ub = vars.ρ_pg_lb, vars.ρ_pg_ub
    ρ_qg_lb, ρ_qg_ub = vars.ρ_qg_lb, vars.ρ_qg_ub

    T = promote_type(eltype(z), eltype(z_switch))

    # Compute branch flows as functions of voltages
    p_fr, q_fr, p_to, q_to = _compute_branch_flows(va, vm, net, ref, z_switch)

    # Build KKT residual vector
    K = T[]

    # =========================================================================
    # 1. Stationarity conditions
    # =========================================================================

    # Stationarity w.r.t. va: ∂L/∂va = 0
    # The Lagrangian gradient w.r.t. va involves:
    # - Power balance constraint gradients: ∂p_bal/∂va, ∂q_bal/∂va
    # - Thermal limit gradients: ∂(p² + q²)/∂va
    K_va = _stationarity_va(va, vm, pg, qg, ν_p_bal, ν_q_bal,
                            λ_thermal_fr, λ_thermal_to,
                            p_fr, q_fr, p_to, q_to, net, ref, prob, z_switch)
    append!(K, K_va)

    # Stationarity w.r.t. vm: ∂L/∂vm = 0
    K_vm = _stationarity_vm(va, vm, pg, qg, ν_p_bal, ν_q_bal,
                            λ_thermal_fr, λ_thermal_to, μ_vm_lb, μ_vm_ub,
                            p_fr, q_fr, p_to, q_to, net, ref, prob, z_switch)
    append!(K, K_vm)

    # Stationarity w.r.t. pg: ∂L/∂pg = cost_grad - ν_p_bal[gen_bus] + ρ_pg_ub - ρ_pg_lb = 0
    K_pg = _stationarity_pg(pg, ν_p_bal, ρ_pg_lb, ρ_pg_ub, ref, prob)
    append!(K, K_pg)

    # Stationarity w.r.t. qg: ∂L/∂qg = -ν_q_bal[gen_bus] + ρ_qg_ub - ρ_qg_lb = 0
    K_qg = _stationarity_qg(qg, ν_q_bal, ρ_qg_lb, ρ_qg_ub, ref, prob)
    append!(K, K_qg)

    # =========================================================================
    # 2. Primal feasibility: Power balance
    # =========================================================================
    K_p_bal, K_q_bal = _power_balance_residuals(va, vm, pg, qg, p_fr, q_fr, p_to, q_to, net, ref, prob)
    append!(K, K_p_bal)
    append!(K, K_q_bal)

    # =========================================================================
    # 3. Complementary slackness conditions
    # =========================================================================

    # Thermal limits: λ * (p² + q² - rate_a²) = 0
    for l in 1:m
        rate_a = ref[:branch][l]["rate_a"]
        slack_fr = p_fr[l]^2 + q_fr[l]^2 - rate_a^2
        slack_to = p_to[l]^2 + q_to[l]^2 - rate_a^2
        push!(K, λ_thermal_fr[l] * slack_fr)
        push!(K, λ_thermal_to[l] * slack_to)
    end

    # Voltage bounds: μ * (vm - bound) = 0
    for i in 1:n
        vm_min = ref[:bus][i]["vmin"]
        vm_max = ref[:bus][i]["vmax"]
        push!(K, μ_vm_lb[i] * (vm[i] - vm_min))
        push!(K, μ_vm_ub[i] * (vm_max - vm[i]))
    end

    # Generation bounds: ρ * (g - bound) = 0
    for i in 1:k
        pg_min = ref[:gen][i]["pmin"]
        pg_max = ref[:gen][i]["pmax"]
        qg_min = ref[:gen][i]["qmin"]
        qg_max = ref[:gen][i]["qmax"]
        push!(K, ρ_pg_lb[i] * (pg[i] - pg_min))
        push!(K, ρ_pg_ub[i] * (pg_max - pg[i]))
        push!(K, ρ_qg_lb[i] * (qg[i] - qg_min))
        push!(K, ρ_qg_ub[i] * (qg_max - qg[i]))
    end

    return K
end

# Convenience method using prob's switching state
ac_kkt(z::AbstractVector, prob::ACOPFProblem) = ac_kkt(z, prob, prob.network.z)

# =============================================================================
# KKT Helper Functions - Branch Flow Calculations
# =============================================================================

"""
Compute all branch flows given voltage state and switching state.
Returns vectors of p_fr, q_fr, p_to, q_to indexed by branch number.

The switching variable z_l multiplies each flow, so z_l=0 means the branch
contributes zero flow (open), z_l=1 means full flow (closed).
"""
function _compute_branch_flows(va, vm, net::ACNetwork, ref, z_switch)
    m = net.m
    T = promote_type(eltype(va), eltype(vm), eltype(z_switch))
    p_fr = zeros(T, m)
    q_fr = zeros(T, m)
    p_to = zeros(T, m)
    q_to = zeros(T, m)

    for (l, branch) in ref[:branch]
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]

        g_br, b_br = PM.calc_branch_y(branch)
        tr, ti = PM.calc_branch_t(branch)
        g_fr_shunt = branch["g_fr"]
        b_fr_shunt = branch["b_fr"]
        g_to_shunt = branch["g_to"]
        b_to_shunt = branch["b_to"]
        tm = branch["tap"]^2

        z_l = z_switch[l]

        vm_fr = vm[f_bus]
        vm_to = vm[t_bus]
        va_fr = va[f_bus]
        va_to = va[t_bus]

        # From side
        p_fr[l] = z_l * ((g_br + g_fr_shunt)/tm * vm_fr^2 +
                  (-g_br*tr + b_br*ti)/tm * (vm_fr * vm_to * cos(va_fr - va_to)) +
                  (-b_br*tr - g_br*ti)/tm * (vm_fr * vm_to * sin(va_fr - va_to)))

        q_fr[l] = z_l * (-(b_br + b_fr_shunt)/tm * vm_fr^2 -
                  (-b_br*tr - g_br*ti)/tm * (vm_fr * vm_to * cos(va_fr - va_to)) +
                  (-g_br*tr + b_br*ti)/tm * (vm_fr * vm_to * sin(va_fr - va_to)))

        # To side
        p_to[l] = z_l * ((g_br + g_to_shunt) * vm_to^2 +
                  (-g_br*tr - b_br*ti)/tm * (vm_to * vm_fr * cos(va_to - va_fr)) +
                  (-b_br*tr + g_br*ti)/tm * (vm_to * vm_fr * sin(va_to - va_fr)))

        q_to[l] = z_l * (-(b_br + b_to_shunt) * vm_to^2 -
                  (-b_br*tr + g_br*ti)/tm * (vm_to * vm_fr * cos(va_fr - va_to)) +
                  (-g_br*tr - b_br*ti)/tm * (vm_to * vm_fr * sin(va_to - va_fr)))
    end

    return p_fr, q_fr, p_to, q_to
end

# =============================================================================
# KKT Helper Functions - Stationarity Conditions
# =============================================================================

"""
Stationarity w.r.t. voltage angles va.

∂L/∂va_i = Σ_l (∂p_flow_l/∂va_i · contribution_from_constraints) = 0

In reduced-space formulation, flows depend on va, so stationarity includes
chain rule through power balance and thermal constraints.
"""
function _stationarity_va(va, vm, pg, qg, ν_p_bal, ν_q_bal,
                          λ_thermal_fr, λ_thermal_to,
                          p_fr, q_fr, p_to, q_to,
                          net::ACNetwork, ref, prob::ACOPFProblem, z_switch)
    n, m = net.n, net.m
    T = promote_type(eltype(va), eltype(vm), eltype(z_switch), eltype(p_fr))
    K_va = zeros(T, n)

    # For each bus i, compute ∂L/∂va_i
    # Contributions come from:
    # 1. Power balance constraints (flows depend on va)
    # 2. Thermal limit constraints (flows depend on va)

    for (l, branch) in ref[:branch]
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        rate_a = branch["rate_a"]

        # Get branch parameters
        g_br, b_br = PM.calc_branch_y(branch)
        tr, ti = PM.calc_branch_t(branch)
        tm = branch["tap"]^2

        z_l = z_switch[l]

        vm_fr = vm[f_bus]
        vm_to = vm[t_bus]
        va_fr = va[f_bus]
        va_to = va[t_bus]
        θ_diff = va_fr - va_to

        # Derivatives of flows w.r.t. va_fr and va_to
        # p_fr = z_l * (... + A*cos(θ_diff) + B*sin(θ_diff))
        # ∂p_fr/∂va_fr = z_l * (-A*sin(θ_diff) + B*cos(θ_diff)) * vm_fr * vm_to
        # ∂p_fr/∂va_to = z_l * (A*sin(θ_diff) - B*cos(θ_diff)) * vm_fr * vm_to

        A_p = (-g_br*tr + b_br*ti)/tm
        B_p = (-b_br*tr - g_br*ti)/tm
        A_q = -(-b_br*tr - g_br*ti)/tm
        B_q = (-g_br*tr + b_br*ti)/tm

        vm_prod = vm_fr * vm_to
        sin_θ = sin(θ_diff)
        cos_θ = cos(θ_diff)

        # From side flow derivatives w.r.t. va_fr
        ∂p_fr_∂va_fr = z_l * (-A_p * sin_θ + B_p * cos_θ) * vm_prod
        ∂q_fr_∂va_fr = z_l * (-A_q * sin_θ + B_q * cos_θ) * vm_prod

        # From side flow derivatives w.r.t. va_to
        ∂p_fr_∂va_to = -∂p_fr_∂va_fr  # opposite sign
        ∂q_fr_∂va_to = -∂q_fr_∂va_fr

        # To side flow derivatives (angle diff is va_to - va_fr = -θ_diff)
        A_p_to = (-g_br*tr - b_br*ti)/tm
        B_p_to = (-b_br*tr + g_br*ti)/tm
        A_q_to = -(-b_br*tr + g_br*ti)/tm
        B_q_to = (-g_br*tr - b_br*ti)/tm

        # ∂p_to/∂va_to (w.r.t. -θ_diff)
        ∂p_to_∂va_to = z_l * (-A_p_to * sin(-θ_diff) + B_p_to * cos(-θ_diff)) * vm_prod
        ∂q_to_∂va_to = z_l * (-A_q_to * sin(-θ_diff) + B_q_to * cos(-θ_diff)) * vm_prod
        ∂p_to_∂va_fr = -∂p_to_∂va_to
        ∂q_to_∂va_fr = -∂q_to_∂va_to

        # Power balance contributions: -ν · ∂(flow)/∂va
        # The power balance is: pg - pd - Σ flows = 0
        # Lagrangian: ν_p_bal' * (pg - pd - Σ flows)
        # ∂L/∂va_i = -Σ ν_p_bal[bus] * ∂flow/∂va_i

        # From side flow affects f_bus balance
        K_va[f_bus] += -ν_p_bal[f_bus] * ∂p_fr_∂va_fr
        K_va[f_bus] += -ν_q_bal[f_bus] * ∂q_fr_∂va_fr
        K_va[t_bus] += -ν_p_bal[f_bus] * ∂p_fr_∂va_to
        K_va[t_bus] += -ν_q_bal[f_bus] * ∂q_fr_∂va_to

        # To side flow affects t_bus balance
        K_va[t_bus] += -ν_p_bal[t_bus] * ∂p_to_∂va_to
        K_va[t_bus] += -ν_q_bal[t_bus] * ∂q_to_∂va_to
        K_va[f_bus] += -ν_p_bal[t_bus] * ∂p_to_∂va_fr
        K_va[f_bus] += -ν_q_bal[t_bus] * ∂q_to_∂va_fr

        # Thermal limit contributions: λ * ∂(p² + q² - rate_a²)/∂va
        # = λ * 2 * (p * ∂p/∂va + q * ∂q/∂va)
        K_va[f_bus] += λ_thermal_fr[l] * 2 * (p_fr[l] * ∂p_fr_∂va_fr + q_fr[l] * ∂q_fr_∂va_fr)
        K_va[t_bus] += λ_thermal_fr[l] * 2 * (p_fr[l] * ∂p_fr_∂va_to + q_fr[l] * ∂q_fr_∂va_to)
        K_va[t_bus] += λ_thermal_to[l] * 2 * (p_to[l] * ∂p_to_∂va_to + q_to[l] * ∂q_to_∂va_to)
        K_va[f_bus] += λ_thermal_to[l] * 2 * (p_to[l] * ∂p_to_∂va_fr + q_to[l] * ∂q_to_∂va_fr)
    end

    return K_va
end

"""
Stationarity w.r.t. voltage magnitudes vm.
"""
function _stationarity_vm(va, vm, pg, qg, ν_p_bal, ν_q_bal,
                          λ_thermal_fr, λ_thermal_to, μ_vm_lb, μ_vm_ub,
                          p_fr, q_fr, p_to, q_to,
                          net::ACNetwork, ref, prob::ACOPFProblem, z_switch)
    n, m = net.n, net.m
    T = promote_type(eltype(va), eltype(vm), eltype(z_switch), eltype(p_fr))
    K_vm = zeros(T, n)

    # Voltage bound dual contributions
    for i in 1:n
        K_vm[i] += μ_vm_ub[i] - μ_vm_lb[i]
    end

    # Shunt contributions to power balance
    for i in 1:n
        bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]
        gs = sum(shunt["gs"] for shunt in bus_shunts; init=zero(T))
        bs = sum(shunt["bs"] for shunt in bus_shunts; init=zero(T))

        # Power balance: pg - pd - gs*vm² - Σ p_flow = 0 (for P)
        # ∂/∂vm_i of (-gs*vm_i²) term gives -2*gs*vm_i
        # Contribution to Lagrangian gradient: -ν_p_bal * (-2*gs*vm) = 2*gs*vm*ν_p_bal
        K_vm[i] += -ν_p_bal[i] * 2 * gs * vm[i]

        # Q balance: qg - qd + bs*vm² - Σ q_flow = 0
        # ∂/∂vm_i of (bs*vm_i²) gives 2*bs*vm_i
        K_vm[i] += -ν_q_bal[i] * 2 * (-bs) * vm[i]
    end

    # Branch flow contributions
    for (l, branch) in ref[:branch]
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        rate_a = branch["rate_a"]

        g_br, b_br = PM.calc_branch_y(branch)
        tr, ti = PM.calc_branch_t(branch)
        g_fr_shunt = branch["g_fr"]
        b_fr_shunt = branch["b_fr"]
        g_to_shunt = branch["g_to"]
        b_to_shunt = branch["b_to"]
        tm = branch["tap"]^2

        z_l = z_switch[l]

        vm_fr = vm[f_bus]
        vm_to = vm[t_bus]
        va_fr = va[f_bus]
        va_to = va[t_bus]
        θ_diff = va_fr - va_to

        sin_θ = sin(θ_diff)
        cos_θ = cos(θ_diff)

        # From side flow derivatives w.r.t. vm_fr
        ∂p_fr_∂vm_fr = z_l * (2 * (g_br + g_fr_shunt)/tm * vm_fr +
                       (-g_br*tr + b_br*ti)/tm * vm_to * cos_θ +
                       (-b_br*tr - g_br*ti)/tm * vm_to * sin_θ)

        ∂q_fr_∂vm_fr = z_l * (-2 * (b_br + b_fr_shunt)/tm * vm_fr -
                       (-b_br*tr - g_br*ti)/tm * vm_to * cos_θ +
                       (-g_br*tr + b_br*ti)/tm * vm_to * sin_θ)

        # From side flow derivatives w.r.t. vm_to
        ∂p_fr_∂vm_to = z_l * ((-g_br*tr + b_br*ti)/tm * vm_fr * cos_θ +
                       (-b_br*tr - g_br*ti)/tm * vm_fr * sin_θ)

        ∂q_fr_∂vm_to = z_l * (-(-b_br*tr - g_br*ti)/tm * vm_fr * cos_θ +
                       (-g_br*tr + b_br*ti)/tm * vm_fr * sin_θ)

        # To side flow derivatives w.r.t. vm_to
        ∂p_to_∂vm_to = z_l * (2 * (g_br + g_to_shunt) * vm_to +
                       (-g_br*tr - b_br*ti)/tm * vm_fr * cos(-θ_diff) +
                       (-b_br*tr + g_br*ti)/tm * vm_fr * sin(-θ_diff))

        ∂q_to_∂vm_to = z_l * (-2 * (b_br + b_to_shunt) * vm_to -
                       (-b_br*tr + g_br*ti)/tm * vm_fr * cos(θ_diff) +
                       (-g_br*tr - b_br*ti)/tm * vm_fr * sin(-θ_diff))

        # To side flow derivatives w.r.t. vm_fr
        ∂p_to_∂vm_fr = z_l * ((-g_br*tr - b_br*ti)/tm * vm_to * cos(-θ_diff) +
                       (-b_br*tr + g_br*ti)/tm * vm_to * sin(-θ_diff))

        ∂q_to_∂vm_fr = z_l * (-(-b_br*tr + g_br*ti)/tm * vm_to * cos(θ_diff) +
                       (-g_br*tr - b_br*ti)/tm * vm_to * sin(-θ_diff))

        # Power balance contributions
        K_vm[f_bus] += -ν_p_bal[f_bus] * ∂p_fr_∂vm_fr
        K_vm[f_bus] += -ν_q_bal[f_bus] * ∂q_fr_∂vm_fr
        K_vm[t_bus] += -ν_p_bal[f_bus] * ∂p_fr_∂vm_to
        K_vm[t_bus] += -ν_q_bal[f_bus] * ∂q_fr_∂vm_to

        K_vm[t_bus] += -ν_p_bal[t_bus] * ∂p_to_∂vm_to
        K_vm[t_bus] += -ν_q_bal[t_bus] * ∂q_to_∂vm_to
        K_vm[f_bus] += -ν_p_bal[t_bus] * ∂p_to_∂vm_fr
        K_vm[f_bus] += -ν_q_bal[t_bus] * ∂q_to_∂vm_fr

        # Thermal limit contributions
        K_vm[f_bus] += λ_thermal_fr[l] * 2 * (p_fr[l] * ∂p_fr_∂vm_fr + q_fr[l] * ∂q_fr_∂vm_fr)
        K_vm[t_bus] += λ_thermal_fr[l] * 2 * (p_fr[l] * ∂p_fr_∂vm_to + q_fr[l] * ∂q_fr_∂vm_to)
        K_vm[t_bus] += λ_thermal_to[l] * 2 * (p_to[l] * ∂p_to_∂vm_to + q_to[l] * ∂q_to_∂vm_to)
        K_vm[f_bus] += λ_thermal_to[l] * 2 * (p_to[l] * ∂p_to_∂vm_fr + q_to[l] * ∂q_to_∂vm_fr)
    end

    return K_vm
end

"""
Stationarity w.r.t. active generation pg.
"""
function _stationarity_pg(pg, ν_p_bal, ρ_pg_lb, ρ_pg_ub, ref, prob::ACOPFProblem)
    k = prob.n_gen
    T = eltype(pg)
    K_pg = zeros(T, k)

    for i in 1:k
        gen = ref[:gen][i]
        bus_idx = gen["gen_bus"]

        # ∂cost/∂pg_i + (−ν_p_bal at gen bus) + ρ_pg_ub - ρ_pg_lb = 0
        cost_grad = 2 * gen["cost"][1] * pg[i] + gen["cost"][2]
        K_pg[i] = cost_grad - ν_p_bal[bus_idx] + ρ_pg_ub[i] - ρ_pg_lb[i]
    end

    return K_pg
end

"""
Stationarity w.r.t. reactive generation qg.
"""
function _stationarity_qg(qg, ν_q_bal, ρ_qg_lb, ρ_qg_ub, ref, prob::ACOPFProblem)
    k = prob.n_gen
    T = eltype(qg)
    K_qg = zeros(T, k)

    for i in 1:k
        gen = ref[:gen][i]
        bus_idx = gen["gen_bus"]

        # 0 - ν_q_bal at gen bus + ρ_qg_ub - ρ_qg_lb = 0
        K_qg[i] = -ν_q_bal[bus_idx] + ρ_qg_ub[i] - ρ_qg_lb[i]
    end

    return K_qg
end

# =============================================================================
# KKT Helper Functions - Primal Feasibility
# =============================================================================

"""
Power balance residuals.
"""
function _power_balance_residuals(va, vm, pg, qg, p_fr, q_fr, p_to, q_to, net::ACNetwork, ref, prob::ACOPFProblem)
    n = net.n
    T = promote_type(eltype(va), eltype(vm), eltype(pg), eltype(p_fr))
    K_p_bal = zeros(T, n)
    K_q_bal = zeros(T, n)

    # Sum flows at each bus
    p_flow_sum = zeros(T, n)
    q_flow_sum = zeros(T, n)

    for (l, branch) in ref[:branch]
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        p_flow_sum[f_bus] += p_fr[l]
        p_flow_sum[t_bus] += p_to[l]
        q_flow_sum[f_bus] += q_fr[l]
        q_flow_sum[t_bus] += q_to[l]
    end

    # Sum generation at each bus (use same T since flows may have wider type)
    pg_sum = zeros(T, n)
    qg_sum = zeros(T, n)
    for i in 1:prob.n_gen
        bus_idx = ref[:gen][i]["gen_bus"]
        pg_sum[bus_idx] += pg[i]
        qg_sum[bus_idx] += qg[i]
    end

    for i in 1:n
        bus = ref[:bus][i]
        bus_loads = [ref[:load][l] for l in ref[:bus_loads][i]]
        bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]

        pd = sum(load["pd"] for load in bus_loads; init=zero(T))
        qd = sum(load["qd"] for load in bus_loads; init=zero(T))
        gs = sum(shunt["gs"] for shunt in bus_shunts; init=zero(T))
        bs = sum(shunt["bs"] for shunt in bus_shunts; init=zero(T))

        # P balance: Σ p_flow + gs*vm² = pg - pd
        K_p_bal[i] = p_flow_sum[i] + gs * vm[i]^2 - pg_sum[i] + pd

        # Q balance: Σ q_flow - bs*vm² = qg - qd
        K_q_bal[i] = q_flow_sum[i] - bs * vm[i]^2 - qg_sum[i] + qd
    end

    return K_p_bal, K_q_bal
end

# =============================================================================
# Switching Sensitivity
# =============================================================================

"""
    calc_ac_kkt_jacobian_switching(prob::ACOPFProblem, sol::ACOPFSolution)

Compute the Jacobian of KKT conditions with respect to switching variables ∂K/∂s.

Uses ForwardDiff for automatic differentiation, avoiding finite differences.

# Returns
Matrix of size (kkt_dims × m).
"""
function calc_ac_kkt_jacobian_switching(prob::ACOPFProblem, sol::ACOPFSolution)
    # Flatten the primal-dual solution
    z0 = ac_flatten_variables(sol, prob)

    # Get the current switching state
    z_switch = prob.network.z

    # Use ForwardDiff to compute ∂K/∂s
    # The KKT function takes z_switch as a parameter
    J_s = ForwardDiff.jacobian(s -> ac_kkt(z0, prob, s), z_switch)

    return J_s
end

"""
    calc_sensitivity_switching(prob::ACOPFProblem) → ACOPFSwitchingSens

Compute sensitivities of AC OPF solution with respect to switching variables.

Uses the implicit function theorem on KKT conditions:
∂x/∂s = -(∂K/∂x)⁻¹ · (∂K/∂s)

where x is the flattened primal-dual variable vector.

# Returns
`ACOPFSwitchingSens` containing Jacobians of solution variables w.r.t. switching:
- `dvm_dz`: ∂vm/∂z (n × m) - voltage magnitudes w.r.t. switching
- `dva_dz`: ∂va/∂z (n × m) - voltage angles w.r.t. switching
- `dpg_dz`: ∂pg/∂z (k × m) - active generation w.r.t. switching
- `dqg_dz`: ∂qg/∂z (k × m) - reactive generation w.r.t. switching
"""
function calc_sensitivity_switching(prob::ACOPFProblem)
    n, m, k = prob.network.n, prob.network.m, prob.n_gen

    # Solve the problem once
    sol = solve!(prob)

    # Compute Jacobians (pass solution to avoid re-solving)
    J_x = calc_ac_kkt_jacobian(prob; sol=sol)  # ∂K/∂x
    J_s = calc_ac_kkt_jacobian_switching(prob, sol)  # ∂K/∂s

    # Implicit function theorem: ∂x/∂s = -(∂K/∂x)⁻¹ · (∂K/∂s)
    # Use backslash for efficiency; add small regularization if needed for stability
    dx_ds = try
        -J_x \ J_s
    catch e
        if e isa LinearAlgebra.SingularException
            # Fall back to regularized solve
            ε = 1e-10
            -((J_x + ε * I) \ J_s)
        else
            rethrow(e)
        end
    end

    # Extract sensitivities using centralized indices
    idx = ac_kkt_indices(n, m, k)

    dva_ds = dx_ds[idx.va, :]
    dvm_ds = dx_ds[idx.vm, :]
    dpg_ds = dx_ds[idx.pg, :]
    dqg_ds = dx_ds[idx.qg, :]

    return ACOPFSwitchingSens(dvm_ds, dva_ds, dpg_ds, dqg_ds)
end
