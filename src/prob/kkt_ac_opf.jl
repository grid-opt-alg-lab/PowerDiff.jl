# =============================================================================
# KKT System for AC OPF
# =============================================================================
#
# Implements KKT conditions for implicit differentiation of AC OPF solutions.
# Design mirrors kkt_dc_opf.jl for consistency.
#
# Uses a reduced-space formulation where branch flows p_fr, q_fr, p_to, q_to
# are functions of voltage state (va, vm), not separate primal variables.
#
# Stationarity conditions are computed via ForwardDiff on the reduced-space
# Lagrangian, eliminating manual derivative errors.

# =============================================================================
# Helpers
# =============================================================================

"""Return sorted reference bus indices for the problem."""
_ref_bus_indices(prob::ACOPFProblem) = sort(collect(keys(prob.ref[:ref_buses])))

# =============================================================================
# Dimension Calculations
# =============================================================================

"""
    ac_kkt_dims(prob::ACOPFProblem)

Compute the dimension of the flattened KKT variable vector for AC OPF.

The KKT system includes:
- Primal: va (n), vm (n), pg (k), qg (k)
- Dual (equality): ν_p_bal (n), ν_q_bal (n), ν_ref_bus (n_ref)
- Dual (inequality): λ_thermal_fr (m), λ_thermal_to (m),
                     λ_angle_lb (m), λ_angle_ub (m),
                     μ_vm_lb (n), μ_vm_ub (n),
                     ρ_pg_lb (k), ρ_pg_ub (k), ρ_qg_lb (k), ρ_qg_ub (k),
                     σ_p_fr_lb (m), σ_p_fr_ub (m), σ_q_fr_lb (m), σ_q_fr_ub (m),
                     σ_p_to_lb (m), σ_p_to_ub (m), σ_q_to_lb (m), σ_q_to_ub (m)

Total: 6n + 12m + 6k + n_ref
"""
function ac_kkt_dims(prob::ACOPFProblem)
    n, m, k = prob.network.n, prob.network.m, prob.n_gen
    n_ref = length(prob.ref[:ref_buses])
    return 6n + 12m + 6k + n_ref
end

"""
    ac_kkt_indices(n, m, k, n_ref) → NamedTuple

Compute all KKT variable indices from problem dimensions.
Single source of truth for index calculations.

# Variable ordering
[va(n), vm(n), pg(k), qg(k),
 ν_p_bal(n), ν_q_bal(n), ν_ref_bus(n_ref),
 λ_thermal_fr(m), λ_thermal_to(m), λ_angle_lb(m), λ_angle_ub(m),
 μ_vm_lb(n), μ_vm_ub(n),
 ρ_pg_lb(k), ρ_pg_ub(k), ρ_qg_lb(k), ρ_qg_ub(k),
 σ_p_fr_lb(m), σ_p_fr_ub(m), σ_q_fr_lb(m), σ_q_fr_ub(m),
 σ_p_to_lb(m), σ_p_to_ub(m), σ_q_to_lb(m), σ_q_to_ub(m)]

# Returns
NamedTuple with index ranges for each variable block.
"""
function ac_kkt_indices(n::Int, m::Int, k::Int, n_ref::Int)
    i = 0
    # Primal
    idx_va = (i+1):(i+n); i += n
    idx_vm = (i+1):(i+n); i += n
    idx_pg = (i+1):(i+k); i += k
    idx_qg = (i+1):(i+k); i += k

    # Dual (equality)
    idx_ν_p_bal = (i+1):(i+n); i += n
    idx_ν_q_bal = (i+1):(i+n); i += n
    idx_ν_ref_bus = (i+1):(i+n_ref); i += n_ref

    # Dual (inequality) — thermal, angle diff
    idx_λ_thermal_fr = (i+1):(i+m); i += m
    idx_λ_thermal_to = (i+1):(i+m); i += m
    idx_λ_angle_lb = (i+1):(i+m); i += m
    idx_λ_angle_ub = (i+1):(i+m); i += m

    # Dual (inequality) — voltage bounds
    idx_μ_vm_lb = (i+1):(i+n); i += n
    idx_μ_vm_ub = (i+1):(i+n); i += n

    # Dual (inequality) — generation bounds
    idx_ρ_pg_lb = (i+1):(i+k); i += k
    idx_ρ_pg_ub = (i+1):(i+k); i += k
    idx_ρ_qg_lb = (i+1):(i+k); i += k
    idx_ρ_qg_ub = (i+1):(i+k); i += k

    # Dual (inequality) — flow variable bounds (reduced-space)
    idx_σ_p_fr_lb = (i+1):(i+m); i += m
    idx_σ_p_fr_ub = (i+1):(i+m); i += m
    idx_σ_q_fr_lb = (i+1):(i+m); i += m
    idx_σ_q_fr_ub = (i+1):(i+m); i += m
    idx_σ_p_to_lb = (i+1):(i+m); i += m
    idx_σ_p_to_ub = (i+1):(i+m); i += m
    idx_σ_q_to_lb = (i+1):(i+m); i += m
    idx_σ_q_to_ub = (i+1):(i+m); i += m

    return (
        va = idx_va, vm = idx_vm, pg = idx_pg, qg = idx_qg,
        ν_p_bal = idx_ν_p_bal, ν_q_bal = idx_ν_q_bal, ν_ref_bus = idx_ν_ref_bus,
        λ_thermal_fr = idx_λ_thermal_fr, λ_thermal_to = idx_λ_thermal_to,
        λ_angle_lb = idx_λ_angle_lb, λ_angle_ub = idx_λ_angle_ub,
        μ_vm_lb = idx_μ_vm_lb, μ_vm_ub = idx_μ_vm_ub,
        ρ_pg_lb = idx_ρ_pg_lb, ρ_pg_ub = idx_ρ_pg_ub,
        ρ_qg_lb = idx_ρ_qg_lb, ρ_qg_ub = idx_ρ_qg_ub,
        σ_p_fr_lb = idx_σ_p_fr_lb, σ_p_fr_ub = idx_σ_p_fr_ub,
        σ_q_fr_lb = idx_σ_q_fr_lb, σ_q_fr_ub = idx_σ_q_fr_ub,
        σ_p_to_lb = idx_σ_p_to_lb, σ_p_to_ub = idx_σ_p_to_ub,
        σ_q_to_lb = idx_σ_q_to_lb, σ_q_to_ub = idx_σ_q_to_ub
    )
end

function ac_kkt_indices(prob::ACOPFProblem)
    n_ref = length(prob.ref[:ref_buses])
    ac_kkt_indices(prob.network.n, prob.network.m, prob.n_gen, n_ref)
end

# =============================================================================
# Variable Flattening/Unflattening
# =============================================================================

"""
    ac_flatten_variables(sol::ACOPFSolution, prob::ACOPFProblem)

Flatten solution primal and dual variables into a single vector for KKT evaluation.
Ordering matches `ac_kkt_indices`.
"""
function ac_flatten_variables(sol::ACOPFSolution, prob::ACOPFProblem)
    return vcat(
        sol.va, sol.vm, sol.pg, sol.qg,
        sol.ν_p_bal, sol.ν_q_bal, sol.ν_ref_bus,
        sol.λ_thermal_fr, sol.λ_thermal_to,
        sol.λ_angle_lb, sol.λ_angle_ub,
        sol.μ_vm_lb, sol.μ_vm_ub,
        sol.ρ_pg_lb, sol.ρ_pg_ub, sol.ρ_qg_lb, sol.ρ_qg_ub,
        sol.σ_p_fr_lb, sol.σ_p_fr_ub, sol.σ_q_fr_lb, sol.σ_q_fr_ub,
        sol.σ_p_to_lb, sol.σ_p_to_ub, sol.σ_q_to_lb, sol.σ_q_to_ub
    )
end

"""
    ac_unflatten_variables(z::AbstractVector, prob::ACOPFProblem)

Unflatten KKT variable vector into named components.

# Returns
NamedTuple with fields for all primal and dual variables.
"""
function ac_unflatten_variables(z::AbstractVector, prob::ACOPFProblem)
    idx = ac_kkt_indices(prob)

    return (
        va = z[idx.va], vm = z[idx.vm], pg = z[idx.pg], qg = z[idx.qg],
        ν_p_bal = z[idx.ν_p_bal], ν_q_bal = z[idx.ν_q_bal], ν_ref_bus = z[idx.ν_ref_bus],
        λ_thermal_fr = z[idx.λ_thermal_fr], λ_thermal_to = z[idx.λ_thermal_to],
        λ_angle_lb = z[idx.λ_angle_lb], λ_angle_ub = z[idx.λ_angle_ub],
        μ_vm_lb = z[idx.μ_vm_lb], μ_vm_ub = z[idx.μ_vm_ub],
        ρ_pg_lb = z[idx.ρ_pg_lb], ρ_pg_ub = z[idx.ρ_pg_ub],
        ρ_qg_lb = z[idx.ρ_qg_lb], ρ_qg_ub = z[idx.ρ_qg_ub],
        σ_p_fr_lb = z[idx.σ_p_fr_lb], σ_p_fr_ub = z[idx.σ_p_fr_ub],
        σ_q_fr_lb = z[idx.σ_q_fr_lb], σ_q_fr_ub = z[idx.σ_q_fr_ub],
        σ_p_to_lb = z[idx.σ_p_to_lb], σ_p_to_ub = z[idx.σ_p_to_ub],
        σ_q_to_lb = z[idx.σ_q_to_lb], σ_q_to_ub = z[idx.σ_q_to_ub]
    )
end

# =============================================================================
# KKT Jacobian via ForwardDiff
# =============================================================================

"""
    calc_ac_kkt_jacobian(prob::ACOPFProblem; sol=nothing)

Compute the Jacobian of the KKT operator using ForwardDiff.

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

    z0 = ac_flatten_variables(sol, prob)
    sw = prob.network.sw
    J = ForwardDiff.jacobian(z -> ac_kkt(z, prob, sw), z0)

    return J
end

# =============================================================================
# Branch Flow Calculations
# =============================================================================

"""
Compute all branch flows given voltage state and switching state.
Returns vectors of p_fr, q_fr, p_to, q_to indexed by branch number.

The switching variable z_l multiplies each flow, so z_l=0 means the branch
contributes zero flow (open), z_l=1 means full flow (closed).
"""
function _compute_branch_flows(va, vm, net::ACNetwork, ref, sw)
    m = net.m
    T = promote_type(eltype(va), eltype(vm), eltype(sw))
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

        sw_l = sw[l]

        vm_fr = vm[f_bus]
        vm_to = vm[t_bus]
        va_fr = va[f_bus]
        va_to = va[t_bus]

        # From side
        p_fr[l] = sw_l *((g_br + g_fr_shunt)/tm * vm_fr^2 +
                  (-g_br*tr + b_br*ti)/tm * (vm_fr * vm_to * cos(va_fr - va_to)) +
                  (-b_br*tr - g_br*ti)/tm * (vm_fr * vm_to * sin(va_fr - va_to)))

        q_fr[l] = sw_l * (-(b_br + b_fr_shunt)/tm * vm_fr^2 -
                  (-b_br*tr - g_br*ti)/tm * (vm_fr * vm_to * cos(va_fr - va_to)) +
                  (-g_br*tr + b_br*ti)/tm * (vm_fr * vm_to * sin(va_fr - va_to)))

        # To side
        p_to[l] = sw_l * ((g_br + g_to_shunt) * vm_to^2 +
                  (-g_br*tr - b_br*ti)/tm * (vm_to * vm_fr * cos(va_to - va_fr)) +
                  (-b_br*tr + g_br*ti)/tm * (vm_to * vm_fr * sin(va_to - va_fr)))

        q_to[l] = sw_l * (-(b_br + b_to_shunt) * vm_to^2 -
                  (-b_br*tr + g_br*ti)/tm * (vm_to * vm_fr * cos(va_fr - va_to)) +
                  (-g_br*tr - b_br*ti)/tm * (vm_to * vm_fr * sin(va_to - va_fr)))
    end

    return p_fr, q_fr, p_to, q_to
end

# =============================================================================
# Reduced-Space Lagrangian (for stationarity via ForwardDiff)
# =============================================================================

"""
Compute the reduced-space Lagrangian L(va, vm, pg, qg; duals, z_switch).

In the reduced space, flows are functions of (va, vm), not separate variables.
ForwardDiff.gradient of this function w.r.t. [va; vm; pg; qg] gives the
stationarity conditions, automatically handling all chain-rule terms including
power balance, thermal limits, flow bounds, angle diffs, and shunt terms.

Uses the JuMP/MOI dual sign convention:
    L = f(x) - Σ dual_i * normalized_residual_i
where normalized_residual = constraint_function - set_value.
"""
function _reduced_lagrangian(x_primal, vars, prob::ACOPFProblem, sw)
    net = prob.network
    ref = prob.ref
    n, m, k = net.n, net.m, prob.n_gen

    va = x_primal[1:n]
    vm = x_primal[n+1:2n]
    pg = x_primal[2n+1:2n+k]
    qg = x_primal[2n+k+1:2n+2k]

    # Compute reduced-space flows
    p_fr, q_fr, p_to, q_to = _compute_branch_flows(va, vm, net, ref, sw)

    # Use widest type from computed flows, which captures both inner (x_primal) and
    # outer (sw) ForwardDiff dual types in the nested differentiation.
    T = eltype(p_fr)
    L = zero(T)

    # ----- Objective -----
    for i in 1:k
        gen = ref[:gen][i]
        L += gen["cost"][1] * pg[i]^2 + gen["cost"][2] * pg[i] + gen["cost"][3]
    end

    # ----- Power balance (equality): h = 0 -----
    # h_P[i] = Σ p_flow + gs*vm² - pg_sum + pd
    # h_Q[i] = Σ q_flow - bs*vm² - qg_sum + qd
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

    pg_sum = zeros(T, n)
    qg_sum = zeros(T, n)
    for i in 1:k
        bus_idx = ref[:gen][i]["gen_bus"]
        pg_sum[bus_idx] += pg[i]
        qg_sum[bus_idx] += qg[i]
    end

    for i in 1:n
        bus_loads = [ref[:load][l] for l in ref[:bus_loads][i]]
        bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]

        pd = sum(load["pd"] for load in bus_loads; init=0.0)
        qd = sum(load["qd"] for load in bus_loads; init=0.0)
        gs = sum(shunt["gs"] for shunt in bus_shunts; init=0.0)
        bs = sum(shunt["bs"] for shunt in bus_shunts; init=0.0)

        h_P = p_flow_sum[i] + gs * vm[i]^2 - pg_sum[i] + pd
        h_Q = q_flow_sum[i] - bs * vm[i]^2 - qg_sum[i] + qd

        L -= vars.ν_p_bal[i] * h_P
        L -= vars.ν_q_bal[i] * h_Q
    end

    # ----- Reference bus (equality): va[ref] - 0 = 0 -----
    ref_bus_keys = _ref_bus_indices(prob)
    for (j, ref_bus_idx) in enumerate(ref_bus_keys)
        L -= vars.ν_ref_bus[j] * va[ref_bus_idx]
    end

    # ----- Thermal limits (inequality): p²+q²-r² ≤ 0 -----
    for l in 1:m
        rate_a = ref[:branch][l]["rate_a"]
        L -= vars.λ_thermal_fr[l] * (p_fr[l]^2 + q_fr[l]^2 - rate_a^2)
        L -= vars.λ_thermal_to[l] * (p_to[l]^2 + q_to[l]^2 - rate_a^2)
    end

    # ----- Angle difference limits (inequality) -----
    # va_fr - va_to >= angmin  →  normalized: (va_fr - va_to) - angmin
    # va_fr - va_to <= angmax  →  normalized: (va_fr - va_to) - angmax
    for l in 1:m
        branch = ref[:branch][l]
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        L -= vars.λ_angle_lb[l] * (va[f_bus] - va[t_bus] - branch["angmin"])
        L -= vars.λ_angle_ub[l] * (va[f_bus] - va[t_bus] - branch["angmax"])
    end

    # ----- Voltage bounds (inequality) -----
    # vm >= vmin  →  normalized: vm - vmin
    # vm <= vmax  →  normalized: vm - vmax
    for i in 1:n
        L -= vars.μ_vm_lb[i] * (vm[i] - ref[:bus][i]["vmin"])
        L -= vars.μ_vm_ub[i] * (vm[i] - ref[:bus][i]["vmax"])
    end

    # ----- Generation bounds (inequality) -----
    # pg >= pmin  →  normalized: pg - pmin
    # pg <= pmax  →  normalized: pg - pmax
    for i in 1:k
        gen = ref[:gen][i]
        L -= vars.ρ_pg_lb[i] * (pg[i] - gen["pmin"])
        L -= vars.ρ_pg_ub[i] * (pg[i] - gen["pmax"])
        L -= vars.ρ_qg_lb[i] * (qg[i] - gen["qmin"])
        L -= vars.ρ_qg_ub[i] * (qg[i] - gen["qmax"])
    end

    # ----- Flow variable bounds (inequality, reduced-space) -----
    # p_fr >= -rate_a  →  normalized: p_fr - (-rate_a) = p_fr + rate_a
    # p_fr <= rate_a   →  normalized: p_fr - rate_a
    for l in 1:m
        rate_a = ref[:branch][l]["rate_a"]
        L -= vars.σ_p_fr_lb[l] * (p_fr[l] + rate_a)
        L -= vars.σ_p_fr_ub[l] * (p_fr[l] - rate_a)
        L -= vars.σ_q_fr_lb[l] * (q_fr[l] + rate_a)
        L -= vars.σ_q_fr_ub[l] * (q_fr[l] - rate_a)
        L -= vars.σ_p_to_lb[l] * (p_to[l] + rate_a)
        L -= vars.σ_p_to_ub[l] * (p_to[l] - rate_a)
        L -= vars.σ_q_to_lb[l] * (q_to[l] + rate_a)
        L -= vars.σ_q_to_ub[l] * (q_to[l] - rate_a)
    end

    return L
end

# =============================================================================
# Power Balance Residuals (primal feasibility)
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

    # Sum generation at each bus
    pg_sum = zeros(T, n)
    qg_sum = zeros(T, n)
    for i in 1:prob.n_gen
        bus_idx = ref[:gen][i]["gen_bus"]
        pg_sum[bus_idx] += pg[i]
        qg_sum[bus_idx] += qg[i]
    end

    for i in 1:n
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
# KKT Operator
# =============================================================================

"""
    ac_kkt(z::AbstractVector, prob::ACOPFProblem, sw::AbstractVector)

Evaluate the KKT conditions for AC OPF at the given variable vector.

The switching state sw is passed as a separate parameter to enable
differentiation with respect to switching using ForwardDiff.

Returns a vector of KKT residuals (should be zero at optimum).

# KKT Conditions (reduced-space formulation)
1. Stationarity w.r.t. va, vm, pg, qg (via ForwardDiff on Lagrangian)
2. Primal feasibility: power balance, reference bus
3. Complementary slackness for all inequality constraints
"""
function ac_kkt(z::AbstractVector, prob::ACOPFProblem, sw::AbstractVector)
    vars = ac_unflatten_variables(z, prob)
    net = prob.network
    ref = prob.ref
    n, m, k = net.n, net.m, prob.n_gen

    va, vm = vars.va, vars.vm
    pg, qg = vars.pg, vars.qg

    T = promote_type(eltype(z), eltype(sw))

    # Compute branch flows as functions of voltages
    p_fr, q_fr, p_to, q_to = _compute_branch_flows(va, vm, net, ref, sw)

    # Build KKT residual vector
    K = T[]

    # =========================================================================
    # 1. Stationarity conditions via ForwardDiff on the Lagrangian
    # =========================================================================
    x_primal = vcat(va, vm, pg, qg)
    grad = ForwardDiff.gradient(
        x -> _reduced_lagrangian(x, vars, prob, sw),
        x_primal
    )
    # grad = [∂L/∂va; ∂L/∂vm; ∂L/∂pg; ∂L/∂qg]
    append!(K, grad)

    # =========================================================================
    # 2. Primal feasibility
    # =========================================================================

    # Power balance
    K_p_bal, K_q_bal = _power_balance_residuals(va, vm, pg, qg, p_fr, q_fr, p_to, q_to, net, ref, prob)
    append!(K, K_p_bal)
    append!(K, K_q_bal)

    # Reference bus: va[ref_bus] == 0
    ref_bus_keys = _ref_bus_indices(prob)
    for ref_bus_idx in ref_bus_keys
        push!(K, va[ref_bus_idx])
    end

    # =========================================================================
    # 3. Complementary slackness conditions
    # =========================================================================

    # Thermal limits
    for l in 1:m
        rate_a = ref[:branch][l]["rate_a"]
        push!(K, vars.λ_thermal_fr[l] * (p_fr[l]^2 + q_fr[l]^2 - rate_a^2))
    end
    for l in 1:m
        rate_a = ref[:branch][l]["rate_a"]
        push!(K, vars.λ_thermal_to[l] * (p_to[l]^2 + q_to[l]^2 - rate_a^2))
    end

    # Angle difference limits
    for l in 1:m
        branch = ref[:branch][l]
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        push!(K, vars.λ_angle_lb[l] * (va[f_bus] - va[t_bus] - branch["angmin"]))
    end
    for l in 1:m
        branch = ref[:branch][l]
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        push!(K, vars.λ_angle_ub[l] * (branch["angmax"] - va[f_bus] + va[t_bus]))
    end

    # Voltage bounds
    for i in 1:n
        push!(K, vars.μ_vm_lb[i] * (vm[i] - ref[:bus][i]["vmin"]))
    end
    for i in 1:n
        push!(K, vars.μ_vm_ub[i] * (ref[:bus][i]["vmax"] - vm[i]))
    end

    # Generation bounds
    for i in 1:k
        push!(K, vars.ρ_pg_lb[i] * (pg[i] - ref[:gen][i]["pmin"]))
    end
    for i in 1:k
        push!(K, vars.ρ_pg_ub[i] * (ref[:gen][i]["pmax"] - pg[i]))
    end
    for i in 1:k
        push!(K, vars.ρ_qg_lb[i] * (qg[i] - ref[:gen][i]["qmin"]))
    end
    for i in 1:k
        push!(K, vars.ρ_qg_ub[i] * (ref[:gen][i]["qmax"] - qg[i]))
    end

    # Flow variable bounds (reduced-space)
    for l in 1:m
        rate_a = ref[:branch][l]["rate_a"]
        push!(K, vars.σ_p_fr_lb[l] * (p_fr[l] + rate_a))
    end
    for l in 1:m
        rate_a = ref[:branch][l]["rate_a"]
        push!(K, vars.σ_p_fr_ub[l] * (rate_a - p_fr[l]))
    end
    for l in 1:m
        rate_a = ref[:branch][l]["rate_a"]
        push!(K, vars.σ_q_fr_lb[l] * (q_fr[l] + rate_a))
    end
    for l in 1:m
        rate_a = ref[:branch][l]["rate_a"]
        push!(K, vars.σ_q_fr_ub[l] * (rate_a - q_fr[l]))
    end
    for l in 1:m
        rate_a = ref[:branch][l]["rate_a"]
        push!(K, vars.σ_p_to_lb[l] * (p_to[l] + rate_a))
    end
    for l in 1:m
        rate_a = ref[:branch][l]["rate_a"]
        push!(K, vars.σ_p_to_ub[l] * (rate_a - p_to[l]))
    end
    for l in 1:m
        rate_a = ref[:branch][l]["rate_a"]
        push!(K, vars.σ_q_to_lb[l] * (q_to[l] + rate_a))
    end
    for l in 1:m
        rate_a = ref[:branch][l]["rate_a"]
        push!(K, vars.σ_q_to_ub[l] * (rate_a - q_to[l]))
    end

    @assert length(K) == ac_kkt_dims(prob) "KKT vector length mismatch: got $(length(K)), expected $(ac_kkt_dims(prob))"

    return K
end

# Convenience method using prob's switching state
ac_kkt(z::AbstractVector, prob::ACOPFProblem) = ac_kkt(z, prob, prob.network.sw)

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
    z0 = ac_flatten_variables(sol, prob)
    sw = prob.network.sw
    J_s = ForwardDiff.jacobian(s -> ac_kkt(z0, prob, s), sw)
    return J_s
end

# =============================================================================
# Cached Solution and KKT Derivative Access
# =============================================================================

"""
    _ensure_ac_solved!(prob::ACOPFProblem) → ACOPFSolution

Ensure the AC OPF problem is solved and return the cached solution.
If not yet solved, calls solve!(prob) and caches the result.
"""
function _ensure_ac_solved!(prob::ACOPFProblem)::ACOPFSolution
    if isnothing(prob.cache.solution)
        prob.cache.solution = solve!(prob)
    end
    return prob.cache.solution
end

"""
    _get_ac_dx_ds!(prob::ACOPFProblem) → Matrix{Float64}

Get or compute the full KKT derivative matrix w.r.t. switching.
Uses cached value if available, otherwise computes and caches.

Uses the implicit function theorem on KKT conditions:
∂x/∂s = -(∂K/∂x)⁻¹ · (∂K/∂s)
"""
function _get_ac_dx_ds!(prob::ACOPFProblem)::Matrix{Float64}
    if isnothing(prob.cache.dx_ds)
        sol = _ensure_ac_solved!(prob)

        J_x = calc_ac_kkt_jacobian(prob; sol=sol)  # ∂K/∂x
        J_s = calc_ac_kkt_jacobian_switching(prob, sol)  # ∂K/∂s

        # Implicit function theorem: ∂x/∂s = -(∂K/∂x)⁻¹ · (∂K/∂s)
        prob.cache.dx_ds = try
            -J_x \ J_s
        catch e
            if e isa LinearAlgebra.SingularException
                ε = 1e-10
                -((J_x + ε * I) \ J_s)
            else
                rethrow(e)
            end
        end
    end
    return prob.cache.dx_ds
end

"""
    calc_sensitivity_switching(prob::ACOPFProblem) → NamedTuple

Compute sensitivities of AC OPF solution with respect to switching variables.

Uses cached KKT derivatives for efficiency — multiple calls with different
operands share the same expensive Jacobian computation.

# Returns
NamedTuple containing Jacobians of solution variables w.r.t. switching:
- `dvm_dsw`: ∂vm/∂sw (n × m) - voltage magnitudes w.r.t. switching
- `dva_dsw`: ∂va/∂sw (n × m) - voltage angles w.r.t. switching
- `dpg_dsw`: ∂pg/∂sw (k × m) - active generation w.r.t. switching
- `dqg_dsw`: ∂qg/∂sw (k × m) - reactive generation w.r.t. switching
"""
function calc_sensitivity_switching(prob::ACOPFProblem)
    dx_ds = _get_ac_dx_ds!(prob)
    idx = ac_kkt_indices(prob)

    return (
        dvm_dsw = dx_ds[idx.vm, :],
        dva_dsw = dx_ds[idx.va, :],
        dpg_dsw = dx_ds[idx.pg, :],
        dqg_dsw = dx_ds[idx.qg, :],
    )
end
