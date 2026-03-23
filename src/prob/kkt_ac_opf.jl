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
    kkt_dims(prob::ACOPFProblem)

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
function kkt_dims(prob::ACOPFProblem)
    n, m, k = prob.network.n, prob.network.m, prob.n_gen
    n_ref = length(prob.ref[:ref_buses])
    return 6n + 12m + 6k + n_ref
end

"""
    kkt_indices(n, m, k, n_ref) → NamedTuple

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
function kkt_indices(n::Int, m::Int, k::Int, n_ref::Int)
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
        nu_p_bal = idx_ν_p_bal, nu_q_bal = idx_ν_q_bal, nu_ref_bus = idx_ν_ref_bus,
        lam_thermal_fr = idx_λ_thermal_fr, lam_thermal_to = idx_λ_thermal_to,
        lam_angle_lb = idx_λ_angle_lb, lam_angle_ub = idx_λ_angle_ub,
        mu_vm_lb = idx_μ_vm_lb, mu_vm_ub = idx_μ_vm_ub,
        rho_pg_lb = idx_ρ_pg_lb, rho_pg_ub = idx_ρ_pg_ub,
        rho_qg_lb = idx_ρ_qg_lb, rho_qg_ub = idx_ρ_qg_ub,
        sig_p_fr_lb = idx_σ_p_fr_lb, sig_p_fr_ub = idx_σ_p_fr_ub,
        sig_q_fr_lb = idx_σ_q_fr_lb, sig_q_fr_ub = idx_σ_q_fr_ub,
        sig_p_to_lb = idx_σ_p_to_lb, sig_p_to_ub = idx_σ_p_to_ub,
        sig_q_to_lb = idx_σ_q_to_lb, sig_q_to_ub = idx_σ_q_to_ub
    )
end

function kkt_indices(prob::ACOPFProblem)
    n_ref = length(prob.ref[:ref_buses])
    kkt_indices(prob.network.n, prob.network.m, prob.n_gen, n_ref)
end

# =============================================================================
# Variable Flattening/Unflattening
# =============================================================================

"""
    flatten_variables(sol::ACOPFSolution, prob::ACOPFProblem)

Flatten solution primal and dual variables into a single vector for KKT evaluation.
Ordering matches `kkt_indices`.
"""
function flatten_variables(sol::ACOPFSolution, prob::ACOPFProblem)
    return vcat(
        sol.va, sol.vm, sol.pg, sol.qg,
        sol.nu_p_bal, sol.nu_q_bal, sol.nu_ref_bus,
        sol.lam_thermal_fr, sol.lam_thermal_to,
        sol.lam_angle_lb, sol.lam_angle_ub,
        sol.mu_vm_lb, sol.mu_vm_ub,
        sol.rho_pg_lb, sol.rho_pg_ub, sol.rho_qg_lb, sol.rho_qg_ub,
        sol.sig_p_fr_lb, sol.sig_p_fr_ub, sol.sig_q_fr_lb, sol.sig_q_fr_ub,
        sol.sig_p_to_lb, sol.sig_p_to_ub, sol.sig_q_to_lb, sol.sig_q_to_ub
    )
end

"""
    unflatten_variables(z::AbstractVector, prob::ACOPFProblem)

Unflatten KKT variable vector into named components.

# Returns
NamedTuple with fields for all primal and dual variables.
"""
function unflatten_variables(z::AbstractVector, prob::ACOPFProblem)
    unflatten_variables(z, kkt_indices(prob))
end

function unflatten_variables(z::AbstractVector, idx::NamedTuple)
    return (
        va = z[idx.va], vm = z[idx.vm], pg = z[idx.pg], qg = z[idx.qg],
        nu_p_bal = z[idx.nu_p_bal], nu_q_bal = z[idx.nu_q_bal], nu_ref_bus = z[idx.nu_ref_bus],
        lam_thermal_fr = z[idx.lam_thermal_fr], lam_thermal_to = z[idx.lam_thermal_to],
        lam_angle_lb = z[idx.lam_angle_lb], lam_angle_ub = z[idx.lam_angle_ub],
        mu_vm_lb = z[idx.mu_vm_lb], mu_vm_ub = z[idx.mu_vm_ub],
        rho_pg_lb = z[idx.rho_pg_lb], rho_pg_ub = z[idx.rho_pg_ub],
        rho_qg_lb = z[idx.rho_qg_lb], rho_qg_ub = z[idx.rho_qg_ub],
        sig_p_fr_lb = z[idx.sig_p_fr_lb], sig_p_fr_ub = z[idx.sig_p_fr_ub],
        sig_q_fr_lb = z[idx.sig_q_fr_lb], sig_q_fr_ub = z[idx.sig_q_fr_ub],
        sig_p_to_lb = z[idx.sig_p_to_lb], sig_p_to_ub = z[idx.sig_p_to_ub],
        sig_q_to_lb = z[idx.sig_q_to_lb], sig_q_to_ub = z[idx.sig_q_to_ub]
    )
end

# =============================================================================
# KKT Jacobian via ForwardDiff
# =============================================================================

"""
    calc_kkt_jacobian(prob::ACOPFProblem; sol=nothing)

Compute the Jacobian of the KKT operator using ForwardDiff.

# Arguments
- `prob`: ACOPFProblem
- `sol`: Optional pre-computed solution. If not provided, ensures the problem is solved (reusing cached solution if available).

# Returns
Matrix ∂K/∂z where z is the flattened variable vector.
"""
function calc_kkt_jacobian(prob::ACOPFProblem; sol::Union{ACOPFSolution,Nothing}=nothing)
    if isnothing(sol)
        sol = _ensure_ac_solved!(prob)
    end

    z0 = flatten_variables(sol, prob)
    sw = prob.network.sw

    # Pre-extract all parameters so ForwardDiff closure avoids Dict lookups
    # and temporary allocations on every evaluation.
    pd0 = _extract_bus_pd(prob)
    qd0 = _extract_bus_qd(prob)
    cq0 = _extract_gen_cq(prob)
    cl0 = _extract_gen_cl(prob)
    fmax0 = _extract_branch_fmax(prob)

    # Pre-compute indices and constants so kkt() doesn't recompute them
    # on every ForwardDiff evaluation
    idx = kkt_indices(prob)
    constants = _extract_kkt_constants(prob)
    prob.cache.kkt_constants = constants

    J = ForwardDiff.jacobian(
        z -> kkt(z, prob, sw; pd=pd0, qd=qd0, cq=cq0, cl=cl0, fmax=fmax0,
                    idx=idx, constants=constants),
        z0
    )

    return J
end

# =============================================================================
# Branch Flow Calculations
# =============================================================================

"""
Compute all branch flows given voltage state and switching state.
Returns vectors of p_fr, q_fr, p_to, q_to indexed by branch number.

The switching variable sw_l multiplies each flow, so sw_l=0 means the branch
contributes zero flow (open), sw_l=1 means full flow (closed).
"""
function _compute_branch_flows(va, vm, net::ACNetwork, ref, sw; constants=nothing)
    m = net.m
    T = promote_type(eltype(va), eltype(vm), eltype(sw))
    p_fr = zeros(T, m)
    q_fr = zeros(T, m)
    p_to = zeros(T, m)
    q_to = zeros(T, m)

    for l in 1:m
        if isnothing(constants)
            branch = ref[:branch][l]
            f_bus = branch["f_bus"]
            t_bus = branch["t_bus"]
            g_br, b_br = PM.calc_branch_y(branch)
            tr, ti = PM.calc_branch_t(branch)
            g_fr_shunt = branch["g_fr"]
            b_fr_shunt = branch["b_fr"]
            g_to_shunt = branch["g_to"]
            b_to_shunt = branch["b_to"]
            tm = branch["tap"]^2
        else
            f_bus = constants.f_bus[l]
            t_bus = constants.t_bus[l]
            g_br = constants.g_br[l]
            b_br = constants.b_br[l]
            tr = constants.tr[l]
            ti = constants.ti[l]
            g_fr_shunt = constants.g_fr[l]
            b_fr_shunt = constants.b_fr[l]
            g_to_shunt = constants.g_to[l]
            b_to_shunt = constants.b_to[l]
            tm = constants.tm[l]
        end

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
Compute the reduced-space Lagrangian L(va, vm, pg, qg; duals, sw).

In the reduced space, flows are functions of (va, vm), not separate variables.
ForwardDiff.gradient of this function w.r.t. [va; vm; pg; qg] gives the
stationarity conditions, automatically handling all chain-rule terms including
power balance, thermal limits, flow bounds, angle diffs, and shunt terms.

Uses the JuMP/MOI dual sign convention:
    L = f(x) - Σ dual_i * normalized_residual_i
where normalized_residual = constraint_function - set_value.
"""
function _reduced_lagrangian(x_primal, vars, prob::ACOPFProblem, sw;
                             pd=nothing, qd=nothing, cq=nothing, cl=nothing, fmax=nothing,
                             constants=nothing)
    net = prob.network
    ref = prob.ref
    n, m, k = net.n, net.m, prob.n_gen

    va = x_primal[1:n]
    vm = x_primal[n+1:2n]
    pg = x_primal[2n+1:2n+k]
    qg = x_primal[2n+k+1:2n+2k]

    # Compute reduced-space flows
    p_fr, q_fr, p_to, q_to = _compute_branch_flows(va, vm, net, ref, sw; constants=constants)

    # Use widest type from computed flows and override vectors, which captures
    # ForwardDiff dual types in nested differentiation.
    _et(x) = isnothing(x) ? Float64 : eltype(x)
    T = promote_type(eltype(p_fr), _et(pd), _et(qd), _et(cq), _et(cl), _et(fmax))
    L = zero(T)

    # Materialize rate_a vector once to avoid repeated isnothing checks
    rate_a = isnothing(fmax) ? T[ref[:branch][l]["rate_a"] for l in 1:m] : fmax

    # ----- Objective -----
    for i in 1:k
        cq_i = isnothing(cq) ? ref[:gen][i]["cost"][1] : cq[i]
        cl_i = isnothing(cl) ? ref[:gen][i]["cost"][2] : cl[i]
        cc_i = isnothing(constants) ? ref[:gen][i]["cost"][3] : constants.cc[i]
        L += cq_i * pg[i]^2 + cl_i * pg[i] + cc_i
    end

    # ----- Power balance (equality): h = 0 -----
    # h_P[i] = Σ p_flow + gs*vm² - pg_sum + pd
    # h_Q[i] = Σ q_flow - bs*vm² - qg_sum + qd
    p_flow_sum = zeros(T, n)
    q_flow_sum = zeros(T, n)
    for l in 1:m
        fb = isnothing(constants) ? ref[:branch][l]["f_bus"] : constants.f_bus[l]
        tb = isnothing(constants) ? ref[:branch][l]["t_bus"] : constants.t_bus[l]
        p_flow_sum[fb] += p_fr[l]
        p_flow_sum[tb] += p_to[l]
        q_flow_sum[fb] += q_fr[l]
        q_flow_sum[tb] += q_to[l]
    end

    pg_sum = zeros(T, n)
    qg_sum = zeros(T, n)
    for i in 1:k
        bus_idx = isnothing(constants) ? ref[:gen][i]["gen_bus"] : constants.gen_bus[i]
        pg_sum[bus_idx] += pg[i]
        qg_sum[bus_idx] += qg[i]
    end

    for i in 1:n
        gs_i = isnothing(constants) ? sum(ref[:shunt][s]["gs"] for s in ref[:bus_shunts][i]; init=0.0) : constants.gs[i]
        bs_i = isnothing(constants) ? sum(ref[:shunt][s]["bs"] for s in ref[:bus_shunts][i]; init=0.0) : constants.bs[i]

        pd_i = isnothing(pd) ? sum(ref[:load][l]["pd"] for l in ref[:bus_loads][i]; init=0.0) : pd[i]
        qd_i = isnothing(qd) ? sum(ref[:load][l]["qd"] for l in ref[:bus_loads][i]; init=0.0) : qd[i]

        h_P = p_flow_sum[i] + gs_i * vm[i]^2 - pg_sum[i] + pd_i
        h_Q = q_flow_sum[i] - bs_i * vm[i]^2 - qg_sum[i] + qd_i

        L -= vars.nu_p_bal[i] * h_P
        L -= vars.nu_q_bal[i] * h_Q
    end

    # ----- Reference bus (equality): va[ref] - 0 = 0 -----
    rbk = isnothing(constants) ? _ref_bus_indices(prob) : constants.ref_bus_keys
    for (j, ref_bus_idx) in enumerate(rbk)
        L -= vars.nu_ref_bus[j] * va[ref_bus_idx]
    end

    # ----- Thermal limits (inequality): p²+q²-r² ≤ 0 -----
    for l in 1:m
        L -= vars.lam_thermal_fr[l] * (p_fr[l]^2 + q_fr[l]^2 - rate_a[l]^2)
        L -= vars.lam_thermal_to[l] * (p_to[l]^2 + q_to[l]^2 - rate_a[l]^2)
    end

    # ----- Angle difference limits (inequality) -----
    for l in 1:m
        fb = isnothing(constants) ? ref[:branch][l]["f_bus"] : constants.f_bus[l]
        tb = isnothing(constants) ? ref[:branch][l]["t_bus"] : constants.t_bus[l]
        amin = isnothing(constants) ? ref[:branch][l]["angmin"] : constants.angmin[l]
        amax = isnothing(constants) ? ref[:branch][l]["angmax"] : constants.angmax[l]
        L -= vars.lam_angle_lb[l] * (va[fb] - va[tb] - amin)
        L -= vars.lam_angle_ub[l] * (va[fb] - va[tb] - amax)
    end

    # ----- Voltage bounds (inequality) -----
    for i in 1:n
        vmin_i = isnothing(constants) ? ref[:bus][i]["vmin"] : constants.vmin[i]
        vmax_i = isnothing(constants) ? ref[:bus][i]["vmax"] : constants.vmax[i]
        L -= vars.mu_vm_lb[i] * (vm[i] - vmin_i)
        L -= vars.mu_vm_ub[i] * (vm[i] - vmax_i)
    end

    # ----- Generation bounds (inequality) -----
    for i in 1:k
        pmin_i = isnothing(constants) ? ref[:gen][i]["pmin"] : constants.pmin[i]
        pmax_i = isnothing(constants) ? ref[:gen][i]["pmax"] : constants.pmax[i]
        qmin_i = isnothing(constants) ? ref[:gen][i]["qmin"] : constants.qmin[i]
        qmax_i = isnothing(constants) ? ref[:gen][i]["qmax"] : constants.qmax[i]
        L -= vars.rho_pg_lb[i] * (pg[i] - pmin_i)
        L -= vars.rho_pg_ub[i] * (pg[i] - pmax_i)
        L -= vars.rho_qg_lb[i] * (qg[i] - qmin_i)
        L -= vars.rho_qg_ub[i] * (qg[i] - qmax_i)
    end

    # ----- Flow variable bounds (inequality, reduced-space) -----
    for l in 1:m
        L -= vars.sig_p_fr_lb[l] * (p_fr[l] + rate_a[l])
        L -= vars.sig_p_fr_ub[l] * (p_fr[l] - rate_a[l])
        L -= vars.sig_q_fr_lb[l] * (q_fr[l] + rate_a[l])
        L -= vars.sig_q_fr_ub[l] * (q_fr[l] - rate_a[l])
        L -= vars.sig_p_to_lb[l] * (p_to[l] + rate_a[l])
        L -= vars.sig_p_to_ub[l] * (p_to[l] - rate_a[l])
        L -= vars.sig_q_to_lb[l] * (q_to[l] + rate_a[l])
        L -= vars.sig_q_to_ub[l] * (q_to[l] - rate_a[l])
    end

    return L
end

# =============================================================================
# Power Balance Residuals (primal feasibility)
# =============================================================================

"""
Power balance residuals.
"""
function _power_balance_residuals(va, vm, pg, qg, p_fr, q_fr, p_to, q_to,
                                  net::ACNetwork, ref, prob::ACOPFProblem;
                                  pd=nothing, qd=nothing, constants=nothing)
    n = net.n
    m = net.m
    _et(x) = isnothing(x) ? Float64 : eltype(x)
    T = promote_type(eltype(va), eltype(vm), eltype(pg), eltype(p_fr), _et(pd), _et(qd))
    K_p_bal = zeros(T, n)
    K_q_bal = zeros(T, n)

    # Sum flows at each bus
    p_flow_sum = zeros(T, n)
    q_flow_sum = zeros(T, n)

    for l in 1:m
        fb = isnothing(constants) ? ref[:branch][l]["f_bus"] : constants.f_bus[l]
        tb = isnothing(constants) ? ref[:branch][l]["t_bus"] : constants.t_bus[l]
        p_flow_sum[fb] += p_fr[l]
        p_flow_sum[tb] += p_to[l]
        q_flow_sum[fb] += q_fr[l]
        q_flow_sum[tb] += q_to[l]
    end

    # Sum generation at each bus
    pg_sum = zeros(T, n)
    qg_sum = zeros(T, n)
    for i in 1:prob.n_gen
        bus_idx = isnothing(constants) ? ref[:gen][i]["gen_bus"] : constants.gen_bus[i]
        pg_sum[bus_idx] += pg[i]
        qg_sum[bus_idx] += qg[i]
    end

    for i in 1:n
        gs_i = isnothing(constants) ? sum(ref[:shunt][s]["gs"] for s in ref[:bus_shunts][i]; init=0.0) : constants.gs[i]
        bs_i = isnothing(constants) ? sum(ref[:shunt][s]["bs"] for s in ref[:bus_shunts][i]; init=0.0) : constants.bs[i]

        pd_i = isnothing(pd) ? sum(ref[:load][l]["pd"] for l in ref[:bus_loads][i]; init=0.0) : pd[i]
        qd_i = isnothing(qd) ? sum(ref[:load][l]["qd"] for l in ref[:bus_loads][i]; init=0.0) : qd[i]

        K_p_bal[i] = p_flow_sum[i] + gs_i * vm[i]^2 - pg_sum[i] + pd_i
        K_q_bal[i] = q_flow_sum[i] - bs_i * vm[i]^2 - qg_sum[i] + qd_i
    end

    return K_p_bal, K_q_bal
end

# =============================================================================
# KKT Operator
# =============================================================================

"""
    kkt(z::AbstractVector, prob::ACOPFProblem, sw::AbstractVector)

Evaluate the KKT conditions for AC OPF at the given variable vector.

The switching state sw is passed as a separate parameter to enable
differentiation with respect to switching using ForwardDiff.

Returns a vector of KKT residuals (should be zero at optimum).

# KKT Conditions (reduced-space formulation)
1. Stationarity w.r.t. va, vm, pg, qg (via ForwardDiff on Lagrangian)
2. Primal feasibility: power balance, reference bus
3. Complementary slackness for all inequality constraints
"""
function kkt(z::AbstractVector, prob::ACOPFProblem, sw::AbstractVector;
                pd=nothing, qd=nothing, cq=nothing, cl=nothing, fmax=nothing,
                idx=nothing, constants=nothing)
    if isnothing(idx)
        idx = kkt_indices(prob)
    end
    vars = unflatten_variables(z, idx)
    net = prob.network
    ref = prob.ref
    n, m, k = net.n, net.m, prob.n_gen

    va, vm = vars.va, vars.vm
    pg, qg = vars.pg, vars.qg

    _et(x) = isnothing(x) ? Float64 : eltype(x)
    T = promote_type(eltype(z), eltype(sw), _et(pd), _et(qd), _et(cq), _et(cl), _et(fmax))

    # Compute branch flows as functions of voltages
    p_fr, q_fr, p_to, q_to = _compute_branch_flows(va, vm, net, ref, sw; constants=constants)

    # Materialize rate_a vector once to avoid repeated isnothing checks
    rate_a = isnothing(fmax) ? T[ref[:branch][l]["rate_a"] for l in 1:m] : fmax

    # Pre-allocate KKT residual vector
    K = fill(T(NaN), last(idx.sig_q_to_ub))

    # =========================================================================
    # 1. Stationarity conditions via ForwardDiff on the Lagrangian
    # =========================================================================
    x_primal = vcat(va, vm, pg, qg)
    grad = ForwardDiff.gradient(
        x -> _reduced_lagrangian(x, vars, prob, sw;
                                 pd=pd, qd=qd, cq=cq, cl=cl, fmax=fmax,
                                 constants=constants),
        x_primal
    )
    # grad = [∂L/∂va; ∂L/∂vm; ∂L/∂pg; ∂L/∂qg]
    K[idx.va] = grad[idx.va]
    K[idx.vm] = grad[idx.vm]
    K[idx.pg] = grad[idx.pg]
    K[idx.qg] = grad[idx.qg]

    # =========================================================================
    # 2. Primal feasibility
    # =========================================================================

    # Power balance
    K_p_bal, K_q_bal = _power_balance_residuals(va, vm, pg, qg, p_fr, q_fr, p_to, q_to,
                                                 net, ref, prob; pd=pd, qd=qd,
                                                 constants=constants)
    K[idx.nu_p_bal] = K_p_bal
    K[idx.nu_q_bal] = K_q_bal

    # Reference bus: va[ref_bus] == 0
    rbk = isnothing(constants) ? _ref_bus_indices(prob) : constants.ref_bus_keys
    for (j, ref_bus_idx) in enumerate(rbk)
        K[idx.nu_ref_bus[j]] = va[ref_bus_idx]
    end

    # =========================================================================
    # 3. Complementary slackness conditions (vectorized)
    # =========================================================================
    # Lower bounds: L -= λ*(x - lb),  CS = λ*(x - lb) = 0  (same residual sign)
    # Upper bounds: L -= λ*(x - ub),  CS = λ*(ub - x) = 0  (negated residual)
    # Both are valid; the sign flip cancels in implicit differentiation.

    # Use pre-extracted bounds when available to avoid allocations in ForwardDiff
    if isnothing(constants)
        vmin = T[ref[:bus][i]["vmin"] for i in 1:n]
        vmax = T[ref[:bus][i]["vmax"] for i in 1:n]
        pmin = T[ref[:gen][i]["pmin"] for i in 1:k]
        pmax = T[ref[:gen][i]["pmax"] for i in 1:k]
        qmin = T[ref[:gen][i]["qmin"] for i in 1:k]
        qmax = T[ref[:gen][i]["qmax"] for i in 1:k]
        f_bus_idx = [ref[:branch][l]["f_bus"] for l in 1:m]
        t_bus_idx = [ref[:branch][l]["t_bus"] for l in 1:m]
        angmin = T[ref[:branch][l]["angmin"] for l in 1:m]
        angmax = T[ref[:branch][l]["angmax"] for l in 1:m]
    else
        vmin = constants.vmin
        vmax = constants.vmax
        pmin = constants.pmin
        pmax = constants.pmax
        qmin = constants.qmin
        qmax = constants.qmax
        f_bus_idx = constants.f_bus
        t_bus_idx = constants.t_bus
        angmin = constants.angmin
        angmax = constants.angmax
    end

    # Thermal limits
    K[idx.lam_thermal_fr] .= vars.lam_thermal_fr .* (p_fr.^2 .+ q_fr.^2 .- rate_a.^2)
    K[idx.lam_thermal_to] .= vars.lam_thermal_to .* (p_to.^2 .+ q_to.^2 .- rate_a.^2)

    # Angle difference limits
    K[idx.lam_angle_lb] .= vars.lam_angle_lb .* (va[f_bus_idx] .- va[t_bus_idx] .- angmin)
    K[idx.lam_angle_ub] .= vars.lam_angle_ub .* (angmax .- va[f_bus_idx] .+ va[t_bus_idx])

    # Voltage bounds
    K[idx.mu_vm_lb] .= vars.mu_vm_lb .* (vm .- vmin)
    K[idx.mu_vm_ub] .= vars.mu_vm_ub .* (vmax .- vm)

    # Generation bounds
    K[idx.rho_pg_lb] .= vars.rho_pg_lb .* (pg .- pmin)
    K[idx.rho_pg_ub] .= vars.rho_pg_ub .* (pmax .- pg)
    K[idx.rho_qg_lb] .= vars.rho_qg_lb .* (qg .- qmin)
    K[idx.rho_qg_ub] .= vars.rho_qg_ub .* (qmax .- qg)

    # Flow variable bounds (reduced-space)
    K[idx.sig_p_fr_lb] .= vars.sig_p_fr_lb .* (p_fr .+ rate_a)
    K[idx.sig_p_fr_ub] .= vars.sig_p_fr_ub .* (rate_a .- p_fr)
    K[idx.sig_q_fr_lb] .= vars.sig_q_fr_lb .* (q_fr .+ rate_a)
    K[idx.sig_q_fr_ub] .= vars.sig_q_fr_ub .* (rate_a .- q_fr)
    K[idx.sig_p_to_lb] .= vars.sig_p_to_lb .* (p_to .+ rate_a)
    K[idx.sig_p_to_ub] .= vars.sig_p_to_ub .* (rate_a .- p_to)
    K[idx.sig_q_to_lb] .= vars.sig_q_to_lb .* (q_to .+ rate_a)
    K[idx.sig_q_to_ub] .= vars.sig_q_to_ub .* (rate_a .- q_to)

    return K
end

# Convenience method using prob's switching state
kkt(z::AbstractVector, prob::ACOPFProblem) = kkt(z, prob, prob.network.sw)

# =============================================================================
# Parameter Extraction Functions
# =============================================================================

"""Extract per-bus aggregated load values for a given key ("pd" or "qd")."""
function _extract_bus_load(prob::ACOPFProblem, key::String)
    ref = prob.ref
    n = prob.network.n
    vals = zeros(n)
    for i in 1:n
        for lid in ref[:bus_loads][i]
            vals[i] += ref[:load][lid][key]
        end
    end
    return vals
end

"""Extract per-generator cost coefficient at a given index (1=quadratic, 2=linear)."""
function _extract_gen_cost(prob::ACOPFProblem, cost_idx::Int)
    k = prob.n_gen
    vals = zeros(k)
    for i in 1:k
        vals[i] = prob.ref[:gen][i]["cost"][cost_idx]
    end
    return vals
end

_extract_bus_pd(prob::ACOPFProblem) = _extract_bus_load(prob, "pd")
_extract_bus_qd(prob::ACOPFProblem) = _extract_bus_load(prob, "qd")
_extract_gen_cq(prob::ACOPFProblem) = _extract_gen_cost(prob, 1)
_extract_gen_cl(prob::ACOPFProblem) = _extract_gen_cost(prob, 2)

"""Extract per-branch flow limits (rate_a) from the problem's ref."""
function _extract_branch_fmax(prob::ACOPFProblem)
    m = prob.network.m
    fmax = zeros(m)
    for l in 1:m
        fmax[l] = prob.ref[:branch][l]["rate_a"]
    end
    return fmax
end

"""
Pre-extract all constant data from the problem's ref for efficient ForwardDiff evaluation.
Avoids repeated Dict lookups and PM.calc_branch_y/t calls inside ForwardDiff closures.
"""
function _extract_kkt_constants(prob::ACOPFProblem)
    ref = prob.ref
    n, m, k = prob.network.n, prob.network.m, prob.n_gen

    # Branch electrical parameters
    bp_g = Vector{Float64}(undef, m)
    bp_b = Vector{Float64}(undef, m)
    bp_tr = Vector{Float64}(undef, m)
    bp_ti = Vector{Float64}(undef, m)
    bp_g_fr = Vector{Float64}(undef, m)
    bp_b_fr = Vector{Float64}(undef, m)
    bp_g_to = Vector{Float64}(undef, m)
    bp_b_to = Vector{Float64}(undef, m)
    bp_tm = Vector{Float64}(undef, m)
    bp_f_bus = Vector{Int}(undef, m)
    bp_t_bus = Vector{Int}(undef, m)
    bp_angmin = Vector{Float64}(undef, m)
    bp_angmax = Vector{Float64}(undef, m)
    for l in 1:m
        branch = ref[:branch][l]
        bp_g[l], bp_b[l] = PM.calc_branch_y(branch)
        bp_tr[l], bp_ti[l] = PM.calc_branch_t(branch)
        bp_g_fr[l] = branch["g_fr"]
        bp_b_fr[l] = branch["b_fr"]
        bp_g_to[l] = branch["g_to"]
        bp_b_to[l] = branch["b_to"]
        bp_tm[l] = branch["tap"]^2
        bp_f_bus[l] = branch["f_bus"]
        bp_t_bus[l] = branch["t_bus"]
        bp_angmin[l] = branch["angmin"]
        bp_angmax[l] = branch["angmax"]
    end

    return (
        # Branch parameters
        g_br = bp_g, b_br = bp_b, tr = bp_tr, ti = bp_ti,
        g_fr = bp_g_fr, b_fr = bp_b_fr, g_to = bp_g_to, b_to = bp_b_to,
        tm = bp_tm, f_bus = bp_f_bus, t_bus = bp_t_bus,
        angmin = bp_angmin, angmax = bp_angmax,
        # Bus bounds
        vmin = Float64[ref[:bus][i]["vmin"] for i in 1:n],
        vmax = Float64[ref[:bus][i]["vmax"] for i in 1:n],
        # Gen bounds and parameters
        pmin = Float64[ref[:gen][i]["pmin"] for i in 1:k],
        pmax = Float64[ref[:gen][i]["pmax"] for i in 1:k],
        qmin = Float64[ref[:gen][i]["qmin"] for i in 1:k],
        qmax = Float64[ref[:gen][i]["qmax"] for i in 1:k],
        gen_bus = Int[ref[:gen][i]["gen_bus"] for i in 1:k],
        cc = Float64[ref[:gen][i]["cost"][3] for i in 1:k],
        # Shunt parameters (aggregated per bus)
        gs = Float64[sum(ref[:shunt][s]["gs"] for s in ref[:bus_shunts][i]; init=0.0) for i in 1:n],
        bs = Float64[sum(ref[:shunt][s]["bs"] for s in ref[:bus_shunts][i]; init=0.0) for i in 1:n],
        # Reference bus
        ref_bus_keys = sort(collect(keys(ref[:ref_buses]))),
    )
end

# =============================================================================
# Parameter Jacobian (via ForwardDiff)
# =============================================================================

# Map parameter symbols to extraction functions
const _AC_PARAM_EXTRACT = Dict{Symbol, Function}(
    :sw   => prob -> prob.network.sw,
    :d    => _extract_bus_pd,
    :qd   => _extract_bus_qd,
    :cq   => _extract_gen_cq,
    :cl   => _extract_gen_cl,
    :fmax => _extract_branch_fmax,
)

# Maps external parameter symbols to kkt() keyword argument names
# (e.g., user-facing :d becomes the internal :pd kwarg for active demand)
const _PARAM_KWARG_MAP = Dict{Symbol, Symbol}(
    :d => :pd, :qd => :qd, :cq => :cq, :cl => :cl, :fmax => :fmax,
)

"""
    calc_kkt_jacobian_param(prob::ACOPFProblem, sol::ACOPFSolution, param::Symbol)

Compute ∂K/∂param via ForwardDiff for any supported parameter symbol.
Returns matrix of size (kkt_dims × param_dims).
"""
function calc_kkt_jacobian_param(prob::ACOPFProblem, sol::ACOPFSolution, param::Symbol)
    haskey(_AC_PARAM_EXTRACT, param) || throw(ArgumentError(
        "Unknown AC OPF parameter: $param. Valid: $(keys(_AC_PARAM_EXTRACT))"))
    z0 = flatten_variables(sol, prob)
    sw = prob.network.sw
    p0 = _AC_PARAM_EXTRACT[param](prob)
    idx = kkt_indices(prob)
    constants = prob.cache.kkt_constants
    if isnothing(constants)
        constants = _extract_kkt_constants(prob)
        prob.cache.kkt_constants = constants
    end

    # Pre-extract all fixed parameters so the ForwardDiff closure avoids Dict lookups.
    # Only the differentiated parameter is passed as the ForwardDiff variable;
    # all others are passed as fixed keyword arguments.
    pd0 = _extract_bus_pd(prob)
    qd0 = _extract_bus_qd(prob)
    cq0 = _extract_gen_cq(prob)
    cl0 = _extract_gen_cl(prob)
    fmax0 = _extract_branch_fmax(prob)

    if param === :sw
        return ForwardDiff.jacobian(
            s -> kkt(z0, prob, s; pd=pd0, qd=qd0, cq=cq0, cl=cl0, fmax=fmax0,
                        idx=idx, constants=constants), p0)
    else
        kw = _PARAM_KWARG_MAP[param]
        # Build fixed kwargs with all parameters except the one being differentiated
        all_params = Dict{Symbol,Any}(:pd => pd0, :qd => qd0, :cq => cq0, :cl => cl0, :fmax => fmax0)
        delete!(all_params, kw)
        fixed_nt = (; (k => v for (k, v) in all_params)...)
        return ForwardDiff.jacobian(
            x -> kkt(z0, prob, sw; NamedTuple{(kw,)}((x,))..., fixed_nt...,
                        idx=idx, constants=constants), p0)
    end
end

# =============================================================================
# Cached Solution and KKT Factorization Access
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
    _ensure_ac_kkt_factor!(prob::ACOPFProblem) → LU

Ensure the KKT Jacobian factorization is computed and cached.
Returns the LU factorization for efficient repeated solves.
"""
function _ensure_ac_kkt_factor!(prob::ACOPFProblem)
    if isnothing(prob.cache.kkt_factor)
        sol = _ensure_ac_solved!(prob)
        J_z = calc_kkt_jacobian(prob; sol=sol)
        prob.cache.kkt_factor = try
            lu(J_z)
        catch e
            if e isa LinearAlgebra.SingularException
                _SILENCE_WARNINGS[] || @warn "AC KKT Jacobian is singular (likely degenerate complementarity, e.g., generators at bounds); applying Tikhonov perturbation (eps=$TIKHONOV_EPS). Sensitivity accuracy may be reduced."
                J_reg = J_z + TIKHONOV_EPS * I
                try
                    lu(J_reg)
                catch e2
                    e2 isa LinearAlgebra.SingularException || rethrow(e2)
                    error("AC KKT Jacobian remains singular after Tikhonov perturbation")
                end
            else
                rethrow(e)
            end
        end
    end
    return prob.cache.kkt_factor
end

# =============================================================================
# Cached Derivative Computation
# =============================================================================

# Map parameter symbols to cache field names
const _AC_CACHE_FIELD = Dict{Symbol, Symbol}(
    :sw => :dz_dsw, :d => :dz_dd, :qd => :dz_dqd,
    :cq => :dz_dcq, :cl => :dz_dcl, :fmax => :dz_dfmax,
)

"""
    _get_ac_dz_dparam!(prob::ACOPFProblem, param::Symbol) → Matrix{Float64}

Get or compute ∂z/∂param = -(∂K/∂z)⁻¹ · (∂K/∂param). Uses shared KKT factorization
and caches the result for reuse across different operand queries.
"""
function _get_ac_dz_dparam!(prob::ACOPFProblem, param::Symbol)::Matrix{Float64}
    field = _AC_CACHE_FIELD[param]
    cached = getfield(prob.cache, field)
    if isnothing(cached)
        kkt_lu = _ensure_ac_kkt_factor!(prob)
        sol = _ensure_ac_solved!(prob)
        J_p = calc_kkt_jacobian_param(prob, sol, param)
        ldiv!(kkt_lu, J_p)
        lmul!(-1, J_p)
        setfield!(prob.cache, field, J_p)
        return J_p
    end
    return cached
end

# =============================================================================
# Single-Column Helpers
# =============================================================================

"""
    _ac_operand_kkt_rows(idx::NamedTuple, op::Symbol) → UnitRange{Int}

Return the KKT index range for an AC OPF operand.
"""
function _ac_operand_kkt_rows(idx::NamedTuple, op::Symbol)
    op === :va   && return idx.va
    op === :vm   && return idx.vm
    op === :pg   && return idx.pg
    op === :qg   && return idx.qg
    op === :lmp  && return idx.nu_p_bal
    op === :qlmp && return idx.nu_q_bal
    throw(ArgumentError("Unknown AC OPF operand: $op"))
end

# AC OPF: LMP = -ν_p_bal (negation required).
# The constraint P_flow + P_d - P_g = 0 places demand positively →
# JuMP dual ν_p_bal < 0 at optimum. See lmp.jl:38-41.
_ac_operand_sign(op::Symbol) = (op === :lmp || op === :qlmp) ? -1.0 : 1.0

"""
    _extract_ac_dz_column(prob, dz_dp::Matrix{Float64}, op::Symbol, col_idx::Int) → Vector{Float64}

Extract operand rows from column col_idx of a cached full dz/dp matrix.
"""
function _extract_ac_dz_column(prob::ACOPFProblem, dz_dp::Matrix{Float64}, op::Symbol, col_idx::Int)
    idx = kkt_indices(prob)
    col = dz_dp[_ac_operand_kkt_rows(idx, op), col_idx]
    _ac_operand_sign(op) == -1.0 && lmul!(-1, col)
    return col
end

"""
    _extract_ac_dz_column_vec(prob, dz_col::Vector{Float64}, op::Symbol) → Vector{Float64}

Extract operand rows from a single dz/dp column vector.
"""
function _extract_ac_dz_column_vec(prob::ACOPFProblem, dz_col::Vector{Float64}, op::Symbol)
    idx = kkt_indices(prob)
    col = dz_col[_ac_operand_kkt_rows(idx, op)]
    _ac_operand_sign(op) == -1.0 && lmul!(-1, col)
    return col
end

"""
    _calc_ac_kkt_param_column(prob, sol, param, col_idx) → Vector{Float64}

Compute a single column of ∂K/∂param via ForwardDiff.derivative (scalar → vector).
Much cheaper than the full Jacobian when only one column is needed.
"""
function _calc_ac_kkt_param_column(prob::ACOPFProblem, sol::ACOPFSolution, param::Symbol, col_idx::Int)
    z0 = flatten_variables(sol, prob)
    sw = prob.network.sw
    p0 = _AC_PARAM_EXTRACT[param](prob)
    idx = kkt_indices(prob)
    constants = prob.cache.kkt_constants
    if isnothing(constants)
        constants = _extract_kkt_constants(prob)
        prob.cache.kkt_constants = constants
    end

    pd0 = _extract_bus_pd(prob)
    qd0 = _extract_bus_qd(prob)
    cq0 = _extract_gen_cq(prob)
    cl0 = _extract_gen_cl(prob)
    fmax0 = _extract_branch_fmax(prob)

    if param === :sw
        f_col = t -> begin
            # Promote to Dual-compatible type so ForwardDiff can track derivatives
            sw_t = typeof(t).(sw)
            sw_t[col_idx] = t
            kkt(z0, prob, sw_t; pd=pd0, qd=qd0, cq=cq0, cl=cl0, fmax=fmax0,
                idx=idx, constants=constants)
        end
        return ForwardDiff.derivative(f_col, sw[col_idx])
    else
        kw = _PARAM_KWARG_MAP[param]
        all_fixed = Dict{Symbol,Any}(:pd => pd0, :qd => qd0, :cq => cq0, :cl => cl0, :fmax => fmax0)
        delete!(all_fixed, kw)
        fixed_nt = (; (k => v for (k, v) in all_fixed)...)

        f_col = t -> begin
            p_t = typeof(t).(p0)
            p_t[col_idx] = t
            kkt(z0, prob, sw; NamedTuple{(kw,)}((p_t,))..., fixed_nt...,
                idx=idx, constants=constants)
        end
        return ForwardDiff.derivative(f_col, p0[col_idx])
    end
end
