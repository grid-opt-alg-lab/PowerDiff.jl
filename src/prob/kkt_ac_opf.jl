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
# Stationarity conditions are assembled analytically in reduced space.

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
# KKT Jacobian (analytical)
# =============================================================================

# =============================================================================
# Branch Flow Calculations
# =============================================================================

"""
Compute all branch flows given voltage state and switching state.
Returns vectors of p_fr, q_fr, p_to, q_to indexed by branch number.

Flow equations match PowerModels' `constraint_ohms_yt_from`/`constraint_ohms_yt_to`
(polar form with `calc_branch_y`/`calc_branch_t` decomposition, `tm = tap^2`).

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

"""
Return branch-flow coefficients in the form

    F = sw * (a * s^2 + b * vm_f * vm_t * cos(δ) + c * vm_f * vm_t * sin(δ))

with `δ = va_f - va_t`, for each of `p_fr`, `q_fr`, `p_to`, `q_to`.
"""
function _branch_flow_coefficients(constants, l::Int)
    g_br = constants.g_br[l]
    b_br = constants.b_br[l]
    tr = constants.tr[l]
    ti = constants.ti[l]
    g_fr = constants.g_fr[l]
    b_fr = constants.b_fr[l]
    g_to = constants.g_to[l]
    b_to = constants.b_to[l]
    tm = constants.tm[l]

    return (
        p_fr = (
            a = (g_br + g_fr) / tm,
            b = (-g_br * tr + b_br * ti) / tm,
            c = (-b_br * tr - g_br * ti) / tm,
        ),
        q_fr = (
            a = -(b_br + b_fr) / tm,
            b = (b_br * tr + g_br * ti) / tm,
            c = (-g_br * tr + b_br * ti) / tm,
        ),
        p_to = (
            a = g_br + g_to,
            b = (-g_br * tr - b_br * ti) / tm,
            c = (b_br * tr - g_br * ti) / tm,
        ),
        q_to = (
            a = -(b_br + b_to),
            b = (b_br * tr - g_br * ti) / tm,
            c = (g_br * tr + b_br * ti) / tm,
        ),
    )
end

@inline function _branch_flow_base(coeffs, self_vm, other_vm, cosδ, sinδ)
    return coeffs.a * self_vm^2 + self_vm * other_vm * (coeffs.b * cosδ + coeffs.c * sinδ)
end

@inline function _branch_flow_local_partials(coeffs, self_vm, other_vm, cosδ, sinδ)
    mix = coeffs.b * cosδ + coeffs.c * sinδ
    dδ = self_vm * other_vm * (-coeffs.b * sinδ + coeffs.c * cosδ)
    dself = 2 * coeffs.a * self_vm + other_vm * mix
    dother = self_vm * mix
    return dδ, dself, dother
end

@inline function _branch_flow_local_second_partials(coeffs, self_vm, other_vm, cosδ, sinδ)
    mix = coeffs.b * cosδ + coeffs.c * sinδ
    dδδ = -self_vm * other_vm * mix
    dδ_self = other_vm * (-coeffs.b * sinδ + coeffs.c * cosδ)
    dδ_other = self_vm * (-coeffs.b * sinδ + coeffs.c * cosδ)
    dself_self = 2 * coeffs.a
    dself_other = mix
    return dδδ, dδ_self, dδ_other, dself_self, dself_other
end

@inline function _global_hessian_from_fr(dδδ, dδ_self, dδ_other, dself_self, dself_other)
    return (
        va_f_va_f = dδδ,
        va_f_va_t = -dδδ,
        va_f_vm_f = dδ_self,
        va_f_vm_t = dδ_other,
        va_t_va_t = dδδ,
        va_t_vm_f = -dδ_self,
        va_t_vm_t = -dδ_other,
        vm_f_vm_f = dself_self,
        vm_f_vm_t = dself_other,
        vm_t_vm_t = zero(dself_self),
    )
end

@inline function _global_hessian_from_to(dδδ, dδ_self, dδ_other, dself_self, dself_other)
    return (
        va_f_va_f = dδδ,
        va_f_va_t = -dδδ,
        va_f_vm_f = dδ_other,
        va_f_vm_t = dδ_self,
        va_t_va_t = dδδ,
        va_t_vm_f = -dδ_other,
        va_t_vm_t = -dδ_self,
        vm_f_vm_f = zero(dself_self),
        vm_f_vm_t = dself_other,
        vm_t_vm_t = dself_self,
    )
end

"""
Return branch-flow bases (with `sw = 1`) and local first derivatives with
respect to `(va_f, va_t, vm_f, vm_t)`.
"""
function _branch_flow_primitives(va, vm, sw, constants, l::Int)
    fb = constants.f_bus[l]
    tb = constants.t_bus[l]
    vf = vm[fb]
    vt = vm[tb]
    δ = va[fb] - va[tb]
    cosδ = cos(δ)
    sinδ = sin(δ)
    sw_l = sw[l]
    coeffs = _branch_flow_coefficients(constants, l)

    p_fr_hat = _branch_flow_base(coeffs.p_fr, vf, vt, cosδ, sinδ)
    q_fr_hat = _branch_flow_base(coeffs.q_fr, vf, vt, cosδ, sinδ)
    p_to_hat = _branch_flow_base(coeffs.p_to, vt, vf, cosδ, sinδ)
    q_to_hat = _branch_flow_base(coeffs.q_to, vt, vf, cosδ, sinδ)

    dp_fr_dδ_hat, dp_fr_dvf_hat, dp_fr_dvt_hat =
        _branch_flow_local_partials(coeffs.p_fr, vf, vt, cosδ, sinδ)
    dq_fr_dδ_hat, dq_fr_dvf_hat, dq_fr_dvt_hat =
        _branch_flow_local_partials(coeffs.q_fr, vf, vt, cosδ, sinδ)
    dp_to_dδ_hat, dp_to_dvt_hat, dp_to_dvf_hat =
        _branch_flow_local_partials(coeffs.p_to, vt, vf, cosδ, sinδ)
    dq_to_dδ_hat, dq_to_dvt_hat, dq_to_dvf_hat =
        _branch_flow_local_partials(coeffs.q_to, vt, vf, cosδ, sinδ)

    d2p_fr = _global_hessian_from_fr(
        _branch_flow_local_second_partials(coeffs.p_fr, vf, vt, cosδ, sinδ)...)
    d2q_fr = _global_hessian_from_fr(
        _branch_flow_local_second_partials(coeffs.q_fr, vf, vt, cosδ, sinδ)...)
    d2p_to = _global_hessian_from_to(
        _branch_flow_local_second_partials(coeffs.p_to, vt, vf, cosδ, sinδ)...)
    d2q_to = _global_hessian_from_to(
        _branch_flow_local_second_partials(coeffs.q_to, vt, vf, cosδ, sinδ)...)

    return (
        fb = fb,
        tb = tb,
        sw = sw_l,
        p_fr_hat = p_fr_hat,
        q_fr_hat = q_fr_hat,
        p_to_hat = p_to_hat,
        q_to_hat = q_to_hat,
        p_fr = sw_l * p_fr_hat,
        q_fr = sw_l * q_fr_hat,
        p_to = sw_l * p_to_hat,
        q_to = sw_l * q_to_hat,
        dp_fr_hat = (dp_fr_dδ_hat, -dp_fr_dδ_hat, dp_fr_dvf_hat, dp_fr_dvt_hat),
        dq_fr_hat = (dq_fr_dδ_hat, -dq_fr_dδ_hat, dq_fr_dvf_hat, dq_fr_dvt_hat),
        dp_to_hat = (dp_to_dδ_hat, -dp_to_dδ_hat, dp_to_dvf_hat, dp_to_dvt_hat),
        dq_to_hat = (dq_to_dδ_hat, -dq_to_dδ_hat, dq_to_dvf_hat, dq_to_dvt_hat),
        d2p_fr_hat = d2p_fr,
        d2q_fr_hat = d2q_fr,
        d2p_to_hat = d2p_to,
        d2q_to_hat = d2q_to,
    )
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

@inline function _add_symmetric_local_hessian!(J::AbstractMatrix, rows::NTuple{4,Int},
                                               H, coeff, outer, grad::NTuple{4})
    i1, i2, i3, i4 = rows
    g1, g2, g3, g4 = grad
    J[i1, i1] += coeff * H.va_f_va_f + outer * g1 * g1
    J[i1, i2] += coeff * H.va_f_va_t + outer * g1 * g2
    J[i2, i1] = J[i1, i2]
    J[i1, i3] += coeff * H.va_f_vm_f + outer * g1 * g3
    J[i3, i1] = J[i1, i3]
    J[i1, i4] += coeff * H.va_f_vm_t + outer * g1 * g4
    J[i4, i1] = J[i1, i4]
    J[i2, i2] += coeff * H.va_t_va_t + outer * g2 * g2
    J[i2, i3] += coeff * H.va_t_vm_f + outer * g2 * g3
    J[i3, i2] = J[i2, i3]
    J[i2, i4] += coeff * H.va_t_vm_t + outer * g2 * g4
    J[i4, i2] = J[i2, i4]
    J[i3, i3] += coeff * H.vm_f_vm_f + outer * g3 * g3
    J[i3, i4] += coeff * H.vm_f_vm_t + outer * g3 * g4
    J[i4, i3] = J[i3, i4]
    J[i4, i4] += coeff * H.vm_t_vm_t + outer * g4 * g4
    return nothing
end

function _stationarity_residual!(K::AbstractVector, vars, prob::ACOPFProblem, sw;
                                 cq, cl, fmax, constants)
    idx = kkt_indices(prob)
    n, m, k = prob.network.n, prob.network.m, prob.n_gen

    fill!(@view(K[idx.va]), 0)
    fill!(@view(K[idx.vm]), 0)

    @inbounds for i in 1:n
        K[idx.vm[i]] += -2 * vars.nu_p_bal[i] * constants.gs[i] * vars.vm[i] +
                        2 * vars.nu_q_bal[i] * constants.bs[i] * vars.vm[i] -
                        vars.mu_vm_lb[i] - vars.mu_vm_ub[i]
    end

    @inbounds for i in 1:k
        bus = constants.gen_bus[i]
        K[idx.pg[i]] = 2 * cq[i] * vars.pg[i] + cl[i] + vars.nu_p_bal[bus] -
                       vars.rho_pg_lb[i] - vars.rho_pg_ub[i]
        K[idx.qg[i]] = vars.nu_q_bal[bus] - vars.rho_qg_lb[i] - vars.rho_qg_ub[i]
    end

    rbk = constants.ref_bus_keys
    @inbounds for (j, ref_bus_idx) in enumerate(rbk)
        K[idx.va[ref_bus_idx]] -= vars.nu_ref_bus[j]
    end

    @inbounds for l in 1:m
        prim = _branch_flow_primitives(vars.va, vars.vm, sw, constants, l)
        fb = prim.fb
        tb = prim.tb

        coeff_p_fr = -vars.nu_p_bal[fb] - 2 * vars.lam_thermal_fr[l] * prim.p_fr -
                     vars.sig_p_fr_lb[l] - vars.sig_p_fr_ub[l]
        coeff_q_fr = -vars.nu_q_bal[fb] - 2 * vars.lam_thermal_fr[l] * prim.q_fr -
                     vars.sig_q_fr_lb[l] - vars.sig_q_fr_ub[l]
        coeff_p_to = -vars.nu_p_bal[tb] - 2 * vars.lam_thermal_to[l] * prim.p_to -
                     vars.sig_p_to_lb[l] - vars.sig_p_to_ub[l]
        coeff_q_to = -vars.nu_q_bal[tb] - 2 * vars.lam_thermal_to[l] * prim.q_to -
                     vars.sig_q_to_lb[l] - vars.sig_q_to_ub[l]

        K[idx.va[fb]] += coeff_p_fr * prim.dp_fr_hat[1] + coeff_q_fr * prim.dq_fr_hat[1] +
                         coeff_p_to * prim.dp_to_hat[1] + coeff_q_to * prim.dq_to_hat[1] -
                         vars.lam_angle_lb[l] - vars.lam_angle_ub[l]
        K[idx.va[tb]] += coeff_p_fr * prim.dp_fr_hat[2] + coeff_q_fr * prim.dq_fr_hat[2] +
                         coeff_p_to * prim.dp_to_hat[2] + coeff_q_to * prim.dq_to_hat[2] +
                         vars.lam_angle_lb[l] + vars.lam_angle_ub[l]
        K[idx.vm[fb]] += coeff_p_fr * prim.dp_fr_hat[3] + coeff_q_fr * prim.dq_fr_hat[3] +
                         coeff_p_to * prim.dp_to_hat[3] + coeff_q_to * prim.dq_to_hat[3]
        K[idx.vm[tb]] += coeff_p_fr * prim.dp_fr_hat[4] + coeff_q_fr * prim.dq_fr_hat[4] +
                         coeff_p_to * prim.dp_to_hat[4] + coeff_q_to * prim.dq_to_hat[4]
    end

    return K
end

"""
    calc_kkt_jacobian(prob::ACOPFProblem; sol=nothing)

Compute the analytical Jacobian of the AC OPF KKT operator.
"""
function calc_kkt_jacobian(prob::ACOPFProblem; sol::Union{ACOPFSolution,Nothing}=nothing)
    if isnothing(sol)
        sol = _ensure_ac_solved!(prob)
    end

    idx = kkt_indices(prob)
    constants = _extract_kkt_constants(prob)
    prob.cache.kkt_constants = constants
    cq = _extract_gen_cq(prob)
    cl = _extract_gen_cl(prob)
    fmax = _extract_branch_fmax(prob)
    sw = prob.network.sw
    vars = unflatten_variables(flatten_variables(sol, prob), idx)
    n, m, k = prob.network.n, prob.network.m, prob.n_gen

    J = spzeros(Float64, kkt_dims(prob), kkt_dims(prob))

    @inbounds for i in 1:k
        J[idx.pg[i], idx.pg[i]] = 2 * cq[i]
    end
    @inbounds for i in 1:n
        J[idx.vm[i], idx.vm[i]] += -2 * vars.nu_p_bal[i] * constants.gs[i] +
                                   2 * vars.nu_q_bal[i] * constants.bs[i]
    end

    @inbounds for i in 1:k
        bus = constants.gen_bus[i]
        J[idx.pg[i], idx.nu_p_bal[bus]] = 1.0
        J[idx.pg[i], idx.rho_pg_lb[i]] = -1.0
        J[idx.pg[i], idx.rho_pg_ub[i]] = -1.0
        J[idx.qg[i], idx.nu_q_bal[bus]] = 1.0
        J[idx.qg[i], idx.rho_qg_lb[i]] = -1.0
        J[idx.qg[i], idx.rho_qg_ub[i]] = -1.0
    end
    @inbounds for i in 1:n
        J[idx.vm[i], idx.mu_vm_lb[i]] = -1.0
        J[idx.vm[i], idx.mu_vm_ub[i]] = -1.0
    end
    @inbounds for (j, ref_bus_idx) in enumerate(constants.ref_bus_keys)
        J[idx.va[ref_bus_idx], idx.nu_ref_bus[j]] = -1.0
    end

    @inbounds for l in 1:m
        prim = _branch_flow_primitives(vars.va, vars.vm, sw, constants, l)
        fb = prim.fb
        tb = prim.tb
        local_idx = (idx.va[fb], idx.va[tb], idx.vm[fb], idx.vm[tb])

        coeff_p_fr = -vars.nu_p_bal[fb] - 2 * vars.lam_thermal_fr[l] * prim.p_fr -
                     vars.sig_p_fr_lb[l] - vars.sig_p_fr_ub[l]
        coeff_q_fr = -vars.nu_q_bal[fb] - 2 * vars.lam_thermal_fr[l] * prim.q_fr -
                     vars.sig_q_fr_lb[l] - vars.sig_q_fr_ub[l]
        coeff_p_to = -vars.nu_p_bal[tb] - 2 * vars.lam_thermal_to[l] * prim.p_to -
                     vars.sig_p_to_lb[l] - vars.sig_p_to_ub[l]
        coeff_q_to = -vars.nu_q_bal[tb] - 2 * vars.lam_thermal_to[l] * prim.q_to -
                     vars.sig_q_to_lb[l] - vars.sig_q_to_ub[l]

        _add_symmetric_local_hessian!(J, local_idx, prim.d2p_fr_hat, coeff_p_fr, -2 * vars.lam_thermal_fr[l], prim.dp_fr_hat)
        _add_symmetric_local_hessian!(J, local_idx, prim.d2q_fr_hat, coeff_q_fr, -2 * vars.lam_thermal_fr[l], prim.dq_fr_hat)
        _add_symmetric_local_hessian!(J, local_idx, prim.d2p_to_hat, coeff_p_to, -2 * vars.lam_thermal_to[l], prim.dp_to_hat)
        _add_symmetric_local_hessian!(J, local_idx, prim.d2q_to_hat, coeff_q_to, -2 * vars.lam_thermal_to[l], prim.dq_to_hat)

        for t in 1:4
            row = local_idx[t]
            J[row, idx.nu_p_bal[fb]] += -prim.dp_fr_hat[t]
            J[row, idx.nu_q_bal[fb]] += -prim.dq_fr_hat[t]
            J[row, idx.nu_p_bal[tb]] += -prim.dp_to_hat[t]
            J[row, idx.nu_q_bal[tb]] += -prim.dq_to_hat[t]
            J[row, idx.lam_thermal_fr[l]] += -2 * (prim.p_fr * prim.dp_fr_hat[t] + prim.q_fr * prim.dq_fr_hat[t])
            J[row, idx.lam_thermal_to[l]] += -2 * (prim.p_to * prim.dp_to_hat[t] + prim.q_to * prim.dq_to_hat[t])
            J[row, idx.sig_p_fr_lb[l]] += -prim.dp_fr_hat[t]
            J[row, idx.sig_p_fr_ub[l]] += -prim.dp_fr_hat[t]
            J[row, idx.sig_q_fr_lb[l]] += -prim.dq_fr_hat[t]
            J[row, idx.sig_q_fr_ub[l]] += -prim.dq_fr_hat[t]
            J[row, idx.sig_p_to_lb[l]] += -prim.dp_to_hat[t]
            J[row, idx.sig_p_to_ub[l]] += -prim.dp_to_hat[t]
            J[row, idx.sig_q_to_lb[l]] += -prim.dq_to_hat[t]
            J[row, idx.sig_q_to_ub[l]] += -prim.dq_to_hat[t]
        end
        J[idx.va[fb], idx.lam_angle_lb[l]] += -1.0
        J[idx.va[tb], idx.lam_angle_lb[l]] += 1.0
        J[idx.va[fb], idx.lam_angle_ub[l]] += -1.0
        J[idx.va[tb], idx.lam_angle_ub[l]] += 1.0

        for t in 1:4
            col = local_idx[t]
            J[idx.nu_p_bal[fb], col] += prim.dp_fr_hat[t]
            J[idx.nu_q_bal[fb], col] += prim.dq_fr_hat[t]
            J[idx.nu_p_bal[tb], col] += prim.dp_to_hat[t]
            J[idx.nu_q_bal[tb], col] += prim.dq_to_hat[t]
            J[idx.lam_thermal_fr[l], col] += vars.lam_thermal_fr[l] *
                                             2 * (prim.p_fr * prim.dp_fr_hat[t] + prim.q_fr * prim.dq_fr_hat[t])
            J[idx.lam_thermal_to[l], col] += vars.lam_thermal_to[l] *
                                             2 * (prim.p_to * prim.dp_to_hat[t] + prim.q_to * prim.dq_to_hat[t])
            J[idx.sig_p_fr_lb[l], col] += vars.sig_p_fr_lb[l] * prim.dp_fr_hat[t]
            J[idx.sig_p_fr_ub[l], col] += -vars.sig_p_fr_ub[l] * prim.dp_fr_hat[t]
            J[idx.sig_q_fr_lb[l], col] += vars.sig_q_fr_lb[l] * prim.dq_fr_hat[t]
            J[idx.sig_q_fr_ub[l], col] += -vars.sig_q_fr_ub[l] * prim.dq_fr_hat[t]
            J[idx.sig_p_to_lb[l], col] += vars.sig_p_to_lb[l] * prim.dp_to_hat[t]
            J[idx.sig_p_to_ub[l], col] += -vars.sig_p_to_ub[l] * prim.dp_to_hat[t]
            J[idx.sig_q_to_lb[l], col] += vars.sig_q_to_lb[l] * prim.dq_to_hat[t]
            J[idx.sig_q_to_ub[l], col] += -vars.sig_q_to_ub[l] * prim.dq_to_hat[t]
        end

        J[idx.lam_angle_lb[l], idx.va[fb]] = vars.lam_angle_lb[l]
        J[idx.lam_angle_lb[l], idx.va[tb]] = -vars.lam_angle_lb[l]
        J[idx.lam_angle_lb[l], idx.lam_angle_lb[l]] = vars.va[fb] - vars.va[tb] - constants.angmin[l]
        J[idx.lam_angle_ub[l], idx.va[fb]] = -vars.lam_angle_ub[l]
        J[idx.lam_angle_ub[l], idx.va[tb]] = vars.lam_angle_ub[l]
        J[idx.lam_angle_ub[l], idx.lam_angle_ub[l]] = constants.angmax[l] - vars.va[fb] + vars.va[tb]

        J[idx.lam_thermal_fr[l], idx.lam_thermal_fr[l]] = prim.p_fr^2 + prim.q_fr^2 - fmax[l]^2
        J[idx.lam_thermal_to[l], idx.lam_thermal_to[l]] = prim.p_to^2 + prim.q_to^2 - fmax[l]^2
    end

    @inbounds for i in 1:n
        J[idx.nu_p_bal[i], idx.vm[i]] += 2 * constants.gs[i] * vars.vm[i]
        J[idx.nu_q_bal[i], idx.vm[i]] += -2 * constants.bs[i] * vars.vm[i]
        J[idx.mu_vm_lb[i], idx.vm[i]] = vars.mu_vm_lb[i]
        J[idx.mu_vm_lb[i], idx.mu_vm_lb[i]] = vars.vm[i] - constants.vmin[i]
        J[idx.mu_vm_ub[i], idx.vm[i]] = -vars.mu_vm_ub[i]
        J[idx.mu_vm_ub[i], idx.mu_vm_ub[i]] = constants.vmax[i] - vars.vm[i]
    end
    @inbounds for (j, ref_bus_idx) in enumerate(constants.ref_bus_keys)
        J[idx.nu_ref_bus[j], idx.va[ref_bus_idx]] = 1.0
    end
    @inbounds for i in 1:k
        J[idx.nu_p_bal[constants.gen_bus[i]], idx.pg[i]] += -1.0
        J[idx.nu_q_bal[constants.gen_bus[i]], idx.qg[i]] += -1.0
        J[idx.rho_pg_lb[i], idx.pg[i]] = vars.rho_pg_lb[i]
        J[idx.rho_pg_lb[i], idx.rho_pg_lb[i]] = vars.pg[i] - constants.pmin[i]
        J[idx.rho_pg_ub[i], idx.pg[i]] = -vars.rho_pg_ub[i]
        J[idx.rho_pg_ub[i], idx.rho_pg_ub[i]] = constants.pmax[i] - vars.pg[i]
        J[idx.rho_qg_lb[i], idx.qg[i]] = vars.rho_qg_lb[i]
        J[idx.rho_qg_lb[i], idx.rho_qg_lb[i]] = vars.qg[i] - constants.qmin[i]
        J[idx.rho_qg_ub[i], idx.qg[i]] = -vars.rho_qg_ub[i]
        J[idx.rho_qg_ub[i], idx.rho_qg_ub[i]] = constants.qmax[i] - vars.qg[i]
    end

    @inbounds for l in 1:m
        prim = _branch_flow_primitives(vars.va, vars.vm, sw, constants, l)
        J[idx.sig_p_fr_lb[l], idx.sig_p_fr_lb[l]] = prim.p_fr + fmax[l]
        J[idx.sig_p_fr_ub[l], idx.sig_p_fr_ub[l]] = fmax[l] - prim.p_fr
        J[idx.sig_q_fr_lb[l], idx.sig_q_fr_lb[l]] = prim.q_fr + fmax[l]
        J[idx.sig_q_fr_ub[l], idx.sig_q_fr_ub[l]] = fmax[l] - prim.q_fr
        J[idx.sig_p_to_lb[l], idx.sig_p_to_lb[l]] = prim.p_to + fmax[l]
        J[idx.sig_p_to_ub[l], idx.sig_p_to_ub[l]] = fmax[l] - prim.p_to
        J[idx.sig_q_to_lb[l], idx.sig_q_to_lb[l]] = prim.q_to + fmax[l]
        J[idx.sig_q_to_ub[l], idx.sig_q_to_ub[l]] = fmax[l] - prim.q_to
    end

    return J
end

# =============================================================================
# KKT Operator
# =============================================================================

"""
    kkt(z::AbstractVector, prob::ACOPFProblem, sw::AbstractVector)

Evaluate the KKT conditions for AC OPF at the given variable vector.

The switching state `sw` is passed separately so the same KKT operator can be
reused for switching sensitivities.

Returns a vector of KKT residuals (should be zero at optimum).

# KKT Conditions (reduced-space formulation)
1. Stationarity w.r.t. va, vm, pg, qg
2. Primal feasibility: power balance, reference bus
3. Complementary slackness for all inequality constraints
"""
function kkt(z::AbstractVector, prob::ACOPFProblem, sw::AbstractVector;
                pd=nothing, qd=nothing, cq=nothing, cl=nothing, fmax=nothing,
                idx=nothing, constants=nothing)
    if isnothing(idx)
        idx = kkt_indices(prob)
    end
    if isnothing(constants)
        constants = _extract_kkt_constants(prob)
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
    cq_vec = isnothing(cq) ? T[ref[:gen][i]["cost"][1] for i in 1:k] : cq
    cl_vec = isnothing(cl) ? T[ref[:gen][i]["cost"][2] for i in 1:k] : cl

    # Pre-allocate KKT residual vector
    K = fill(T(NaN), last(idx.sig_q_to_ub))

    # =========================================================================
    # 1. Stationarity conditions
    # =========================================================================
    _stationarity_residual!(K, vars, prob, sw; cq=cq_vec, cl=cl_vec, fmax=rate_a, constants=constants)

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

    # Use pre-extracted bounds when available to avoid repeated ref lookups
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
Pre-extract all constant data from the problem's ref for efficient analytical
KKT assembly and repeated sensitivity evaluation.
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
# Parameter Jacobian (analytical)
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

@inline function _add_stationarity_sw_flow_terms!(
    Jp::AbstractMatrix, idx::NamedTuple, vars, prim, l::Int,
    coeff_p_fr, coeff_q_fr, coeff_p_to, coeff_q_to)
    fb = prim.fb
    tb = prim.tb

    @inbounds begin
        Jp[idx.va[fb], l] += coeff_p_fr * prim.dp_fr_hat[1] +
                             coeff_q_fr * prim.dq_fr_hat[1] +
                             coeff_p_to * prim.dp_to_hat[1] +
                             coeff_q_to * prim.dq_to_hat[1]
        Jp[idx.va[tb], l] += coeff_p_fr * prim.dp_fr_hat[2] +
                             coeff_q_fr * prim.dq_fr_hat[2] +
                             coeff_p_to * prim.dp_to_hat[2] +
                             coeff_q_to * prim.dq_to_hat[2]

        Jp[idx.vm[fb], l] += coeff_p_fr * prim.dp_fr_hat[3] +
                             coeff_q_fr * prim.dq_fr_hat[3] +
                             coeff_p_to * prim.dp_to_hat[3] +
                             coeff_q_to * prim.dq_to_hat[3]
        Jp[idx.vm[tb], l] += coeff_p_fr * prim.dp_fr_hat[4] +
                             coeff_q_fr * prim.dq_fr_hat[4] +
                             coeff_p_to * prim.dp_to_hat[4] +
                             coeff_q_to * prim.dq_to_hat[4]
    end
    return nothing
end

function _fill_ac_param_jacobian_sw!(Jp::AbstractMatrix, prob::ACOPFProblem,
                                     sol::ACOPFSolution, idx::NamedTuple, constants)
    vars = unflatten_variables(flatten_variables(sol, prob), idx)
    va = vars.va
    vm = vars.vm
    sw = prob.network.sw

    @inbounds for l in 1:prob.network.m
        prim = _branch_flow_primitives(va, vm, sw, constants, l)
        fb = prim.fb
        tb = prim.tb

        coeff_p_fr = -vars.nu_p_bal[fb] - 4 * vars.lam_thermal_fr[l] * prim.p_fr -
                     vars.sig_p_fr_lb[l] - vars.sig_p_fr_ub[l]
        coeff_q_fr = -vars.nu_q_bal[fb] - 4 * vars.lam_thermal_fr[l] * prim.q_fr -
                     vars.sig_q_fr_lb[l] - vars.sig_q_fr_ub[l]
        coeff_p_to = -vars.nu_p_bal[tb] - 4 * vars.lam_thermal_to[l] * prim.p_to -
                     vars.sig_p_to_lb[l] - vars.sig_p_to_ub[l]
        coeff_q_to = -vars.nu_q_bal[tb] - 4 * vars.lam_thermal_to[l] * prim.q_to -
                     vars.sig_q_to_lb[l] - vars.sig_q_to_ub[l]

        _add_stationarity_sw_flow_terms!(
            Jp, idx, vars, prim, l, coeff_p_fr, coeff_q_fr, coeff_p_to, coeff_q_to)

        Jp[idx.nu_p_bal[fb], l] += prim.p_fr_hat
        Jp[idx.nu_p_bal[tb], l] += prim.p_to_hat
        Jp[idx.nu_q_bal[fb], l] += prim.q_fr_hat
        Jp[idx.nu_q_bal[tb], l] += prim.q_to_hat

        Jp[idx.lam_thermal_fr[l], l] = vars.lam_thermal_fr[l] *
                                       2 * (prim.p_fr * prim.p_fr_hat + prim.q_fr * prim.q_fr_hat)
        Jp[idx.lam_thermal_to[l], l] = vars.lam_thermal_to[l] *
                                       2 * (prim.p_to * prim.p_to_hat + prim.q_to * prim.q_to_hat)

        Jp[idx.sig_p_fr_lb[l], l] = vars.sig_p_fr_lb[l] * prim.p_fr_hat
        Jp[idx.sig_p_fr_ub[l], l] = -vars.sig_p_fr_ub[l] * prim.p_fr_hat
        Jp[idx.sig_q_fr_lb[l], l] = vars.sig_q_fr_lb[l] * prim.q_fr_hat
        Jp[idx.sig_q_fr_ub[l], l] = -vars.sig_q_fr_ub[l] * prim.q_fr_hat
        Jp[idx.sig_p_to_lb[l], l] = vars.sig_p_to_lb[l] * prim.p_to_hat
        Jp[idx.sig_p_to_ub[l], l] = -vars.sig_p_to_ub[l] * prim.p_to_hat
        Jp[idx.sig_q_to_lb[l], l] = vars.sig_q_to_lb[l] * prim.q_to_hat
        Jp[idx.sig_q_to_ub[l], l] = -vars.sig_q_to_ub[l] * prim.q_to_hat
    end
    return Jp
end

function _fill_ac_param_jacobian_demand!(Jp::AbstractMatrix, idx::NamedTuple, n::Int, rows)
    @inbounds for i in 1:n
        Jp[rows[i], i] = 1.0
    end
    return Jp
end

function _fill_ac_param_jacobian_cost_linear!(Jp::AbstractMatrix, idx::NamedTuple, k::Int)
    @inbounds for i in 1:k
        Jp[idx.pg[i], i] = 1.0
    end
    return Jp
end

function _fill_ac_param_jacobian_cost_quadratic!(Jp::AbstractMatrix, idx::NamedTuple,
                                                 pg::AbstractVector, k::Int)
    @inbounds for i in 1:k
        Jp[idx.pg[i], i] = 2 * pg[i]
    end
    return Jp
end

function _fill_ac_param_jacobian_fmax!(Jp::AbstractMatrix, idx::NamedTuple,
                                       vars, fmax::AbstractVector, m::Int)
    @inbounds for l in 1:m
        Jp[idx.lam_thermal_fr[l], l] = -2 * vars.lam_thermal_fr[l] * fmax[l]
        Jp[idx.lam_thermal_to[l], l] = -2 * vars.lam_thermal_to[l] * fmax[l]

        Jp[idx.sig_p_fr_lb[l], l] = vars.sig_p_fr_lb[l]
        Jp[idx.sig_p_fr_ub[l], l] = vars.sig_p_fr_ub[l]
        Jp[idx.sig_q_fr_lb[l], l] = vars.sig_q_fr_lb[l]
        Jp[idx.sig_q_fr_ub[l], l] = vars.sig_q_fr_ub[l]
        Jp[idx.sig_p_to_lb[l], l] = vars.sig_p_to_lb[l]
        Jp[idx.sig_p_to_ub[l], l] = vars.sig_p_to_ub[l]
        Jp[idx.sig_q_to_lb[l], l] = vars.sig_q_to_lb[l]
        Jp[idx.sig_q_to_ub[l], l] = vars.sig_q_to_ub[l]
    end
    return Jp
end

"""
    calc_kkt_jacobian_param(prob::ACOPFProblem, sol::ACOPFSolution, param::Symbol)

Compute the analytical parameter Jacobian ∂K/∂param for AC OPF.
Returns a dense matrix of size `(kkt_dims(prob), length(param))`.
"""
function calc_kkt_jacobian_param(prob::ACOPFProblem, sol::ACOPFSolution, param::Symbol)
    haskey(_AC_PARAM_EXTRACT, param) || throw(ArgumentError(
        "Unknown AC OPF parameter: $param. Valid: $(keys(_AC_PARAM_EXTRACT))"))

    idx = kkt_indices(prob)
    z0 = flatten_variables(sol, prob)
    vars = unflatten_variables(z0, idx)
    p0 = _AC_PARAM_EXTRACT[param](prob)
    Jp = zeros(Float64, kkt_dims(prob), length(p0))

    constants = prob.cache.kkt_constants
    if isnothing(constants)
        constants = _extract_kkt_constants(prob)
        prob.cache.kkt_constants = constants
    end

    if param === :sw
        return _fill_ac_param_jacobian_sw!(Jp, prob, sol, idx, constants)
    elseif param === :d
        return _fill_ac_param_jacobian_demand!(Jp, idx, prob.network.n, idx.nu_p_bal)
    elseif param === :qd
        return _fill_ac_param_jacobian_demand!(Jp, idx, prob.network.n, idx.nu_q_bal)
    elseif param === :cl
        return _fill_ac_param_jacobian_cost_linear!(Jp, idx, prob.n_gen)
    elseif param === :cq
        return _fill_ac_param_jacobian_cost_quadratic!(Jp, idx, vars.pg, prob.n_gen)
    elseif param === :fmax
        return _fill_ac_param_jacobian_fmax!(Jp, idx, vars, p0, prob.network.m)
    end

    error("Unhandled AC OPF parameter: $param")
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

Compute a single analytical column of ∂K/∂param without materializing the full
parameter Jacobian.
"""
function _calc_ac_kkt_param_column(prob::ACOPFProblem, sol::ACOPFSolution, param::Symbol, col_idx::Int)
    idx = kkt_indices(prob)
    z0 = flatten_variables(sol, prob)
    vars = unflatten_variables(z0, idx)
    p0 = _AC_PARAM_EXTRACT[param](prob)
    Kcol = zeros(Float64, kkt_dims(prob))

    constants = prob.cache.kkt_constants
    if isnothing(constants)
        constants = _extract_kkt_constants(prob)
        prob.cache.kkt_constants = constants
    end

    if param === :sw
        prim = _branch_flow_primitives(vars.va, vars.vm, prob.network.sw, constants, col_idx)
        fb = prim.fb
        tb = prim.tb

        coeff_p_fr = -vars.nu_p_bal[fb] - 4 * vars.lam_thermal_fr[col_idx] * prim.p_fr -
                     vars.sig_p_fr_lb[col_idx] - vars.sig_p_fr_ub[col_idx]
        coeff_q_fr = -vars.nu_q_bal[fb] - 4 * vars.lam_thermal_fr[col_idx] * prim.q_fr -
                     vars.sig_q_fr_lb[col_idx] - vars.sig_q_fr_ub[col_idx]
        coeff_p_to = -vars.nu_p_bal[tb] - 4 * vars.lam_thermal_to[col_idx] * prim.p_to -
                     vars.sig_p_to_lb[col_idx] - vars.sig_p_to_ub[col_idx]
        coeff_q_to = -vars.nu_q_bal[tb] - 4 * vars.lam_thermal_to[col_idx] * prim.q_to -
                     vars.sig_q_to_lb[col_idx] - vars.sig_q_to_ub[col_idx]

        Kcol[idx.va[fb]] += coeff_p_fr * prim.dp_fr_hat[1] +
                            coeff_q_fr * prim.dq_fr_hat[1] +
                            coeff_p_to * prim.dp_to_hat[1] +
                            coeff_q_to * prim.dq_to_hat[1]
        Kcol[idx.va[tb]] += coeff_p_fr * prim.dp_fr_hat[2] +
                            coeff_q_fr * prim.dq_fr_hat[2] +
                            coeff_p_to * prim.dp_to_hat[2] +
                            coeff_q_to * prim.dq_to_hat[2]
        Kcol[idx.vm[fb]] += coeff_p_fr * prim.dp_fr_hat[3] +
                            coeff_q_fr * prim.dq_fr_hat[3] +
                            coeff_p_to * prim.dp_to_hat[3] +
                            coeff_q_to * prim.dq_to_hat[3]
        Kcol[idx.vm[tb]] += coeff_p_fr * prim.dp_fr_hat[4] +
                            coeff_q_fr * prim.dq_fr_hat[4] +
                            coeff_p_to * prim.dp_to_hat[4] +
                            coeff_q_to * prim.dq_to_hat[4]

        Kcol[idx.nu_p_bal[fb]] += prim.p_fr_hat
        Kcol[idx.nu_p_bal[tb]] += prim.p_to_hat
        Kcol[idx.nu_q_bal[fb]] += prim.q_fr_hat
        Kcol[idx.nu_q_bal[tb]] += prim.q_to_hat

        Kcol[idx.lam_thermal_fr[col_idx]] = vars.lam_thermal_fr[col_idx] *
                                            2 * (prim.p_fr * prim.p_fr_hat + prim.q_fr * prim.q_fr_hat)
        Kcol[idx.lam_thermal_to[col_idx]] = vars.lam_thermal_to[col_idx] *
                                            2 * (prim.p_to * prim.p_to_hat + prim.q_to * prim.q_to_hat)

        Kcol[idx.sig_p_fr_lb[col_idx]] = vars.sig_p_fr_lb[col_idx] * prim.p_fr_hat
        Kcol[idx.sig_p_fr_ub[col_idx]] = -vars.sig_p_fr_ub[col_idx] * prim.p_fr_hat
        Kcol[idx.sig_q_fr_lb[col_idx]] = vars.sig_q_fr_lb[col_idx] * prim.q_fr_hat
        Kcol[idx.sig_q_fr_ub[col_idx]] = -vars.sig_q_fr_ub[col_idx] * prim.q_fr_hat
        Kcol[idx.sig_p_to_lb[col_idx]] = vars.sig_p_to_lb[col_idx] * prim.p_to_hat
        Kcol[idx.sig_p_to_ub[col_idx]] = -vars.sig_p_to_ub[col_idx] * prim.p_to_hat
        Kcol[idx.sig_q_to_lb[col_idx]] = vars.sig_q_to_lb[col_idx] * prim.q_to_hat
        Kcol[idx.sig_q_to_ub[col_idx]] = -vars.sig_q_to_ub[col_idx] * prim.q_to_hat
        return Kcol
    end

    if param === :d
        Kcol[idx.nu_p_bal[col_idx]] = 1.0
    elseif param === :qd
        Kcol[idx.nu_q_bal[col_idx]] = 1.0
    elseif param === :cl
        Kcol[idx.pg[col_idx]] = 1.0
    elseif param === :cq
        Kcol[idx.pg[col_idx]] = 2 * vars.pg[col_idx]
    elseif param === :fmax
        Kcol[idx.lam_thermal_fr[col_idx]] = -2 * vars.lam_thermal_fr[col_idx] * p0[col_idx]
        Kcol[idx.lam_thermal_to[col_idx]] = -2 * vars.lam_thermal_to[col_idx] * p0[col_idx]
        Kcol[idx.sig_p_fr_lb[col_idx]] = vars.sig_p_fr_lb[col_idx]
        Kcol[idx.sig_p_fr_ub[col_idx]] = vars.sig_p_fr_ub[col_idx]
        Kcol[idx.sig_q_fr_lb[col_idx]] = vars.sig_q_fr_lb[col_idx]
        Kcol[idx.sig_q_fr_ub[col_idx]] = vars.sig_q_fr_ub[col_idx]
        Kcol[idx.sig_p_to_lb[col_idx]] = vars.sig_p_to_lb[col_idx]
        Kcol[idx.sig_p_to_ub[col_idx]] = vars.sig_p_to_ub[col_idx]
        Kcol[idx.sig_q_to_lb[col_idx]] = vars.sig_q_to_lb[col_idx]
        Kcol[idx.sig_q_to_ub[col_idx]] = vars.sig_q_to_ub[col_idx]
    else
        error("Unhandled AC OPF parameter: $param")
    end

    return Kcol
end
