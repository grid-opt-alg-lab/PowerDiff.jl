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
_ref_bus_indices(prob::ACOPFProblem) = prob.data.ref_bus_keys

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
    n_ref = length(prob.data.ref_bus_keys)
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
    n_ref = length(prob.data.ref_bus_keys)
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

Flow equations use the polar pi-model: series admittance `g + jb`, line-charging shunts
`g_fr/b_fr/g_to/b_to`, complex tap (`tap`/`shift`), and `tm = tap^2`. These match the AC OPF
flow constraints assembled by the solver backends.

The switching variable sw_l multiplies each flow, so sw_l=0 means the branch
contributes zero flow (open), sw_l=1 means full flow (closed).
"""
function _compute_branch_flows(va, vm, net::ACNetwork, sw; constants)
    m = net.m
    T = promote_type(eltype(va), eltype(vm), eltype(sw))
    p_fr = zeros(T, m)
    q_fr = zeros(T, m)
    p_to = zeros(T, m)
    q_to = zeros(T, m)

    for l in 1:m
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
Return local branch-flow coefficients.

Each tuple describes the unscaled polynomial

    F_hat = a * self_vm^2 + self_vm * other_vm * (b * cos(delta) + c * sin(delta))

For from-side flows, `self_vm`/`other_vm` are `(vm_f, vm_t)`. For to-side
flows, they are `(vm_t, vm_f)`. Multiplying `F_hat` by `sw[l]` gives the
physical branch flow.
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

@inline function _branch_flow_base(coeffs, self_vm, other_vm, cos_delta, sin_delta)
    return coeffs.a * self_vm^2 +
           self_vm * other_vm * (coeffs.b * cos_delta + coeffs.c * sin_delta)
end

@inline function _branch_flow_local_partials(coeffs, self_vm, other_vm,
                                             cos_delta, sin_delta)
    mix = coeffs.b * cos_delta + coeffs.c * sin_delta
    d_delta = self_vm * other_vm * (-coeffs.b * sin_delta + coeffs.c * cos_delta)
    dself = 2 * coeffs.a * self_vm + other_vm * mix
    dother = self_vm * mix
    return d_delta, dself, dother
end

@inline function _branch_flow_local_second_partials(coeffs, self_vm, other_vm,
                                                    cos_delta, sin_delta)
    mix = coeffs.b * cos_delta + coeffs.c * sin_delta
    d_delta_delta = -self_vm * other_vm * mix
    d_delta_self = other_vm * (-coeffs.b * sin_delta + coeffs.c * cos_delta)
    d_delta_other = self_vm * (-coeffs.b * sin_delta + coeffs.c * cos_delta)
    dself_self = 2 * coeffs.a
    dself_other = mix
    return d_delta_delta, d_delta_self, d_delta_other, dself_self, dself_other
end

@inline function _global_hessian_from_fr(d_delta_delta, d_delta_self, d_delta_other,
                                         dself_self, dself_other)
    return (
        va_f_va_f = d_delta_delta,
        va_f_va_t = -d_delta_delta,
        va_f_vm_f = d_delta_self,
        va_f_vm_t = d_delta_other,
        va_t_va_t = d_delta_delta,
        va_t_vm_f = -d_delta_self,
        va_t_vm_t = -d_delta_other,
        vm_f_vm_f = dself_self,
        vm_f_vm_t = dself_other,
        vm_t_vm_t = zero(dself_self),
    )
end

@inline function _global_hessian_from_to(d_delta_delta, d_delta_self, d_delta_other,
                                         dself_self, dself_other)
    return (
        va_f_va_f = d_delta_delta,
        va_f_va_t = -d_delta_delta,
        va_f_vm_f = d_delta_other,
        va_f_vm_t = d_delta_self,
        va_t_va_t = d_delta_delta,
        va_t_vm_f = -d_delta_other,
        va_t_vm_t = -d_delta_self,
        vm_f_vm_f = zero(dself_self),
        vm_f_vm_t = dself_other,
        vm_t_vm_t = dself_self,
    )
end

@inline function _branch_flow_inputs(va, vm, sw, constants, l::Int)
    fb = constants.f_bus[l]
    tb = constants.t_bus[l]
    vf = vm[fb]
    vt = vm[tb]
    delta = va[fb] - va[tb]
    return (
        fb = fb,
        tb = tb,
        vf = vf,
        vt = vt,
        cos_delta = cos(delta),
        sin_delta = sin(delta),
        sw = sw[l],
        coeffs = _branch_flow_coefficients(constants, l),
    )
end

@inline function _branch_flow_values_from_inputs(data)
    coeffs = data.coeffs
    vf = data.vf
    vt = data.vt
    cos_delta = data.cos_delta
    sin_delta = data.sin_delta

    p_fr_hat = _branch_flow_base(coeffs.p_fr, vf, vt, cos_delta, sin_delta)
    q_fr_hat = _branch_flow_base(coeffs.q_fr, vf, vt, cos_delta, sin_delta)
    p_to_hat = _branch_flow_base(coeffs.p_to, vt, vf, cos_delta, sin_delta)
    q_to_hat = _branch_flow_base(coeffs.q_to, vt, vf, cos_delta, sin_delta)

    return (
        fb = data.fb,
        tb = data.tb,
        sw = data.sw,
        p_fr_hat = p_fr_hat,
        q_fr_hat = q_fr_hat,
        p_to_hat = p_to_hat,
        q_to_hat = q_to_hat,
        p_fr = data.sw * p_fr_hat,
        q_fr = data.sw * q_fr_hat,
        p_to = data.sw * p_to_hat,
        q_to = data.sw * q_to_hat,
    )
end

@inline function _branch_flow_gradient_partials(data)
    coeffs = data.coeffs
    vf = data.vf
    vt = data.vt
    cos_delta = data.cos_delta
    sin_delta = data.sin_delta

    dp_fr_delta_hat, dp_fr_dvf_hat, dp_fr_dvt_hat =
        _branch_flow_local_partials(coeffs.p_fr, vf, vt, cos_delta, sin_delta)
    dq_fr_delta_hat, dq_fr_dvf_hat, dq_fr_dvt_hat =
        _branch_flow_local_partials(coeffs.q_fr, vf, vt, cos_delta, sin_delta)
    dp_to_delta_hat, dp_to_dvt_hat, dp_to_dvf_hat =
        _branch_flow_local_partials(coeffs.p_to, vt, vf, cos_delta, sin_delta)
    dq_to_delta_hat, dq_to_dvt_hat, dq_to_dvf_hat =
        _branch_flow_local_partials(coeffs.q_to, vt, vf, cos_delta, sin_delta)

    return (
        dp_fr_hat = (dp_fr_delta_hat, -dp_fr_delta_hat, dp_fr_dvf_hat, dp_fr_dvt_hat),
        dq_fr_hat = (dq_fr_delta_hat, -dq_fr_delta_hat, dq_fr_dvf_hat, dq_fr_dvt_hat),
        dp_to_hat = (dp_to_delta_hat, -dp_to_delta_hat, dp_to_dvf_hat, dp_to_dvt_hat),
        dq_to_hat = (dq_to_delta_hat, -dq_to_delta_hat, dq_to_dvf_hat, dq_to_dvt_hat),
    )
end

"""Return branch-flow values only."""
function _branch_flow_values(va, vm, sw, constants, l::Int)
    return _branch_flow_values_from_inputs(_branch_flow_inputs(va, vm, sw, constants, l))
end

"""Return branch-flow values and first derivatives with respect to local state."""
function _branch_flow_gradients(va, vm, sw, constants, l::Int)
    data = _branch_flow_inputs(va, vm, sw, constants, l)
    vals = _branch_flow_values_from_inputs(data)
    grads = _branch_flow_gradient_partials(data)
    return (; vals..., grads...)
end

"""Return branch-flow values, first derivatives, and Hessian primitives."""
function _branch_flow_hessian_primitives(va, vm, sw, constants, l::Int)
    data = _branch_flow_inputs(va, vm, sw, constants, l)
    vals = _branch_flow_values_from_inputs(data)
    grads = _branch_flow_gradient_partials(data)
    coeffs = data.coeffs
    vf = data.vf
    vt = data.vt
    cos_delta = data.cos_delta
    sin_delta = data.sin_delta

    d2p_fr = _global_hessian_from_fr(
        _branch_flow_local_second_partials(coeffs.p_fr, vf, vt, cos_delta, sin_delta)...)
    d2q_fr = _global_hessian_from_fr(
        _branch_flow_local_second_partials(coeffs.q_fr, vf, vt, cos_delta, sin_delta)...)
    d2p_to = _global_hessian_from_to(
        _branch_flow_local_second_partials(coeffs.p_to, vt, vf, cos_delta, sin_delta)...)
    d2q_to = _global_hessian_from_to(
        _branch_flow_local_second_partials(coeffs.q_to, vt, vf, cos_delta, sin_delta)...)

    return (;
        vals...,
        grads...,
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
                                  net::ACNetwork, prob::ACOPFProblem;
                                  pd=nothing, qd=nothing, constants)
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
        fb = constants.f_bus[l]
        tb = constants.t_bus[l]
        p_flow_sum[fb] += p_fr[l]
        p_flow_sum[tb] += p_to[l]
        q_flow_sum[fb] += q_fr[l]
        q_flow_sum[tb] += q_to[l]
    end

    # Sum generation at each bus
    pg_sum = zeros(T, n)
    qg_sum = zeros(T, n)
    for i in 1:prob.n_gen
        bus_idx = constants.gen_bus[i]
        pg_sum[bus_idx] += pg[i]
        qg_sum[bus_idx] += qg[i]
    end

    for i in 1:n
        gs_i = constants.gs[i]
        bs_i = constants.bs[i]
        pd_i = isnothing(pd) ? constants.pd[i] : pd[i]
        qd_i = isnothing(qd) ? constants.qd[i] : qd[i]

        K_p_bal[i] = p_flow_sum[i] + gs_i * vm[i]^2 - pg_sum[i] + pd_i
        K_q_bal[i] = q_flow_sum[i] - bs_i * vm[i]^2 - qg_sum[i] + qd_i
    end

    return K_p_bal, K_q_bal
end

@inline function _add_sparse_entry!(I::Vector{Int}, J::Vector{Int}, V::Vector{Float64},
                                    row::Int, col::Int, val)
    iszero(val) && return nothing
    push!(I, row)
    push!(J, col)
    push!(V, Float64(val))
    return nothing
end

@inline function _add_symmetric_local_hessian_entries!(
    I::Vector{Int}, J::Vector{Int}, V::Vector{Float64},
    rows::NTuple{4,Int}, H, coeff, outer, grad::NTuple{4})
    i1, i2, i3, i4 = rows
    g1, g2, g3, g4 = grad

    v11 = coeff * H.va_f_va_f + outer * g1 * g1
    v12 = coeff * H.va_f_va_t + outer * g1 * g2
    v13 = coeff * H.va_f_vm_f + outer * g1 * g3
    v14 = coeff * H.va_f_vm_t + outer * g1 * g4
    v22 = coeff * H.va_t_va_t + outer * g2 * g2
    v23 = coeff * H.va_t_vm_f + outer * g2 * g3
    v24 = coeff * H.va_t_vm_t + outer * g2 * g4
    v33 = coeff * H.vm_f_vm_f + outer * g3 * g3
    v34 = coeff * H.vm_f_vm_t + outer * g3 * g4
    v44 = coeff * H.vm_t_vm_t + outer * g4 * g4

    _add_sparse_entry!(I, J, V, i1, i1, v11)
    _add_sparse_entry!(I, J, V, i1, i2, v12)
    _add_sparse_entry!(I, J, V, i2, i1, v12)
    _add_sparse_entry!(I, J, V, i1, i3, v13)
    _add_sparse_entry!(I, J, V, i3, i1, v13)
    _add_sparse_entry!(I, J, V, i1, i4, v14)
    _add_sparse_entry!(I, J, V, i4, i1, v14)
    _add_sparse_entry!(I, J, V, i2, i2, v22)
    _add_sparse_entry!(I, J, V, i2, i3, v23)
    _add_sparse_entry!(I, J, V, i3, i2, v23)
    _add_sparse_entry!(I, J, V, i2, i4, v24)
    _add_sparse_entry!(I, J, V, i4, i2, v24)
    _add_sparse_entry!(I, J, V, i3, i3, v33)
    _add_sparse_entry!(I, J, V, i3, i4, v34)
    _add_sparse_entry!(I, J, V, i4, i3, v34)
    _add_sparse_entry!(I, J, V, i4, i4, v44)
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
        prim = _branch_flow_gradients(vars.va, vars.vm, sw, constants, l)
        fb = prim.fb
        tb = prim.tb
        sw_l = prim.sw

        coeff_p_fr = -vars.nu_p_bal[fb] - 2 * vars.lam_thermal_fr[l] * prim.p_fr -
                     vars.sig_p_fr_lb[l] - vars.sig_p_fr_ub[l]
        coeff_q_fr = -vars.nu_q_bal[fb] - 2 * vars.lam_thermal_fr[l] * prim.q_fr -
                     vars.sig_q_fr_lb[l] - vars.sig_q_fr_ub[l]
        coeff_p_to = -vars.nu_p_bal[tb] - 2 * vars.lam_thermal_to[l] * prim.p_to -
                     vars.sig_p_to_lb[l] - vars.sig_p_to_ub[l]
        coeff_q_to = -vars.nu_q_bal[tb] - 2 * vars.lam_thermal_to[l] * prim.q_to -
                     vars.sig_q_to_lb[l] - vars.sig_q_to_ub[l]

        K[idx.va[fb]] += sw_l * (coeff_p_fr * prim.dp_fr_hat[1] + coeff_q_fr * prim.dq_fr_hat[1] +
                                 coeff_p_to * prim.dp_to_hat[1] + coeff_q_to * prim.dq_to_hat[1]) -
                         prim.sw * (vars.lam_angle_lb[l] + vars.lam_angle_ub[l])
        K[idx.va[tb]] += sw_l * (coeff_p_fr * prim.dp_fr_hat[2] + coeff_q_fr * prim.dq_fr_hat[2] +
                                 coeff_p_to * prim.dp_to_hat[2] + coeff_q_to * prim.dq_to_hat[2]) +
                         prim.sw * (vars.lam_angle_lb[l] + vars.lam_angle_ub[l])
        K[idx.vm[fb]] += sw_l * (coeff_p_fr * prim.dp_fr_hat[3] + coeff_q_fr * prim.dq_fr_hat[3] +
                                 coeff_p_to * prim.dp_to_hat[3] + coeff_q_to * prim.dq_to_hat[3])
        K[idx.vm[tb]] += sw_l * (coeff_p_fr * prim.dp_fr_hat[4] + coeff_q_fr * prim.dq_fr_hat[4] +
                                 coeff_p_to * prim.dp_to_hat[4] + coeff_q_to * prim.dq_to_hat[4])
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
    fmax = _extract_branch_fmax(prob)
    sw = prob.network.sw
    vars = unflatten_variables(flatten_variables(sol, prob), idx)
    n, m, k = prob.network.n, prob.network.m, prob.n_gen

    dim = kkt_dims(prob)
    row_idxs = Int[]
    col_idxs = Int[]
    vals = Float64[]
    nnz_hint = 32 * n + 180 * m + 12 * k + 2 * length(constants.ref_bus_keys)
    sizehint!(row_idxs, nnz_hint)
    sizehint!(col_idxs, nnz_hint)
    sizehint!(vals, nnz_hint)

    @inbounds for i in 1:k
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.pg[i], idx.pg[i], 2 * cq[i])
    end
    @inbounds for i in 1:n
        _add_sparse_entry!(
            row_idxs, col_idxs, vals, idx.vm[i], idx.vm[i],
            -2 * vars.nu_p_bal[i] * constants.gs[i] +
            2 * vars.nu_q_bal[i] * constants.bs[i])
    end

    @inbounds for i in 1:k
        bus = constants.gen_bus[i]
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.pg[i], idx.nu_p_bal[bus], 1.0)
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.pg[i], idx.rho_pg_lb[i], -1.0)
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.pg[i], idx.rho_pg_ub[i], -1.0)
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.qg[i], idx.nu_q_bal[bus], 1.0)
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.qg[i], idx.rho_qg_lb[i], -1.0)
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.qg[i], idx.rho_qg_ub[i], -1.0)
    end
    @inbounds for i in 1:n
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.vm[i], idx.mu_vm_lb[i], -1.0)
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.vm[i], idx.mu_vm_ub[i], -1.0)
    end
    @inbounds for (j, ref_bus_idx) in enumerate(constants.ref_bus_keys)
        _add_sparse_entry!(
            row_idxs, col_idxs, vals, idx.va[ref_bus_idx], idx.nu_ref_bus[j], -1.0)
    end

    @inbounds for l in 1:m
        prim = _branch_flow_hessian_primitives(vars.va, vars.vm, sw, constants, l)
        fb = prim.fb
        tb = prim.tb
        sw_l = prim.sw
        local_idx = (idx.va[fb], idx.va[tb], idx.vm[fb], idx.vm[tb])

        coeff_p_fr = -vars.nu_p_bal[fb] - 2 * vars.lam_thermal_fr[l] * prim.p_fr -
                     vars.sig_p_fr_lb[l] - vars.sig_p_fr_ub[l]
        coeff_q_fr = -vars.nu_q_bal[fb] - 2 * vars.lam_thermal_fr[l] * prim.q_fr -
                     vars.sig_q_fr_lb[l] - vars.sig_q_fr_ub[l]
        coeff_p_to = -vars.nu_p_bal[tb] - 2 * vars.lam_thermal_to[l] * prim.p_to -
                     vars.sig_p_to_lb[l] - vars.sig_p_to_ub[l]
        coeff_q_to = -vars.nu_q_bal[tb] - 2 * vars.lam_thermal_to[l] * prim.q_to -
                     vars.sig_q_to_lb[l] - vars.sig_q_to_ub[l]

        _add_symmetric_local_hessian_entries!(
            row_idxs, col_idxs, vals, local_idx, prim.d2p_fr_hat, coeff_p_fr * sw_l,
            -2 * vars.lam_thermal_fr[l] * sw_l^2, prim.dp_fr_hat)
        _add_symmetric_local_hessian_entries!(
            row_idxs, col_idxs, vals, local_idx, prim.d2q_fr_hat, coeff_q_fr * sw_l,
            -2 * vars.lam_thermal_fr[l] * sw_l^2, prim.dq_fr_hat)
        _add_symmetric_local_hessian_entries!(
            row_idxs, col_idxs, vals, local_idx, prim.d2p_to_hat, coeff_p_to * sw_l,
            -2 * vars.lam_thermal_to[l] * sw_l^2, prim.dp_to_hat)
        _add_symmetric_local_hessian_entries!(
            row_idxs, col_idxs, vals, local_idx, prim.d2q_to_hat, coeff_q_to * sw_l,
            -2 * vars.lam_thermal_to[l] * sw_l^2, prim.dq_to_hat)

        for t in 1:4
            row = local_idx[t]
            _add_sparse_entry!(row_idxs, col_idxs, vals, row, idx.nu_p_bal[fb],
                               -sw_l * prim.dp_fr_hat[t])
            _add_sparse_entry!(row_idxs, col_idxs, vals, row, idx.nu_q_bal[fb],
                               -sw_l * prim.dq_fr_hat[t])
            _add_sparse_entry!(row_idxs, col_idxs, vals, row, idx.nu_p_bal[tb],
                               -sw_l * prim.dp_to_hat[t])
            _add_sparse_entry!(row_idxs, col_idxs, vals, row, idx.nu_q_bal[tb],
                               -sw_l * prim.dq_to_hat[t])
            _add_sparse_entry!(
                row_idxs, col_idxs, vals, row, idx.lam_thermal_fr[l],
                -2 * sw_l * (prim.p_fr * prim.dp_fr_hat[t] +
                              prim.q_fr * prim.dq_fr_hat[t]))
            _add_sparse_entry!(
                row_idxs, col_idxs, vals, row, idx.lam_thermal_to[l],
                -2 * sw_l * (prim.p_to * prim.dp_to_hat[t] +
                              prim.q_to * prim.dq_to_hat[t]))
            _add_sparse_entry!(row_idxs, col_idxs, vals, row, idx.sig_p_fr_lb[l],
                               -sw_l * prim.dp_fr_hat[t])
            _add_sparse_entry!(row_idxs, col_idxs, vals, row, idx.sig_p_fr_ub[l],
                               -sw_l * prim.dp_fr_hat[t])
            _add_sparse_entry!(row_idxs, col_idxs, vals, row, idx.sig_q_fr_lb[l],
                               -sw_l * prim.dq_fr_hat[t])
            _add_sparse_entry!(row_idxs, col_idxs, vals, row, idx.sig_q_fr_ub[l],
                               -sw_l * prim.dq_fr_hat[t])
            _add_sparse_entry!(row_idxs, col_idxs, vals, row, idx.sig_p_to_lb[l],
                               -sw_l * prim.dp_to_hat[t])
            _add_sparse_entry!(row_idxs, col_idxs, vals, row, idx.sig_p_to_ub[l],
                               -sw_l * prim.dp_to_hat[t])
            _add_sparse_entry!(row_idxs, col_idxs, vals, row, idx.sig_q_to_lb[l],
                               -sw_l * prim.dq_to_hat[t])
            _add_sparse_entry!(row_idxs, col_idxs, vals, row, idx.sig_q_to_ub[l],
                               -sw_l * prim.dq_to_hat[t])
        end
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.va[fb], idx.lam_angle_lb[l],
                           -sw[l])
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.va[tb], idx.lam_angle_lb[l],
                           sw[l])
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.va[fb], idx.lam_angle_ub[l],
                           -sw[l])
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.va[tb], idx.lam_angle_ub[l],
                           sw[l])

        for t in 1:4
            col = local_idx[t]
            _add_sparse_entry!(row_idxs, col_idxs, vals, idx.nu_p_bal[fb], col,
                               sw_l * prim.dp_fr_hat[t])
            _add_sparse_entry!(row_idxs, col_idxs, vals, idx.nu_q_bal[fb], col,
                               sw_l * prim.dq_fr_hat[t])
            _add_sparse_entry!(row_idxs, col_idxs, vals, idx.nu_p_bal[tb], col,
                               sw_l * prim.dp_to_hat[t])
            _add_sparse_entry!(row_idxs, col_idxs, vals, idx.nu_q_bal[tb], col,
                               sw_l * prim.dq_to_hat[t])
            _add_sparse_entry!(
                row_idxs, col_idxs, vals, idx.lam_thermal_fr[l], col,
                vars.lam_thermal_fr[l] * 2 * sw_l *
                (prim.p_fr * prim.dp_fr_hat[t] + prim.q_fr * prim.dq_fr_hat[t]))
            _add_sparse_entry!(
                row_idxs, col_idxs, vals, idx.lam_thermal_to[l], col,
                vars.lam_thermal_to[l] * 2 * sw_l *
                (prim.p_to * prim.dp_to_hat[t] + prim.q_to * prim.dq_to_hat[t]))
            _add_sparse_entry!(row_idxs, col_idxs, vals, idx.sig_p_fr_lb[l], col,
                               vars.sig_p_fr_lb[l] * sw_l * prim.dp_fr_hat[t])
            _add_sparse_entry!(row_idxs, col_idxs, vals, idx.sig_p_fr_ub[l], col,
                               -vars.sig_p_fr_ub[l] * sw_l * prim.dp_fr_hat[t])
            _add_sparse_entry!(row_idxs, col_idxs, vals, idx.sig_q_fr_lb[l], col,
                               vars.sig_q_fr_lb[l] * sw_l * prim.dq_fr_hat[t])
            _add_sparse_entry!(row_idxs, col_idxs, vals, idx.sig_q_fr_ub[l], col,
                               -vars.sig_q_fr_ub[l] * sw_l * prim.dq_fr_hat[t])
            _add_sparse_entry!(row_idxs, col_idxs, vals, idx.sig_p_to_lb[l], col,
                               vars.sig_p_to_lb[l] * sw_l * prim.dp_to_hat[t])
            _add_sparse_entry!(row_idxs, col_idxs, vals, idx.sig_p_to_ub[l], col,
                               -vars.sig_p_to_ub[l] * sw_l * prim.dp_to_hat[t])
            _add_sparse_entry!(row_idxs, col_idxs, vals, idx.sig_q_to_lb[l], col,
                               vars.sig_q_to_lb[l] * sw_l * prim.dq_to_hat[t])
            _add_sparse_entry!(row_idxs, col_idxs, vals, idx.sig_q_to_ub[l], col,
                               -vars.sig_q_to_ub[l] * sw_l * prim.dq_to_hat[t])
        end

        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.lam_angle_lb[l], idx.va[fb],
                           vars.lam_angle_lb[l] * sw[l])
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.lam_angle_lb[l], idx.va[tb],
                           -vars.lam_angle_lb[l] * sw[l])
        _add_sparse_entry!(
            row_idxs, col_idxs, vals, idx.lam_angle_lb[l], idx.lam_angle_lb[l],
            sw[l] * (vars.va[fb] - vars.va[tb] - constants.angmin[l]))
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.lam_angle_ub[l], idx.va[fb],
                           -vars.lam_angle_ub[l] * sw[l])
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.lam_angle_ub[l], idx.va[tb],
                           vars.lam_angle_ub[l] * sw[l])
        _add_sparse_entry!(
            row_idxs, col_idxs, vals, idx.lam_angle_ub[l], idx.lam_angle_ub[l],
            sw[l] * (constants.angmax[l] - vars.va[fb] + vars.va[tb]))

        _add_sparse_entry!(
            row_idxs, col_idxs, vals, idx.lam_thermal_fr[l], idx.lam_thermal_fr[l],
            prim.p_fr^2 + prim.q_fr^2 - fmax[l]^2)
        _add_sparse_entry!(
            row_idxs, col_idxs, vals, idx.lam_thermal_to[l], idx.lam_thermal_to[l],
            prim.p_to^2 + prim.q_to^2 - fmax[l]^2)
    end

    @inbounds for i in 1:n
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.nu_p_bal[i], idx.vm[i],
                           2 * constants.gs[i] * vars.vm[i])
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.nu_q_bal[i], idx.vm[i],
                           -2 * constants.bs[i] * vars.vm[i])
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.mu_vm_lb[i], idx.vm[i],
                           vars.mu_vm_lb[i])
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.mu_vm_lb[i], idx.mu_vm_lb[i],
                           vars.vm[i] - constants.vmin[i])
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.mu_vm_ub[i], idx.vm[i],
                           -vars.mu_vm_ub[i])
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.mu_vm_ub[i], idx.mu_vm_ub[i],
                           constants.vmax[i] - vars.vm[i])
    end
    @inbounds for (j, ref_bus_idx) in enumerate(constants.ref_bus_keys)
        _add_sparse_entry!(
            row_idxs, col_idxs, vals, idx.nu_ref_bus[j], idx.va[ref_bus_idx], 1.0)
    end
    @inbounds for i in 1:k
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.nu_p_bal[constants.gen_bus[i]],
                           idx.pg[i], -1.0)
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.nu_q_bal[constants.gen_bus[i]],
                           idx.qg[i], -1.0)
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.rho_pg_lb[i], idx.pg[i],
                           vars.rho_pg_lb[i])
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.rho_pg_lb[i], idx.rho_pg_lb[i],
                           vars.pg[i] - constants.pmin[i])
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.rho_pg_ub[i], idx.pg[i],
                           -vars.rho_pg_ub[i])
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.rho_pg_ub[i], idx.rho_pg_ub[i],
                           constants.pmax[i] - vars.pg[i])
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.rho_qg_lb[i], idx.qg[i],
                           vars.rho_qg_lb[i])
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.rho_qg_lb[i], idx.rho_qg_lb[i],
                           vars.qg[i] - constants.qmin[i])
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.rho_qg_ub[i], idx.qg[i],
                           -vars.rho_qg_ub[i])
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.rho_qg_ub[i], idx.rho_qg_ub[i],
                           constants.qmax[i] - vars.qg[i])
    end

    @inbounds for l in 1:m
        prim = _branch_flow_values(vars.va, vars.vm, sw, constants, l)
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.sig_p_fr_lb[l],
                           idx.sig_p_fr_lb[l], prim.p_fr + fmax[l])
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.sig_p_fr_ub[l],
                           idx.sig_p_fr_ub[l], fmax[l] - prim.p_fr)
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.sig_q_fr_lb[l],
                           idx.sig_q_fr_lb[l], prim.q_fr + fmax[l])
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.sig_q_fr_ub[l],
                           idx.sig_q_fr_ub[l], fmax[l] - prim.q_fr)
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.sig_p_to_lb[l],
                           idx.sig_p_to_lb[l], prim.p_to + fmax[l])
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.sig_p_to_ub[l],
                           idx.sig_p_to_ub[l], fmax[l] - prim.p_to)
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.sig_q_to_lb[l],
                           idx.sig_q_to_lb[l], prim.q_to + fmax[l])
        _add_sparse_entry!(row_idxs, col_idxs, vals, idx.sig_q_to_ub[l],
                           idx.sig_q_to_ub[l], fmax[l] - prim.q_to)
    end

    return sparse(row_idxs, col_idxs, vals, dim, dim)
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
    n, m, k = net.n, net.m, prob.n_gen

    va, vm = vars.va, vars.vm
    pg, qg = vars.pg, vars.qg

    _et(x) = isnothing(x) ? Float64 : eltype(x)
    T = promote_type(eltype(z), eltype(sw), _et(pd), _et(qd), _et(cq), _et(cl), _et(fmax))

    # Compute branch flows as functions of voltages
    p_fr, q_fr, p_to, q_to = _compute_branch_flows(va, vm, net, sw; constants=constants)

    rate_a = isnothing(fmax) ? T.(constants.fmax) : fmax
    cq_vec = isnothing(cq) ? T.(constants.cq) : cq
    cl_vec = isnothing(cl) ? T.(constants.cl) : cl

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
                                                 net, prob; pd=pd, qd=qd, constants=constants)
    K[idx.nu_p_bal] = K_p_bal
    K[idx.nu_q_bal] = K_q_bal

    # Reference bus: va[ref_bus] == 0
    rbk = constants.ref_bus_keys
    for (j, ref_bus_idx) in enumerate(rbk)
        K[idx.nu_ref_bus[j]] = va[ref_bus_idx]
    end

    # =========================================================================
    # 3. Complementary slackness conditions (vectorized)
    # =========================================================================
    # Lower bounds: L -= λ*(x - lb),  CS = λ*(x - lb) = 0  (same residual sign)
    # Upper bounds: L -= λ*(x - ub),  CS = λ*(ub - x) = 0  (negated residual)
    # Both are valid; the sign flip cancels in implicit differentiation.

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

    # Thermal limits
    K[idx.lam_thermal_fr] .= vars.lam_thermal_fr .* (p_fr.^2 .+ q_fr.^2 .- rate_a.^2)
    K[idx.lam_thermal_to] .= vars.lam_thermal_to .* (p_to.^2 .+ q_to.^2 .- rate_a.^2)

    # Angle difference limits
    K[idx.lam_angle_lb] .= vars.lam_angle_lb .* sw .* (va[f_bus_idx] .- va[t_bus_idx] .- angmin)
    K[idx.lam_angle_ub] .= vars.lam_angle_ub .* sw .* (angmax .- va[f_bus_idx] .+ va[t_bus_idx])

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

@inline function _require_kkt_constants(prob::ACOPFProblem)
    constants = prob.cache.kkt_constants
    isnothing(constants) && error(
        "ACOPFProblem cache invariant violated: kkt_constants are missing. " *
        "ACOPFProblem constructors and rebuilds should populate prob.cache.kkt_constants."
    )
    return constants
end

"""Extract per-bus aggregated load values for a given key ("pd" or "qd")."""
function _extract_bus_load(prob::ACOPFProblem, key::String)
    constants = _require_kkt_constants(prob)
    return key == "pd" ? copy(constants.pd) : copy(constants.qd)
end

"""
Extract per-generator cost coefficient at a given index (1=quadratic, 2=linear).

MATPOWER parsing standardizes costs before this point, so the analytical path
assumes finite numeric coefficients.
"""
function _extract_gen_cost(prob::ACOPFProblem, cost_idx::Int)
    constants = _require_kkt_constants(prob)
    return cost_idx == 1 ? copy(constants.cq) : copy(constants.cl)
end

_extract_bus_pd(prob::ACOPFProblem) = _extract_bus_load(prob, "pd")
_extract_bus_qd(prob::ACOPFProblem) = _extract_bus_load(prob, "qd")
_extract_gen_cq(prob::ACOPFProblem) = _extract_gen_cost(prob, 1)
_extract_gen_cl(prob::ACOPFProblem) = _extract_gen_cost(prob, 2)

"""
Extract per-branch flow limits (`rate_a`) from cached constants.

MATPOWER parsing normalizes thermal limits before this point, so the analytical
path assumes finite numeric limits.
"""
function _extract_branch_fmax(prob::ACOPFProblem)
    return copy(_require_kkt_constants(prob).fmax)
end

"""
Pre-extract all constant data from the problem for efficient analytical
KKT assembly and repeated sensitivity evaluation.
"""
function _extract_kkt_constants(prob::ACOPFProblem)
    return prob.data.constants
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

@inline function _ac_sw_param_column_terms(idx::NamedTuple, vars, constants, prim, l::Int)
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
    angle_dual = vars.lam_angle_lb[l] + vars.lam_angle_ub[l]

    d_va_f = coeff_p_fr * prim.dp_fr_hat[1] + coeff_q_fr * prim.dq_fr_hat[1] +
             coeff_p_to * prim.dp_to_hat[1] + coeff_q_to * prim.dq_to_hat[1] -
             angle_dual
    d_va_t = coeff_p_fr * prim.dp_fr_hat[2] + coeff_q_fr * prim.dq_fr_hat[2] +
             coeff_p_to * prim.dp_to_hat[2] + coeff_q_to * prim.dq_to_hat[2] +
             angle_dual
    d_vm_f = coeff_p_fr * prim.dp_fr_hat[3] + coeff_q_fr * prim.dq_fr_hat[3] +
             coeff_p_to * prim.dp_to_hat[3] + coeff_q_to * prim.dq_to_hat[3]
    d_vm_t = coeff_p_fr * prim.dp_fr_hat[4] + coeff_q_fr * prim.dq_fr_hat[4] +
             coeff_p_to * prim.dp_to_hat[4] + coeff_q_to * prim.dq_to_hat[4]

    return (
        rows = (
            idx.va[fb], idx.va[tb], idx.vm[fb], idx.vm[tb],
            idx.nu_p_bal[fb], idx.nu_p_bal[tb], idx.nu_q_bal[fb], idx.nu_q_bal[tb],
            idx.lam_thermal_fr[l], idx.lam_thermal_to[l],
            idx.sig_p_fr_lb[l], idx.sig_p_fr_ub[l],
            idx.sig_q_fr_lb[l], idx.sig_q_fr_ub[l],
            idx.sig_p_to_lb[l], idx.sig_p_to_ub[l],
            idx.sig_q_to_lb[l], idx.sig_q_to_ub[l],
            idx.lam_angle_lb[l], idx.lam_angle_ub[l],
        ),
        vals = (
            d_va_f, d_va_t, d_vm_f, d_vm_t,
            prim.p_fr_hat, prim.p_to_hat, prim.q_fr_hat, prim.q_to_hat,
            vars.lam_thermal_fr[l] * 2 * (prim.p_fr * prim.p_fr_hat +
                                           prim.q_fr * prim.q_fr_hat),
            vars.lam_thermal_to[l] * 2 * (prim.p_to * prim.p_to_hat +
                                           prim.q_to * prim.q_to_hat),
            vars.sig_p_fr_lb[l] * prim.p_fr_hat,
            -vars.sig_p_fr_ub[l] * prim.p_fr_hat,
            vars.sig_q_fr_lb[l] * prim.q_fr_hat,
            -vars.sig_q_fr_ub[l] * prim.q_fr_hat,
            vars.sig_p_to_lb[l] * prim.p_to_hat,
            -vars.sig_p_to_ub[l] * prim.p_to_hat,
            vars.sig_q_to_lb[l] * prim.q_to_hat,
            -vars.sig_q_to_ub[l] * prim.q_to_hat,
            vars.lam_angle_lb[l] * (vars.va[fb] - vars.va[tb] - constants.angmin[l]),
            vars.lam_angle_ub[l] * (constants.angmax[l] - vars.va[fb] + vars.va[tb]),
        ),
    )
end

@inline function _add_ac_sw_param_column!(out::AbstractVector, idx::NamedTuple, vars,
                                          constants, prim, l::Int, scale=1.0)
    terms = _ac_sw_param_column_terms(idx, vars, constants, prim, l)
    @inbounds for q in eachindex(terms.rows)
        out[terms.rows[q]] += scale * terms.vals[q]
    end
    return out
end

@inline function _dot_ac_sw_param_column(u::AbstractVector, idx::NamedTuple, vars,
                                         constants, prim, l::Int)
    terms = _ac_sw_param_column_terms(idx, vars, constants, prim, l)
    acc = zero(eltype(u))
    @inbounds for q in eachindex(terms.rows)
        acc += u[terms.rows[q]] * terms.vals[q]
    end
    return acc
end

function _fill_ac_param_jacobian_sw!(Jp::AbstractMatrix, prob::ACOPFProblem,
                                     sol::ACOPFSolution, idx::NamedTuple, constants)
    vars = sol
    va = sol.va
    vm = sol.vm
    sw = prob.network.sw

    @inbounds for l in 1:prob.network.m
        prim = _branch_flow_gradients(va, vm, sw, constants, l)
        _add_ac_sw_param_column!(@view(Jp[:, l]), idx, vars, constants, prim, l)
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
    p0 = _AC_PARAM_EXTRACT[param](prob)
    Jp = zeros(Float64, kkt_dims(prob), length(p0))

    constants = _require_kkt_constants(prob)

    if param === :sw
        return _fill_ac_param_jacobian_sw!(Jp, prob, sol, idx, constants)
    elseif param === :d
        return _fill_ac_param_jacobian_demand!(Jp, idx, prob.network.n, idx.nu_p_bal)
    elseif param === :qd
        return _fill_ac_param_jacobian_demand!(Jp, idx, prob.network.n, idx.nu_q_bal)
    elseif param === :cl
        return _fill_ac_param_jacobian_cost_linear!(Jp, idx, prob.n_gen)
    elseif param === :cq
        return _fill_ac_param_jacobian_cost_quadratic!(Jp, idx, sol.pg, prob.n_gen)
    elseif param === :fmax
        return _fill_ac_param_jacobian_fmax!(Jp, idx, sol, p0, prob.network.m)
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
    vars = sol
    p0 = _AC_PARAM_EXTRACT[param](prob)
    Kcol = zeros(Float64, kkt_dims(prob))

    constants = _require_kkt_constants(prob)

    if param === :sw
        prim = _branch_flow_gradients(vars.va, vars.vm, prob.network.sw, constants, col_idx)
        _add_ac_sw_param_column!(Kcol, idx, vars, constants, prim, col_idx)
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
