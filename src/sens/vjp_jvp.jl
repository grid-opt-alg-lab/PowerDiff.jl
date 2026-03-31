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
# Efficient VJP/JVP Through KKT Systems
# =============================================================================
#
# Computes vector-Jacobian products (VJP) and Jacobian-vector products (JVP)
# without materializing the full sensitivity matrix.
#
# Math (implicit differentiation):
#   S = sign · E_op · (-(dK/dz)⁻¹ · dK/dp)
#
#   VJP: Sᵀ · adj = -(dK/dp)ᵀ · (dK/dz)⁻ᵀ · E_opᵀ · sign · adj
#   JVP: S · tang = sign · E_op · (-(dK/dz)⁻¹ · dK/dp · tang)

# =============================================================================
# Internal Dict ↔ Vector Helpers
# =============================================================================

"""Standalone dict → vector conversion (no Sensitivity object needed)."""
function _dict_to_vec(d::AbstractDict{Int,<:Number}, id_to_idx::Dict{Int,Int}, n::Int)
    v = zeros(n)
    for (id, val) in d
        idx = get(id_to_idx, id, nothing)
        isnothing(idx) && throw(ArgumentError("unknown element ID $id"))
        v[idx] = val
    end
    return v
end

"""Standalone vector → dict conversion (no Sensitivity object needed)."""
_vec_to_dict(v::AbstractVector, idx_to_id::Vector{Int}) =
    Dict{Int,Float64}(idx_to_id[i] => v[i] for i in eachindex(v))

# =============================================================================
# DC OPF: Inline Parameter VJP  (out = -(dK/dp)ᵀ · u)
# =============================================================================
#
# Each parameter's dK/dp has O(1) nonzeros per column. Instead of building
# the full sparse matrix, we compute the dot product inline per column.

function _dc_param_vjp!(out::AbstractVector, prob::DCOPFProblem,
                        sol::DCOPFSolution, param::Symbol,
                        u::AbstractVector, idx::NamedTuple)
    if param === :d
        _dc_demand_vjp!(out, sol, u, idx, prob.network.n)
    elseif param === :cl
        _dc_cost_linear_vjp!(out, u, idx, prob.network.k)
    elseif param === :cq
        _dc_cost_quadratic_vjp!(out, sol, u, idx, prob.network.k)
    elseif param === :fmax
        _dc_flowlimit_vjp!(out, sol, u, idx, prob.network.m)
    elseif param === :sw
        _dc_topo_vjp!(out, prob, sol, u, idx, prob.network.b)
    else  # :b
        _dc_topo_vjp!(out, prob, sol, u, idx, prob.network.sw)
    end
    return out
end

# `:d` — 2 nonzeros/col at nu_bal[j]=-1, mu_ub[j]=sol.mu_ub[j]
function _dc_demand_vjp!(out, sol, u, idx, n)
    @inbounds for j in 1:n
        out[j] = u[idx.nu_bal[j]] - sol.mu_ub[j] * u[idx.mu_ub[j]]
    end
end

# `:cl` — 1 nonzero/col at pg[j]=1
function _dc_cost_linear_vjp!(out, u, idx, k)
    @inbounds for j in 1:k
        out[j] = -u[idx.pg[j]]
    end
end

# `:cq` — 1 nonzero/col at pg[j]=2*g[j]
function _dc_cost_quadratic_vjp!(out, sol, u, idx, k)
    @inbounds for j in 1:k
        out[j] = -2.0 * sol.pg[j] * u[idx.pg[j]]
    end
end

# `:fmax` — 2 nonzeros/col at lam_lb[e], lam_ub[e]
function _dc_flowlimit_vjp!(out, sol, u, idx, m)
    @inbounds for e in 1:m
        out[e] = -(sol.lam_lb[e] * u[idx.lam_lb[e]] + sol.lam_ub[e] * u[idx.lam_ub[e]])
    end
end

# `:sw`/`:b` — 3 KKT blocks (nu_bal, nu_flow, va), vectorized via sparse matvecs.
# `coeff` is net.b for :sw, net.sw for :b (the coefficient that multiplies the parameter).
function _dc_topo_vjp!(out, prob, sol, u, idx, coeff)
    A = prob.network.A
    Aθ = A * sol.va
    Au_nb = A * @view(u[idx.nu_bal])
    Au_va = A * @view(u[idx.va])
    Aν = A * sol.nu_bal
    @inbounds for e in 1:prob.network.m
        out[e] = -coeff[e] * (Aθ[e] * (Au_nb[e] + u[idx.nu_flow[e]]) -
                               (Aν[e] + sol.nu_flow[e]) * Au_va[e])
    end
end

# =============================================================================
# DC OPF: Inline Parameter JVP  (v += dK/dp · tang, v is kkt-space accumulator)
# =============================================================================
#
# Scatters tang[j]-weighted nonzeros into the kkt-dims accumulator.

function _dc_param_jvp!(v::AbstractVector, prob::DCOPFProblem,
                        sol::DCOPFSolution, param::Symbol,
                        tang::AbstractVector, idx::NamedTuple)
    if param === :d
        _dc_demand_jvp!(v, sol, tang, idx, prob.network.n)
    elseif param === :cl
        _dc_cost_linear_jvp!(v, tang, idx, prob.network.k)
    elseif param === :cq
        _dc_cost_quadratic_jvp!(v, sol, tang, idx, prob.network.k)
    elseif param === :fmax
        _dc_flowlimit_jvp!(v, sol, tang, idx, prob.network.m)
    elseif param === :sw
        _dc_topo_jvp!(v, prob, sol, tang, idx, prob.network.b)
    else  # :b
        _dc_topo_jvp!(v, prob, sol, tang, idx, prob.network.sw)
    end
    return v
end

# `:d` — scatter -tang[j] into nu_bal, mu_ub[j]*tang[j] into mu_ub
function _dc_demand_jvp!(v, sol, tang, idx, n)
    @inbounds for j in 1:n
        v[idx.nu_bal[j]] += -tang[j]
        v[idx.mu_ub[j]] += sol.mu_ub[j] * tang[j]
    end
end

function _dc_cost_linear_jvp!(v, tang, idx, k)
    @inbounds for j in 1:k
        v[idx.pg[j]] += tang[j]
    end
end

function _dc_cost_quadratic_jvp!(v, sol, tang, idx, k)
    @inbounds for j in 1:k
        v[idx.pg[j]] += 2.0 * sol.pg[j] * tang[j]
    end
end

function _dc_flowlimit_jvp!(v, sol, tang, idx, m)
    @inbounds for e in 1:m
        v[idx.lam_lb[e]] += sol.lam_lb[e] * tang[e]
        v[idx.lam_ub[e]] += sol.lam_ub[e] * tang[e]
    end
end

# `:sw`/`:b` — 2 sparse transpose-matvecs + elementwise scatter
function _dc_topo_jvp!(v, prob, sol, tang, idx, coeff)
    A = prob.network.A
    Aθ = A * sol.va
    Aν = A * sol.nu_bal

    # nu_bal block: v[nu_bal] += A' * (coeff .* Aθ .* tang)
    w_nb = coeff .* Aθ .* tang
    mul!(@view(v[idx.nu_bal]), A', w_nb, 1.0, 1.0)

    # nu_flow block: v[nu_flow] += coeff .* Aθ .* tang  (same as w_nb)
    @views v[idx.nu_flow] .+= w_nb

    # va block: v[va] += A' * (-coeff .* (Aν .+ nu_flow) .* tang)
    w_va = (-coeff) .* (Aν .+ sol.nu_flow) .* tang
    mul!(@view(v[idx.va]), A', w_va, 1.0, 1.0)
end

# =============================================================================
# DC OPF: VJP/JVP Core (in-place with caller-provided workspace)
# =============================================================================

"""
    _dcopf_vjp!(out, prob, op, param, adj, work) → out

In-place DC OPF VJP. `work` must be `Vector{Float64}` of length `kkt_dims(prob)`.
"""
function _dcopf_vjp!(out::AbstractVector, prob::DCOPFProblem, op::Symbol, param::Symbol,
                     adj::AbstractVector, work::AbstractVector)
    idx = kkt_indices(prob)
    op_rows = _dc_operand_kkt_rows(idx, op)

    # Fast path: if full dz/dp is cached, use matrix multiply
    field = _DC_CACHE_FIELD[param]
    cached = getfield(prob.cache, field)
    if !isnothing(cached)
        mul!(out, (@view cached[op_rows, :])', adj)
        return out
    end

    # Slow path: in-place transpose solve + inline parameter VJP
    kkt_lu = _ensure_kkt_factor!(prob)
    sol = _ensure_solved!(prob)

    # Step 1: Lift adjoint into KKT space (no sign flip for DC OPF)
    fill!(work, 0.0)
    @views work[op_rows] .= adj

    # Step 2: In-place transpose solve: work ← (dK/dz)⁻ᵀ · work
    ldiv!(transpose(kkt_lu), work)

    # Step 3: Inline parameter VJP: out = -(dK/dp)ᵀ · work
    _dc_param_vjp!(out, prob, sol, param, work, idx)

    return out
end

"""
    _dcopf_jvp!(out, prob, op, param, tang, work) → out

In-place DC OPF JVP. `work` must be `Vector{Float64}` of length `kkt_dims(prob)`.
"""
function _dcopf_jvp!(out::AbstractVector, prob::DCOPFProblem, op::Symbol, param::Symbol,
                     tang::AbstractVector, work::AbstractVector)
    idx = kkt_indices(prob)
    op_rows = _dc_operand_kkt_rows(idx, op)

    # Fast path: if full dz/dp is cached, use matrix multiply
    field = _DC_CACHE_FIELD[param]
    cached = getfield(prob.cache, field)
    if !isnothing(cached)
        mul!(out, @view(cached[op_rows, :]), tang)
        return out
    end

    # Slow path: inline parameter JVP + in-place forward solve
    kkt_lu = _ensure_kkt_factor!(prob)
    sol = _ensure_solved!(prob)

    # Step 1: Scatter dK/dp · tang into work (kkt-space accumulator)
    fill!(work, 0.0)
    _dc_param_jvp!(work, prob, sol, param, tang, idx)

    # Step 2: In-place forward solve: work ← (dK/dz)⁻¹ · work
    ldiv!(kkt_lu, work)

    # Step 3: Extract operand rows with negation (no sign flip for DC OPF)
    @inbounds for (i, r) in enumerate(op_rows)
        out[i] = -work[r]
    end

    return out
end

# Allocating wrappers
function _dcopf_vjp(prob::DCOPFProblem, op::Symbol, param::Symbol, adj::AbstractVector)
    pdim = _param_dim(prob, param)
    out = Vector{Float64}(undef, pdim)
    work = Vector{Float64}(undef, kkt_dims(prob))
    return _dcopf_vjp!(out, prob, op, param, adj, work)
end

function _dcopf_jvp(prob::DCOPFProblem, op::Symbol, param::Symbol, tang::AbstractVector)
    idx = kkt_indices(prob)
    odim = length(_dc_operand_kkt_rows(idx, op))
    out = Vector{Float64}(undef, odim)
    work = Vector{Float64}(undef, kkt_dims(prob))
    return _dcopf_jvp!(out, prob, op, param, tang, work)
end

# Parameter dimension helper
function _param_dim(prob::DCOPFProblem, param::Symbol)
    net = prob.network
    param in (:d,) && return net.n
    param in (:sw, :fmax, :b) && return net.m
    param in (:cq, :cl) && return net.k
    error("Unknown DC OPF parameter: $param")
end

function _param_dim(prob::ACOPFProblem, param::Symbol)
    net = prob.network
    param in (:d, :qd) && return net.n
    param in (:sw, :fmax) && return net.m
    param in (:cq, :cl) && return prob.n_gen
    error("Unknown AC OPF parameter: $param")
end

# =============================================================================
# AC OPF: Analytical Parameter Actions
# =============================================================================

"""Pre-extract all AC OPF parameter/state data needed by the analytical slow path."""
function _ac_kkt_context(prob::ACOPFProblem)
    sol = _ensure_ac_solved!(prob)
    idx = kkt_indices(prob)
    constants = _require_kkt_constants(prob)
    pd0 = _extract_bus_pd(prob)
    qd0 = _extract_bus_qd(prob)
    cq0 = _extract_gen_cq(prob)
    cl0 = _extract_gen_cl(prob)
    fmax0 = _extract_branch_fmax(prob)
    return (; sol, idx, constants, pd0, qd0, cq0, cl0, fmax0)
end

function _ac_param_vjp!(out::AbstractVector, prob::ACOPFProblem, ctx, param::Symbol, u::AbstractVector)
    sol = ctx.sol
    idx = ctx.idx
    vars = unflatten_variables(flatten_variables(sol, prob), idx)

    if param === :d
        copyto!(out, @view(u[idx.nu_p_bal]))
        return out
    elseif param === :qd
        copyto!(out, @view(u[idx.nu_q_bal]))
        return out
    elseif param === :cl
        copyto!(out, @view(u[idx.pg]))
        return out
    elseif param === :cq
        @inbounds for i in eachindex(sol.pg)
            out[i] = 2 * sol.pg[i] * u[idx.pg[i]]
        end
        return out
    elseif param === :fmax
        @inbounds for l in eachindex(ctx.fmax0)
            out[l] = -2 * ctx.fmax0[l] * (
                sol.lam_thermal_fr[l] * u[idx.lam_thermal_fr[l]] +
                sol.lam_thermal_to[l] * u[idx.lam_thermal_to[l]]
            ) +
            sol.sig_p_fr_lb[l] * u[idx.sig_p_fr_lb[l]] +
            sol.sig_p_fr_ub[l] * u[idx.sig_p_fr_ub[l]] +
            sol.sig_q_fr_lb[l] * u[idx.sig_q_fr_lb[l]] +
            sol.sig_q_fr_ub[l] * u[idx.sig_q_fr_ub[l]] +
            sol.sig_p_to_lb[l] * u[idx.sig_p_to_lb[l]] +
            sol.sig_p_to_ub[l] * u[idx.sig_p_to_ub[l]] +
            sol.sig_q_to_lb[l] * u[idx.sig_q_to_lb[l]] +
            sol.sig_q_to_ub[l] * u[idx.sig_q_to_ub[l]]
        end
        return out
    end

    @inbounds for l in 1:prob.network.m
        prim = _branch_flow_primitives(sol.va, sol.vm, prob.network.sw, ctx.constants, l)
        fb = prim.fb
        tb = prim.tb

        coeff_p_fr = -sol.nu_p_bal[fb] - 4 * sol.lam_thermal_fr[l] * prim.p_fr -
                     sol.sig_p_fr_lb[l] - sol.sig_p_fr_ub[l]
        coeff_q_fr = -sol.nu_q_bal[fb] - 4 * sol.lam_thermal_fr[l] * prim.q_fr -
                     sol.sig_q_fr_lb[l] - sol.sig_q_fr_ub[l]
        coeff_p_to = -sol.nu_p_bal[tb] - 4 * sol.lam_thermal_to[l] * prim.p_to -
                     sol.sig_p_to_lb[l] - sol.sig_p_to_ub[l]
        coeff_q_to = -sol.nu_q_bal[tb] - 4 * sol.lam_thermal_to[l] * prim.q_to -
                     sol.sig_q_to_lb[l] - sol.sig_q_to_ub[l]

        out[l] =
            u[idx.va[fb]] * (coeff_p_fr * prim.dp_fr_hat[1] + coeff_q_fr * prim.dq_fr_hat[1] +
                             coeff_p_to * prim.dp_to_hat[1] + coeff_q_to * prim.dq_to_hat[1]) +
            u[idx.va[tb]] * (coeff_p_fr * prim.dp_fr_hat[2] + coeff_q_fr * prim.dq_fr_hat[2] +
                             coeff_p_to * prim.dp_to_hat[2] + coeff_q_to * prim.dq_to_hat[2]) +
            u[idx.vm[fb]] * (coeff_p_fr * prim.dp_fr_hat[3] + coeff_q_fr * prim.dq_fr_hat[3] +
                             coeff_p_to * prim.dp_to_hat[3] + coeff_q_to * prim.dq_to_hat[3]) +
            u[idx.vm[tb]] * (coeff_p_fr * prim.dp_fr_hat[4] + coeff_q_fr * prim.dq_fr_hat[4] +
                             coeff_p_to * prim.dp_to_hat[4] + coeff_q_to * prim.dq_to_hat[4]) +
            u[idx.nu_p_bal[fb]] * prim.p_fr_hat +
            u[idx.nu_p_bal[tb]] * prim.p_to_hat +
            u[idx.nu_q_bal[fb]] * prim.q_fr_hat +
            u[idx.nu_q_bal[tb]] * prim.q_to_hat +
            u[idx.lam_thermal_fr[l]] * (2 * sol.lam_thermal_fr[l] * (prim.p_fr * prim.p_fr_hat + prim.q_fr * prim.q_fr_hat)) +
            u[idx.lam_thermal_to[l]] * (2 * sol.lam_thermal_to[l] * (prim.p_to * prim.p_to_hat + prim.q_to * prim.q_to_hat)) +
            u[idx.sig_p_fr_lb[l]] * (sol.sig_p_fr_lb[l] * prim.p_fr_hat) +
            u[idx.sig_p_fr_ub[l]] * (-sol.sig_p_fr_ub[l] * prim.p_fr_hat) +
            u[idx.sig_q_fr_lb[l]] * (sol.sig_q_fr_lb[l] * prim.q_fr_hat) +
            u[idx.sig_q_fr_ub[l]] * (-sol.sig_q_fr_ub[l] * prim.q_fr_hat) +
            u[idx.sig_p_to_lb[l]] * (sol.sig_p_to_lb[l] * prim.p_to_hat) +
            u[idx.sig_p_to_ub[l]] * (-sol.sig_p_to_ub[l] * prim.p_to_hat) +
            u[idx.sig_q_to_lb[l]] * (sol.sig_q_to_lb[l] * prim.q_to_hat) +
            u[idx.sig_q_to_ub[l]] * (-sol.sig_q_to_ub[l] * prim.q_to_hat)
    end
    return out
end

function _ac_param_jvp!(v::AbstractVector, prob::ACOPFProblem, ctx, param::Symbol,
                        tang::AbstractVector)
    sol = ctx.sol
    idx = ctx.idx

    if param === :d
        @views v[idx.nu_p_bal] .+= tang
        return v
    elseif param === :qd
        @views v[idx.nu_q_bal] .+= tang
        return v
    elseif param === :cl
        @views v[idx.pg] .+= tang
        return v
    elseif param === :cq
        @inbounds for i in eachindex(sol.pg)
            v[idx.pg[i]] += 2 * sol.pg[i] * tang[i]
        end
        return v
    elseif param === :fmax
        @inbounds for l in eachindex(ctx.fmax0)
            t = tang[l]
            v[idx.lam_thermal_fr[l]] += -2 * sol.lam_thermal_fr[l] * ctx.fmax0[l] * t
            v[idx.lam_thermal_to[l]] += -2 * sol.lam_thermal_to[l] * ctx.fmax0[l] * t
            v[idx.sig_p_fr_lb[l]] += sol.sig_p_fr_lb[l] * t
            v[idx.sig_p_fr_ub[l]] += sol.sig_p_fr_ub[l] * t
            v[idx.sig_q_fr_lb[l]] += sol.sig_q_fr_lb[l] * t
            v[idx.sig_q_fr_ub[l]] += sol.sig_q_fr_ub[l] * t
            v[idx.sig_p_to_lb[l]] += sol.sig_p_to_lb[l] * t
            v[idx.sig_p_to_ub[l]] += sol.sig_p_to_ub[l] * t
            v[idx.sig_q_to_lb[l]] += sol.sig_q_to_lb[l] * t
            v[idx.sig_q_to_ub[l]] += sol.sig_q_to_ub[l] * t
        end
        return v
    end

    @inbounds for l in 1:prob.network.m
        t = tang[l]
        t == 0.0 && continue
        prim = _branch_flow_primitives(sol.va, sol.vm, prob.network.sw, ctx.constants, l)
        fb = prim.fb
        tb = prim.tb

        coeff_p_fr = -sol.nu_p_bal[fb] - 4 * sol.lam_thermal_fr[l] * prim.p_fr -
                     sol.sig_p_fr_lb[l] - sol.sig_p_fr_ub[l]
        coeff_q_fr = -sol.nu_q_bal[fb] - 4 * sol.lam_thermal_fr[l] * prim.q_fr -
                     sol.sig_q_fr_lb[l] - sol.sig_q_fr_ub[l]
        coeff_p_to = -sol.nu_p_bal[tb] - 4 * sol.lam_thermal_to[l] * prim.p_to -
                     sol.sig_p_to_lb[l] - sol.sig_p_to_ub[l]
        coeff_q_to = -sol.nu_q_bal[tb] - 4 * sol.lam_thermal_to[l] * prim.q_to -
                     sol.sig_q_to_lb[l] - sol.sig_q_to_ub[l]

        v[idx.va[fb]] += t * (coeff_p_fr * prim.dp_fr_hat[1] + coeff_q_fr * prim.dq_fr_hat[1] +
                              coeff_p_to * prim.dp_to_hat[1] + coeff_q_to * prim.dq_to_hat[1])
        v[idx.va[tb]] += t * (coeff_p_fr * prim.dp_fr_hat[2] + coeff_q_fr * prim.dq_fr_hat[2] +
                              coeff_p_to * prim.dp_to_hat[2] + coeff_q_to * prim.dq_to_hat[2])
        v[idx.vm[fb]] += t * (coeff_p_fr * prim.dp_fr_hat[3] + coeff_q_fr * prim.dq_fr_hat[3] +
                              coeff_p_to * prim.dp_to_hat[3] + coeff_q_to * prim.dq_to_hat[3])
        v[idx.vm[tb]] += t * (coeff_p_fr * prim.dp_fr_hat[4] + coeff_q_fr * prim.dq_fr_hat[4] +
                              coeff_p_to * prim.dp_to_hat[4] + coeff_q_to * prim.dq_to_hat[4])

        v[idx.nu_p_bal[fb]] += t * prim.p_fr_hat
        v[idx.nu_p_bal[tb]] += t * prim.p_to_hat
        v[idx.nu_q_bal[fb]] += t * prim.q_fr_hat
        v[idx.nu_q_bal[tb]] += t * prim.q_to_hat

        v[idx.lam_thermal_fr[l]] += t * 2 * sol.lam_thermal_fr[l] * (prim.p_fr * prim.p_fr_hat + prim.q_fr * prim.q_fr_hat)
        v[idx.lam_thermal_to[l]] += t * 2 * sol.lam_thermal_to[l] * (prim.p_to * prim.p_to_hat + prim.q_to * prim.q_to_hat)

        v[idx.sig_p_fr_lb[l]] += t * sol.sig_p_fr_lb[l] * prim.p_fr_hat
        v[idx.sig_p_fr_ub[l]] += -t * sol.sig_p_fr_ub[l] * prim.p_fr_hat
        v[idx.sig_q_fr_lb[l]] += t * sol.sig_q_fr_lb[l] * prim.q_fr_hat
        v[idx.sig_q_fr_ub[l]] += -t * sol.sig_q_fr_ub[l] * prim.q_fr_hat
        v[idx.sig_p_to_lb[l]] += t * sol.sig_p_to_lb[l] * prim.p_to_hat
        v[idx.sig_p_to_ub[l]] += -t * sol.sig_p_to_ub[l] * prim.p_to_hat
        v[idx.sig_q_to_lb[l]] += t * sol.sig_q_to_lb[l] * prim.q_to_hat
        v[idx.sig_q_to_ub[l]] += -t * sol.sig_q_to_ub[l] * prim.q_to_hat
    end
    return v
end

# =============================================================================
# AC OPF: VJP/JVP Core
# =============================================================================

function _acopf_vjp(prob::ACOPFProblem, op::Symbol, param::Symbol, adj::AbstractVector)
    idx = kkt_indices(prob)
    op_rows = _ac_operand_kkt_rows(idx, op)
    sign = _ac_operand_sign(op)

    # Fast path: if full dz/dp is cached, use matrix multiply
    field = _AC_CACHE_FIELD[param]
    cached = getfield(prob.cache, field)
    if !isnothing(cached)
        return Vector((sign .* cached[op_rows, :])' * adj)
    end

    # Slow path: one transpose solve + one analytical parameter VJP
    kkt_lu = _ensure_ac_kkt_factor!(prob)
    ctx = _ac_kkt_context(prob)

    w = zeros(kkt_dims(prob))
    w[op_rows] .= sign .* adj
    u = kkt_lu' \ w

    out = Vector{Float64}(undef, _param_dim(prob, param))
    _ac_param_vjp!(out, prob, ctx, param, u)
    lmul!(-1, out)
    return out
end

function _acopf_jvp(prob::ACOPFProblem, op::Symbol, param::Symbol, tang::AbstractVector)
    idx = kkt_indices(prob)
    op_rows = _ac_operand_kkt_rows(idx, op)
    sign = _ac_operand_sign(op)

    # Fast path: if full dz/dp is cached, use matrix multiply
    field = _AC_CACHE_FIELD[param]
    cached = getfield(prob.cache, field)
    if !isnothing(cached)
        return Vector(sign .* (cached[op_rows, :] * tang))
    end

    # Slow path: one analytical parameter JVP + one forward solve
    kkt_lu = _ensure_ac_kkt_factor!(prob)
    ctx = _ac_kkt_context(prob)

    v = zeros(kkt_dims(prob))
    _ac_param_jvp!(v, prob, ctx, param, tang)
    u = kkt_lu \ v

    return sign .* (-u[op_rows])
end

# =============================================================================
# DC PF: VJP/JVP Core (vectorized, no per-branch loops)
# =============================================================================

function _dcpf_vjp(state::DCPowerFlowState, op::Symbol, param::Symbol, adj::AbstractVector)
    net = state.net
    nr = state.non_ref
    n, m = net.n, net.m

    # Lift :f adjoint into :va space via (W·A)ᵀ = Aᵀ·W
    if op === :f
        w_diag = -net.b .* net.sw
        adj_va = Vector(net.A' * (w_diag .* adj))
    else
        adj_va = adj
    end

    # Core: B_r is symmetric, so B_r⁻ᵀ = B_r⁻¹
    u = state.B_r_factor \ adj_va[nr]

    if param === :d
        result = zeros(n)
        result[nr] .= -u
        return result
    end

    # :sw/:b — vectorized via sparse matvecs
    coeffs = param === :sw ? net.b : net.sw
    Aθ = net.A * state.va
    u_full = zeros(n)
    u_full[nr] .= u
    Au = net.A * u_full

    result = coeffs .* Aθ .* Au
    if op === :f
        result .+= (-coeffs) .* Aθ .* adj
    end
    return result
end

function _dcpf_jvp(state::DCPowerFlowState, op::Symbol, param::Symbol, tang::AbstractVector)
    net = state.net
    nr = state.non_ref
    n, m = net.n, net.m

    if param === :d
        dva = zeros(n)
        dva[nr] .= -(state.B_r_factor \ tang[nr])
    else
        # :sw/:b — vectorized: rhs = A'[:, nr] * (weighted), then solve
        coeffs = param === :sw ? net.b : net.sw
        Aθ = net.A * state.va
        weighted = (-coeffs) .* Aθ .* tang
        rhs_full = Vector(net.A' * weighted)
        dva = zeros(n)
        dva[nr] .= -(state.B_r_factor \ rhs_full[nr])
    end

    op === :va && return dva

    # :f operand: W·A·dva + direct term
    w_diag = -net.b .* net.sw
    df = w_diag .* Vector(net.A * dva)
    if param !== :d
        coeffs = param === :sw ? net.b : net.sw
        Aθ = net.A * state.va
        df .+= (-coeffs) .* Aθ .* tang
    end
    return df
end

# =============================================================================
# Public API: Symbol Dispatch
# =============================================================================

# --- VJP: Vector input → Vector output ---

"""
    vjp(state, operand::Symbol, parameter::Symbol, adj::AbstractVector) → Vector

Efficient vector-Jacobian product `(∂operand/∂parameter)ᵀ · adj` without
materializing the full sensitivity matrix.

Uses a single KKT transpose-solve (O(nnz)) instead of building the full
O(n²) sensitivity matrix. For performance-critical loops, use `vjp!` with
pre-allocated buffers.

# Examples
```julia
prob = DCOPFProblem(net, d); solve!(prob)
w = randn(net.n)
grad_d = vjp(prob, :lmp, :d, w)   # ∂LMP/∂dᵀ · w, length n
```
"""
function vjp(state, operand::Symbol, parameter::Symbol, adj::AbstractVector)
    op = _resolve_operand(operand)
    param = _resolve_parameter(parameter)
    T = typeof(state)
    (op, param) in _valid_combinations(T) || _throw_invalid_combo(state, op, param)
    return _vjp_core(state, op, param, adj)
end

_vjp_core(prob::DCOPFProblem, op, param, adj) = _dcopf_vjp(prob, op, param, adj)
_vjp_core(prob::ACOPFProblem, op, param, adj) = _acopf_vjp(prob, op, param, adj)
_vjp_core(state::DCPowerFlowState, op, param, adj) = _dcpf_vjp(state, op, param, adj)
_vjp_core(state::ACPowerFlowState, op, param, adj) = throw(ArgumentError(
    "Efficient VJP is not supported for ACPowerFlowState. " *
    "Use `vjp(calc_sensitivity(state, :$op, :$param), adj)` instead."))

# --- VJP: Dict input → Dict output ---

"""
    vjp(state, operand::Symbol, parameter::Symbol, adj::AbstractDict{Int}) → Dict{Int,Float64}

ID-aware VJP. Input keyed by operand element IDs, output keyed by parameter element IDs.

Note: allocates a Dict for the output. For performance-critical loops, prefer
the `AbstractVector` interface.
"""
function vjp(state, operand::Symbol, parameter::Symbol, adj::AbstractDict{Int,<:Number})
    op = _resolve_operand(operand)
    param = _resolve_parameter(parameter)
    T = typeof(state)
    (op, param) in _valid_combinations(T) || _throw_invalid_combo(state, op, param)

    row_element = _OPERAND_ELEMENT[op]
    row_to_id, id_to_row = _element_mapping(state, row_element)
    adj_vec = _dict_to_vec(adj, id_to_row, length(row_to_id))
    result_vec = _vjp_core(state, op, param, adj_vec)
    col_element = _PARAM_ELEMENT[param]
    col_to_id, _ = _element_mapping(state, col_element)
    return _vec_to_dict(result_vec, col_to_id)
end

# --- VJP!: In-place with caller-provided workspace ---

"""
    vjp!(out, prob::DCOPFProblem, operand::Symbol, parameter::Symbol,
         adj::AbstractVector; work=nothing) → out

In-place VJP for performance-critical loops (e.g., bilevel optimization).

`out` must be pre-allocated to the parameter dimension. Pass `work` (a
`Vector{Float64}` of length `kkt_dims(prob)`) to override the internal
workspace (e.g., for thread-local buffers); if omitted, a workspace is
lazily allocated in `prob.cache` and reused across calls.

# Examples
```julia
prob = DCOPFProblem(net, d); solve!(prob)
out = zeros(net.n)

# Simple (workspace managed by cache):
vjp!(out, prob, :lmp, :d, adj)

# Thread-local override:
work = zeros(kkt_dims(prob))
vjp!(out, prob, :lmp, :d, adj; work=work)
```
"""
function vjp!(out::AbstractVector, prob::DCOPFProblem, operand::Symbol, parameter::Symbol,
              adj::AbstractVector; work::Union{Nothing,AbstractVector}=nothing)
    op = _resolve_operand(operand)
    param = _resolve_parameter(parameter)
    (op, param) in _valid_combinations(DCOPFProblem) || _throw_invalid_combo(prob, op, param)
    w = if !isnothing(work)
        work
    else
        if isnothing(prob.cache.work)
            prob.cache.work = Vector{Float64}(undef, kkt_dims(prob))
        end
        prob.cache.work
    end
    return _dcopf_vjp!(out, prob, op, param, adj, w)
end

# --- JVP: Vector input → Vector output ---

"""
    jvp(state, operand::Symbol, parameter::Symbol, tang::AbstractVector) → Vector

Efficient Jacobian-vector product `(∂operand/∂parameter) · tang` without
materializing the full sensitivity matrix.

Uses a single KKT forward-solve (O(nnz)) instead of building the full
O(n²) sensitivity matrix. For performance-critical loops, use `jvp!` with
pre-allocated buffers.

# Examples
```julia
prob = DCOPFProblem(net, d); solve!(prob)
δd = randn(net.n)
δlmp = jvp(prob, :lmp, :d, δd)   # ∂LMP/∂d · δd, length n
```
"""
function jvp(state, operand::Symbol, parameter::Symbol, tang::AbstractVector)
    op = _resolve_operand(operand)
    param = _resolve_parameter(parameter)
    T = typeof(state)
    (op, param) in _valid_combinations(T) || _throw_invalid_combo(state, op, param)
    return _jvp_core(state, op, param, tang)
end

_jvp_core(prob::DCOPFProblem, op, param, tang) = _dcopf_jvp(prob, op, param, tang)
_jvp_core(prob::ACOPFProblem, op, param, tang) = _acopf_jvp(prob, op, param, tang)
_jvp_core(state::DCPowerFlowState, op, param, tang) = _dcpf_jvp(state, op, param, tang)
_jvp_core(state::ACPowerFlowState, op, param, tang) = throw(ArgumentError(
    "Efficient JVP is not supported for ACPowerFlowState. " *
    "Use `jvp(calc_sensitivity(state, :$op, :$param), tang)` instead."))

# --- JVP: Dict input → Dict output ---

"""
    jvp(state, operand::Symbol, parameter::Symbol, tang::AbstractDict{Int}) → Dict{Int,Float64}

ID-aware JVP. Input keyed by parameter element IDs, output keyed by operand element IDs.

Note: allocates a Dict for the output. For performance-critical loops, prefer
the `AbstractVector` interface.
"""
function jvp(state, operand::Symbol, parameter::Symbol, tang::AbstractDict{Int,<:Number})
    op = _resolve_operand(operand)
    param = _resolve_parameter(parameter)
    T = typeof(state)
    (op, param) in _valid_combinations(T) || _throw_invalid_combo(state, op, param)

    col_element = _PARAM_ELEMENT[param]
    col_to_id, id_to_col = _element_mapping(state, col_element)
    tang_vec = _dict_to_vec(tang, id_to_col, length(col_to_id))
    result_vec = _jvp_core(state, op, param, tang_vec)
    row_element = _OPERAND_ELEMENT[op]
    row_to_id, _ = _element_mapping(state, row_element)
    return _vec_to_dict(result_vec, row_to_id)
end

# --- JVP!: In-place with caller-provided workspace ---

"""
    jvp!(out, prob::DCOPFProblem, operand::Symbol, parameter::Symbol,
         tang::AbstractVector; work=nothing) → out

In-place JVP for performance-critical loops (e.g., bilevel optimization).

`out` must be pre-allocated to the operand dimension. Pass `work` (a
`Vector{Float64}` of length `kkt_dims(prob)`) to override the internal
workspace (e.g., for thread-local buffers); if omitted, a workspace is
lazily allocated in `prob.cache` and reused across calls.

# Examples
```julia
prob = DCOPFProblem(net, d); solve!(prob)
out = zeros(net.n)

# Simple (workspace managed by cache):
jvp!(out, prob, :lmp, :d, tang)

# Thread-local override:
work = zeros(kkt_dims(prob))
jvp!(out, prob, :lmp, :d, tang; work=work)
```
"""
function jvp!(out::AbstractVector, prob::DCOPFProblem, operand::Symbol, parameter::Symbol,
              tang::AbstractVector; work::Union{Nothing,AbstractVector}=nothing)
    op = _resolve_operand(operand)
    param = _resolve_parameter(parameter)
    (op, param) in _valid_combinations(DCOPFProblem) || _throw_invalid_combo(prob, op, param)
    w = if !isnothing(work)
        work
    else
        if isnothing(prob.cache.work)
            prob.cache.work = Vector{Float64}(undef, kkt_dims(prob))
        end
        prob.cache.work
    end
    return _dcopf_jvp!(out, prob, op, param, tang, w)
end
