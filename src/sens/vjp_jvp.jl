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
# DC OPF: Parameter Jacobian Builders (full sparse matrices)
# =============================================================================

const _DC_PARAM_JAC_FN = Dict{Symbol, Function}(
    :d    => (prob, sol) -> calc_kkt_jacobian_demand(prob.network, prob.d, sol),
    :sw   => (prob, sol) -> calc_kkt_jacobian_switching(prob, sol),
    :cq   => (prob, sol) -> calc_kkt_jacobian_cost_quadratic(prob, sol),
    :cl   => (prob, _)   -> calc_kkt_jacobian_cost_linear(prob.network),
    :fmax => (prob, sol) -> calc_kkt_jacobian_flowlimit(prob, sol),
    :b    => (prob, sol) -> calc_kkt_jacobian_susceptance(prob, sol),
)

# =============================================================================
# DC OPF: VJP/JVP Core
# =============================================================================

function _dcopf_vjp(prob::DCOPFProblem, op::Symbol, param::Symbol, adj::AbstractVector)
    idx = kkt_indices(prob)
    op_rows = _dc_operand_kkt_rows(idx, op)

    # Fast path: if full dz/dp is cached, use matrix multiply
    field = _DC_CACHE_FIELD[param]
    cached = getfield(prob.cache, field)
    if !isnothing(cached)
        return Vector(cached[op_rows, :]' * adj)
    end

    # Slow path: one transpose solve + one sparse matvec
    kkt_lu = _ensure_kkt_factor!(prob)
    sol = _ensure_solved!(prob)

    # Step 1: Lift adjoint into KKT space (no sign flip for DC OPF)
    w = zeros(kkt_dims(prob))
    w[op_rows] .= adj

    # Step 2: Transpose solve: u = (dK/dz)⁻ᵀ · w
    u = kkt_lu' \ w

    # Step 3: result = -(dK/dp)ᵀ · u
    J_p = _DC_PARAM_JAC_FN[param](prob, sol)
    return Vector(-(J_p' * u))
end

function _dcopf_jvp(prob::DCOPFProblem, op::Symbol, param::Symbol, tang::AbstractVector)
    idx = kkt_indices(prob)
    op_rows = _dc_operand_kkt_rows(idx, op)

    # Fast path: if full dz/dp is cached, use matrix multiply
    field = _DC_CACHE_FIELD[param]
    cached = getfield(prob.cache, field)
    if !isnothing(cached)
        return Vector(cached[op_rows, :] * tang)
    end

    # Slow path: one sparse matvec + one forward solve
    kkt_lu = _ensure_kkt_factor!(prob)
    sol = _ensure_solved!(prob)

    # Step 1: v = dK/dp · tang
    J_p = _DC_PARAM_JAC_FN[param](prob, sol)
    v = Vector(J_p * tang)

    # Step 2: u = -(dK/dz)⁻¹ · v
    u = kkt_lu \ v

    # Step 3: Extract operand rows (no sign flip for DC OPF)
    return -u[op_rows]
end

# =============================================================================
# AC OPF: ForwardDiff Context Setup
# =============================================================================

"""Pre-extract all AC OPF parameters for ForwardDiff closures."""
function _ac_kkt_context(prob::ACOPFProblem)
    sol = _ensure_ac_solved!(prob)
    z0 = flatten_variables(sol, prob)
    sw = prob.network.sw
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
    return (; sol, z0, sw, idx, constants, pd0, qd0, cq0, cl0, fmax0)
end

"""
Compute (dK/dp)ᵀ · u via ForwardDiff.gradient of dot(u, kkt(...)).

Uses a single forward-mode pass: ∇_p [uᵀ · K(z,p)] = J_Kpᵀ · u.
"""
function _ac_param_vjp_grad(prob::ACOPFProblem, ctx, param::Symbol, u::AbstractVector)
    if param === :sw
        return ForwardDiff.gradient(
            s -> dot(u, kkt(ctx.z0, prob, s;
                pd=ctx.pd0, qd=ctx.qd0, cq=ctx.cq0, cl=ctx.cl0, fmax=ctx.fmax0,
                idx=ctx.idx, constants=ctx.constants)),
            ctx.sw)
    else
        kw = _PARAM_KWARG_MAP[param]
        p0 = _AC_PARAM_EXTRACT[param](prob)
        all_fixed = Dict{Symbol,Any}(
            :pd => ctx.pd0, :qd => ctx.qd0, :cq => ctx.cq0,
            :cl => ctx.cl0, :fmax => ctx.fmax0)
        delete!(all_fixed, kw)
        fixed_nt = (; (k => v for (k, v) in all_fixed)...)
        return ForwardDiff.gradient(
            x -> dot(u, kkt(ctx.z0, prob, ctx.sw;
                NamedTuple{(kw,)}((x,))..., fixed_nt...,
                idx=ctx.idx, constants=ctx.constants)),
            p0)
    end
end

"""
Compute dK/dp · tang via ForwardDiff.derivative (directional derivative).

A single scalar derivative: f'(0) where f(t) = K(z, p₀ + t·tang).
Cost is O(kkt_dims), independent of param_dims.
"""
function _ac_param_jvp_deriv(prob::ACOPFProblem, ctx, param::Symbol, tang::AbstractVector)
    if param === :sw
        return ForwardDiff.derivative(
            t -> kkt(ctx.z0, prob, ctx.sw .+ t .* tang;
                pd=ctx.pd0, qd=ctx.qd0, cq=ctx.cq0, cl=ctx.cl0, fmax=ctx.fmax0,
                idx=ctx.idx, constants=ctx.constants),
            0.0)
    else
        kw = _PARAM_KWARG_MAP[param]
        p0 = _AC_PARAM_EXTRACT[param](prob)
        all_fixed = Dict{Symbol,Any}(
            :pd => ctx.pd0, :qd => ctx.qd0, :cq => ctx.cq0,
            :cl => ctx.cl0, :fmax => ctx.fmax0)
        delete!(all_fixed, kw)
        fixed_nt = (; (k => v for (k, v) in all_fixed)...)
        return ForwardDiff.derivative(
            t -> kkt(ctx.z0, prob, ctx.sw;
                NamedTuple{(kw,)}((p0 .+ t .* tang,))..., fixed_nt...,
                idx=ctx.idx, constants=ctx.constants),
            0.0)
    end
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

    # Slow path: one transpose solve + one ForwardDiff gradient
    kkt_lu = _ensure_ac_kkt_factor!(prob)
    ctx = _ac_kkt_context(prob)

    # Step 1: Lift adjoint into KKT space with sign
    w = zeros(kkt_dims(prob))
    w[op_rows] .= sign .* adj

    # Step 2: Transpose solve
    u = kkt_lu' \ w

    # Step 3: result = -(dK/dp)ᵀ · u via ForwardDiff.gradient
    return -_ac_param_vjp_grad(prob, ctx, param, u)
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

    # Slow path: one ForwardDiff derivative + one forward solve
    kkt_lu = _ensure_ac_kkt_factor!(prob)
    ctx = _ac_kkt_context(prob)

    # Step 1: v = dK/dp · tang via ForwardDiff.derivative
    v = _ac_param_jvp_deriv(prob, ctx, param, tang)

    # Step 2: u = -(dK/dz)⁻¹ · v
    u = kkt_lu \ v

    # Step 3: Extract operand rows with sign
    return sign .* (-u[op_rows])
end

# =============================================================================
# DC PF: VJP/JVP Core
# =============================================================================

function _dcpf_vjp(state::DCPowerFlowState, op::Symbol, param::Symbol, adj::AbstractVector)
    net = state.net
    nr = state.non_ref
    n, m = net.n, net.m

    # Lift :f adjoint into :va space: adj_va = Aᵀ · W · adj_f
    if op === :f
        W = Diagonal(-net.b .* net.sw)
        adj_va = Vector(net.A' * W * adj)
    else
        adj_va = adj
    end

    # Core: solve Bᵣᵀ \ adj_va[nr] (B_r is symmetric, so same as forward solve)
    u = state.B_r_factor \ adj_va[nr]

    if param === :d
        # VJP of dva/dd: result[nr] = -(B_r⁻¹ · adj_va[nr])
        result = zeros(n)
        result[nr] .= -u
        if op === :f
            # Direct term is zero for demand (no direct dependence of f on d)
        end
        return result
    end

    # :sw and :b share structure: differ only in coefficient per branch
    Aθ = net.A * state.va
    # Embed u into full n-vector so dot(A[e,:], u_full) works without slicing
    u_full = zeros(n)
    u_full[nr] .= u
    result = zeros(m)
    for e in 1:m
        coeff = param === :sw ? net.b[e] : net.sw[e]
        # Indirect term: coeff · (A·θ)[e] · dot(A[e,:], u_full)
        result[e] = coeff * Aθ[e] * dot(net.A[e, :], u_full)
    end

    if op === :f
        # Direct term: -coeff_e · (A·θ)[e] · adj_f[e]
        for e in 1:m
            coeff = param === :sw ? net.b[e] : net.sw[e]
            result[e] += -coeff * Aθ[e] * adj[e]
        end
    end

    return result
end

function _dcpf_jvp(state::DCPowerFlowState, op::Symbol, param::Symbol, tang::AbstractVector)
    net = state.net
    nr = state.non_ref
    n, m = net.n, net.m

    Aθ = param !== :d ? net.A * state.va : nothing

    if param === :d
        # JVP of dva/dd · tang: -(B_r \ tang[nr])
        dva = zeros(n)
        dva[nr] .= -(state.B_r_factor \ tang[nr])
    else
        # :sw / :b: accumulate RHS = Σ_e tang[e] · (-coeff_e · Aθ_e · a_{e,r})
        n_r = length(nr)
        rhs = zeros(n_r)
        for e in 1:m
            coeff = param === :sw ? net.b[e] : net.sw[e]
            a_e_r = Vector(net.A[e, nr])
            rhs .+= tang[e] .* (-coeff * Aθ[e]) .* a_e_r
        end
        dva = zeros(n)
        dva[nr] .= -(state.B_r_factor \ rhs)
    end

    if op === :va
        return dva
    end

    # :f operand: f = W·A·va + direct term for :sw/:b
    W = Diagonal(-net.b .* net.sw)
    df = Vector(W * net.A * dva)
    if param !== :d
        # Direct term: -coeff_e · (A·θ)[e] · tang[e]
        for e in 1:m
            coeff = param === :sw ? net.b[e] : net.sw[e]
            df[e] += -coeff * Aθ[e] * tang[e]
        end
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
O(n²) sensitivity matrix.

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
"""
function vjp(state, operand::Symbol, parameter::Symbol, adj::AbstractDict{Int,<:Number})
    op = _resolve_operand(operand)
    param = _resolve_parameter(parameter)
    T = typeof(state)
    (op, param) in _valid_combinations(T) || _throw_invalid_combo(state, op, param)

    # Convert Dict → Vector
    row_element = _OPERAND_ELEMENT[op]
    row_to_id, id_to_row = _element_mapping(state, row_element)
    adj_vec = _dict_to_vec(adj, id_to_row, length(row_to_id))

    result_vec = _vjp_core(state, op, param, adj_vec)

    # Convert Vector → Dict
    col_element = _PARAM_ELEMENT[param]
    col_to_id, _ = _element_mapping(state, col_element)
    return _vec_to_dict(result_vec, col_to_id)
end

# --- JVP: Vector input → Vector output ---

"""
    jvp(state, operand::Symbol, parameter::Symbol, tang::AbstractVector) → Vector

Efficient Jacobian-vector product `(∂operand/∂parameter) · tang` without
materializing the full sensitivity matrix.

Uses a single KKT forward-solve (O(nnz)) instead of building the full
O(n²) sensitivity matrix.

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
"""
function jvp(state, operand::Symbol, parameter::Symbol, tang::AbstractDict{Int,<:Number})
    op = _resolve_operand(operand)
    param = _resolve_parameter(parameter)
    T = typeof(state)
    (op, param) in _valid_combinations(T) || _throw_invalid_combo(state, op, param)

    # Convert Dict → Vector
    col_element = _PARAM_ELEMENT[param]
    col_to_id, id_to_col = _element_mapping(state, col_element)
    tang_vec = _dict_to_vec(tang, id_to_col, length(col_to_id))

    result_vec = _jvp_core(state, op, param, tang_vec)

    # Convert Vector → Dict
    row_element = _OPERAND_ELEMENT[op]
    row_to_id, _ = _element_mapping(state, row_element)
    return _vec_to_dict(result_vec, row_to_id)
end
