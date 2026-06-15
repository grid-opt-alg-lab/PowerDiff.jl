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
# KKT System for DC OPF
# =============================================================================
#
# Implements KKT conditions for implicit differentiation of DC OPF solutions.

# Tikhonov regularization for degenerate complementarity (e.g., all generators at bounds).
# Smallest perturbation that restores invertibility without introducing O(eps) error.
const TIKHONOV_EPS = 1e-10

# =============================================================================
# Cached Solution and KKT Factorization Access
# =============================================================================

"""
    _ensure_solved!(prob::DCOPFProblem) → DCOPFSolution

Ensure the problem is solved and return the cached solution.
If not yet solved, calls solve!(prob) and caches the result.
"""
function _ensure_solved!(prob::DCOPFProblem)::DCOPFSolution
    if isnothing(prob.cache.solution)
        prob.cache.solution = solve!(prob)
    end
    return prob.cache.solution
end

"""
    _ensure_kkt_factor!(prob::DCOPFProblem) → LU

Ensure the KKT Jacobian factorization is computed and cached.
Returns the LU factorization for efficient repeated solves.
"""
function _ensure_kkt_factor!(prob::DCOPFProblem)
    if isnothing(prob.cache.kkt_factor)
        sol = _ensure_solved!(prob)
        J_z = calc_kkt_jacobian(prob; sol=sol)
        # Tikhonov regularization only when needed: degenerate complementarity
        # (e.g. psh=0 at buses with d=0) can produce exact singularity.
        prob.cache.kkt_factor = try
            lu(J_z)   # sparse LU (UmfpackLU)
        catch e
            if e isa LinearAlgebra.SingularException
                _SILENCE_WARNINGS[] || @warn "KKT Jacobian is singular (likely degenerate complementarity, e.g., generators at bounds); applying Tikhonov perturbation (eps=$TIKHONOV_EPS). Sensitivity accuracy may be reduced."
                J_reg = J_z + TIKHONOV_EPS * sparse(I, size(J_z, 1), size(J_z, 1))
                try
                    lu(J_reg)
                catch e2
                    e2 isa LinearAlgebra.SingularException || rethrow(e2)
                    error("KKT Jacobian remains singular after Tikhonov perturbation")
                end
            else
                rethrow(e)
            end
        end
    end
    return prob.cache.kkt_factor
end

# Must match solve!'s snap tolerance so the canonicalized duals agree with the KKT system.
@inline _is_fixed_zero_shed(d::Real) = _shed_capacity(d) < COMPLEMENTARITY_SNAP_TOL
@inline _shed_capacity_derivative(d::Real) = _is_fixed_zero_shed(d) ? zero(d) : one(d)

# =============================================================================
# Cached Derivative Computation Functions
# =============================================================================
#
# Each _get_dz_d*! function computes dz/dp = -(dK/dz)⁻¹ · (dK/dp) for one
# parameter type, caching the result. The sparse RHS from calc_kkt_jacobian_*
# is converted to dense via Matrix() before the backsolve because Julia ≥1.12
# does not support UmfpackLU \ SparseMatrixCSC (MethodError in ldiv!).

"""
    _get_dz_dd!(prob::DCOPFProblem) → Matrix{Float64}

Get or compute the full KKT derivative matrix w.r.t. demand.
Uses cached value if available, otherwise computes and caches.
"""
function _get_dz_dd!(prob::DCOPFProblem)::Matrix{Float64}
    if isnothing(prob.cache.dz_dd)
        kkt_lu = _ensure_kkt_factor!(prob)
        sol = _ensure_solved!(prob)
        J_d = calc_kkt_jacobian_demand(prob.network, prob.d, sol)
        rhs = Matrix(J_d)
        ldiv!(kkt_lu, rhs)
        prob.cache.dz_dd = lmul!(-1, rhs)
    end
    return prob.cache.dz_dd
end

"""
    _get_dz_dcl!(prob::DCOPFProblem) → Matrix{Float64}

Get or compute the full KKT derivative matrix w.r.t. linear cost.
"""
function _get_dz_dcl!(prob::DCOPFProblem)::Matrix{Float64}
    if isnothing(prob.cache.dz_dcl)
        kkt_lu = _ensure_kkt_factor!(prob)
        J_cl = calc_kkt_jacobian_cost_linear(prob.network)
        rhs = Matrix(J_cl)
        ldiv!(kkt_lu, rhs)
        prob.cache.dz_dcl = lmul!(-1, rhs)
    end
    return prob.cache.dz_dcl
end

"""
    _get_dz_dcq!(prob::DCOPFProblem) → Matrix{Float64}

Get or compute the full KKT derivative matrix w.r.t. quadratic cost.
"""
function _get_dz_dcq!(prob::DCOPFProblem)::Matrix{Float64}
    if isnothing(prob.cache.dz_dcq)
        kkt_lu = _ensure_kkt_factor!(prob)
        sol = _ensure_solved!(prob)
        J_cq = calc_kkt_jacobian_cost_quadratic(prob, sol)
        rhs = Matrix(J_cq)
        ldiv!(kkt_lu, rhs)
        prob.cache.dz_dcq = lmul!(-1, rhs)
    end
    return prob.cache.dz_dcq
end

"""
    _get_dz_dsw!(prob::DCOPFProblem) → Matrix{Float64}

Get or compute the full KKT derivative matrix w.r.t. switching.
"""
function _get_dz_dsw!(prob::DCOPFProblem)::Matrix{Float64}
    if isnothing(prob.cache.dz_dsw)
        kkt_lu = _ensure_kkt_factor!(prob)
        sol = _ensure_solved!(prob)
        J_s = calc_kkt_jacobian_switching(prob, sol)
        rhs = Matrix(J_s)
        ldiv!(kkt_lu, rhs)
        prob.cache.dz_dsw = lmul!(-1, rhs)
    end
    return prob.cache.dz_dsw
end

"""
    _get_dz_dfmax!(prob::DCOPFProblem) → Matrix{Float64}

Get or compute the full KKT derivative matrix w.r.t. flow limits.
"""
function _get_dz_dfmax!(prob::DCOPFProblem)::Matrix{Float64}
    if isnothing(prob.cache.dz_dfmax)
        kkt_lu = _ensure_kkt_factor!(prob)
        sol = _ensure_solved!(prob)
        J_fmax = calc_kkt_jacobian_flowlimit(prob, sol)
        rhs = Matrix(J_fmax)
        ldiv!(kkt_lu, rhs)
        prob.cache.dz_dfmax = lmul!(-1, rhs)
    end
    return prob.cache.dz_dfmax
end

"""
    _get_dz_db!(prob::DCOPFProblem) → Matrix{Float64}

Get or compute the full KKT derivative matrix w.r.t. susceptances.
"""
function _get_dz_db!(prob::DCOPFProblem)::Matrix{Float64}
    if isnothing(prob.cache.dz_db)
        kkt_lu = _ensure_kkt_factor!(prob)
        sol = _ensure_solved!(prob)
        J_b = calc_kkt_jacobian_susceptance(prob, sol)
        rhs = Matrix(J_b)
        ldiv!(kkt_lu, rhs)
        prob.cache.dz_db = lmul!(-1, rhs)
    end
    return prob.cache.dz_db
end

# =============================================================================
# Single-Matrix Extraction Functions
# =============================================================================

"""
    _extract_sensitivity(prob::DCOPFProblem, dz_dp::Matrix, operand::Symbol) → Matrix{Float64}

Extract a single sensitivity matrix from the full KKT derivative.

# Arguments
- `prob`: The DC OPF problem
- `dz_dp`: Full derivative matrix from KKT system
- `operand`: Which operand to extract (:va, :pg, :f, :psh, :lmp)
"""
function _extract_sensitivity(prob::DCOPFProblem, dz_dp::Matrix{Float64}, operand::Symbol)::Matrix{Float64}
    idx = kkt_indices(prob)

    if operand === :va
        return dz_dp[idx.va, :]
    elseif operand === :pg
        return dz_dp[idx.pg, :]
    elseif operand === :f
        return dz_dp[idx.f, :]
    elseif operand === :psh
        return dz_dp[idx.psh, :]
    elseif operand === :lmp
        # DC OPF: LMP = ν_bal directly (no negation).
        # The constraint G*g + psh - d = B*θ places demand negatively →
        # JuMP dual ν_bal > 0 at optimum. See lmp.jl:31-35.
        return dz_dp[idx.nu_bal, :]
    else
        throw(ArgumentError("Unknown operand: $operand"))
    end
end

# =============================================================================
# Single-Column Extraction Helpers
# =============================================================================

# Map parameter symbols to cache field names (mirrors _AC_CACHE_FIELD in kkt_ac_opf.jl)
const _DC_CACHE_FIELD = Dict{Symbol, Symbol}(
    :d => :dz_dd, :sw => :dz_dsw, :cq => :dz_dcq,
    :cl => :dz_dcl, :fmax => :dz_dfmax, :b => :dz_db,
)

# Map parameter symbols to single-column Jacobian builder functions.
# Each returns Vector{Float64} of size kkt_dims — O(1) nonzeros per column.
const _DC_PARAM_COL_FN = Dict{Symbol, Function}(
    :d    => (prob, sol, j) -> calc_kkt_jacobian_demand_column(prob.network, prob.d, sol, j),
    :sw   => (prob, sol, e) -> calc_kkt_jacobian_switching_column(prob, sol, e),
    :cq   => (prob, sol, j) -> calc_kkt_jacobian_cost_quadratic_column(prob.network, sol, j),
    :cl   => (prob, _, j)   -> calc_kkt_jacobian_cost_linear_column(prob.network, j),
    :fmax => (prob, sol, e) -> calc_kkt_jacobian_flowlimit_column(prob.network, sol, e),
    :b    => (prob, sol, e) -> calc_kkt_jacobian_susceptance_column(prob, sol, e),
)

"""
    _dc_operand_kkt_rows(idx::NamedTuple, op::Symbol) → UnitRange{Int}

Return the KKT index range for a DC OPF operand.
"""
function _dc_operand_kkt_rows(idx::NamedTuple, op::Symbol)
    op === :va  && return idx.va
    op === :pg  && return idx.pg
    op === :f   && return idx.f
    op === :psh && return idx.psh
    # DC OPF: no sign flip for LMP — ν_bal > 0 by DC constraint convention (see lmp.jl:31-35)
    op === :lmp && return idx.nu_bal
    throw(ArgumentError("Unknown DC OPF operand: $op"))
end

"""
    _extract_dz_column(prob::DCOPFProblem, dz_col::Vector{Float64}, op::Symbol) → Vector{Float64}

Extract operand rows from a single dz/dp column vector.
"""
function _extract_dz_column(prob::DCOPFProblem, dz_col::Vector{Float64}, op::Symbol)
    idx = kkt_indices(prob)
    return dz_col[_dc_operand_kkt_rows(idx, op)]
end

# =============================================================================
# Dimension Calculations
# =============================================================================

"""
    kkt_dims(prob::DCOPFProblem)
    kkt_dims(network::DCNetwork)

Compute the dimension of the flattened KKT variable vector.

The KKT system includes:
- Primal: va (n), pg (k), f (m), psh (n)
- Dual (inequality): lam_lb (m), lam_ub (m), gamma_lb (m), gamma_ub (m), rho_lb (k), rho_ub (k), mu_lb (n), mu_ub (n)
- Dual (equality): nu_bal (n), nu_flow (m)
- Reference bus constraint: 1

Total: 5n + 6m + 3k + 1
"""
kkt_dims(prob::DCOPFProblem) = kkt_dims(prob.network)
kkt_dims(n::Int, m::Int, k::Int) = 5n + 6m + 3k + 1

function kkt_dims(net::DCNetwork)
    n = getfield(net, :n)
    m = getfield(net, :m)
    k = getfield(net, :k)
    # va(n) + pg(k) + f(m) + psh(n) + lam_lb(m) + lam_ub(m) + gamma_lb(m) + gamma_ub(m) + rho_lb(k) + rho_ub(k) + mu_lb(n) + mu_ub(n) + nu_bal(n) + nu_flow(m) + ref(1)
    return kkt_dims(n, m, k)
end

"""
    kkt_indices(n, m, k) → NamedTuple

Compute all KKT variable indices from network dimensions.
Single source of truth for index calculations.

# Variable ordering
[va(n), pg(k), f(m), psh(n), lam_lb(m), lam_ub(m), gamma_lb(m), gamma_ub(m), rho_lb(k), rho_ub(k), mu_lb(n), mu_ub(n), nu_bal(n), nu_flow(m), eta(1)]

# Returns
NamedTuple with index ranges for each variable block.
"""
function kkt_indices(n::Int, m::Int, k::Int)
    i = 0
    idx_θ = (i+1):(i+n); i += n
    idx_g = (i+1):(i+k); i += k
    idx_f = (i+1):(i+m); i += m
    idx_psh = (i+1):(i+n); i += n
    idx_λ_lb = (i+1):(i+m); i += m
    idx_λ_ub = (i+1):(i+m); i += m
    idx_γ_lb = (i+1):(i+m); i += m
    idx_γ_ub = (i+1):(i+m); i += m
    idx_ρ_lb = (i+1):(i+k); i += k
    idx_ρ_ub = (i+1):(i+k); i += k
    idx_μ_lb = (i+1):(i+n); i += n
    idx_μ_ub = (i+1):(i+n); i += n
    idx_ν_bal = (i+1):(i+n); i += n
    idx_ν_flow = (i+1):(i+m); i += m
    idx_η = i + 1

    return (
        va = idx_θ, pg = idx_g, f = idx_f, psh = idx_psh,
        lam_lb = idx_λ_lb, lam_ub = idx_λ_ub,
        gamma_lb = idx_γ_lb, gamma_ub = idx_γ_ub,
        rho_lb = idx_ρ_lb, rho_ub = idx_ρ_ub,
        mu_lb = idx_μ_lb, mu_ub = idx_μ_ub,
        nu_bal = idx_ν_bal, nu_flow = idx_ν_flow, η = idx_η
    )
end

kkt_indices(net::DCNetwork) = kkt_indices(net.n, net.m, net.k)
kkt_indices(prob::DCOPFProblem) = kkt_indices(prob.network)

# =============================================================================
# Variable Flattening/Unflattening
# =============================================================================

"""
    flatten_variables(sol::DCOPFSolution, prob::DCOPFProblem)

Flatten solution primal and dual variables into a single vector for KKT evaluation.

# Variable ordering
[va; pg; f; psh; lam_lb; lam_ub; gamma_lb; gamma_ub; rho_lb; rho_ub; mu_lb; mu_ub; nu_bal; nu_flow; eta]

where eta is the dual for the reference bus constraint `va[ref_bus] == 0`, packed
from the recovered `sol.eta_ref` (the KKT operator includes this entry).
"""
function flatten_variables(sol::DCOPFSolution, prob::DCOPFProblem)
    return vcat(
        sol.va,
        sol.pg,
        sol.f,
        sol.psh,
        sol.lam_lb,
        sol.lam_ub,
        sol.gamma_lb,
        sol.gamma_ub,
        sol.rho_lb,
        sol.rho_ub,
        sol.mu_lb,
        sol.mu_ub,
        sol.nu_bal,
        sol.nu_flow,
        [sol.eta_ref]
    )
end

"""
    unflatten_variables(z::AbstractVector, prob::DCOPFProblem)

Unflatten KKT variable vector into named components.

# Returns
NamedTuple with fields: va, pg, f, psh, lam_lb, lam_ub, gamma_lb, gamma_ub, rho_lb, rho_ub, mu_lb, mu_ub, nu_bal, nu_flow, eta
"""
function unflatten_variables(z::AbstractVector, prob::DCOPFProblem)
    return unflatten_variables(z, prob.network)
end

function unflatten_variables(z::AbstractVector, net::DCNetwork)
    idx = kkt_indices(net)
    return (
        va = z[idx.va], pg = z[idx.pg], f = z[idx.f], psh = z[idx.psh],
        lam_lb = z[idx.lam_lb], lam_ub = z[idx.lam_ub],
        gamma_lb = z[idx.gamma_lb], gamma_ub = z[idx.gamma_ub],
        rho_lb = z[idx.rho_lb], rho_ub = z[idx.rho_ub],
        mu_lb = z[idx.mu_lb], mu_ub = z[idx.mu_ub],
        nu_bal = z[idx.nu_bal], nu_flow = z[idx.nu_flow],
        η_ref = z[idx.η]
    )
end

# =============================================================================
# KKT Operator
# =============================================================================

"""
    kkt(z::AbstractVector, prob::DCOPFProblem, d::AbstractVector)

Evaluate the KKT conditions for the B-θ DC OPF problem.

The KKT system for DC OPF:
```
min  g' Cq g + cl' g + c_shed' * psh + (τ²/2) ||f||²
s.t. G_inc * g + psh - d = B * θ   (ν_bal)
     f = W * A * θ                  (ν_flow)
     f ≥ -fmax                      (λ_lb)
     f ≤ fmax                       (λ_ub)
     A * θ ≥ angmin                 (γ_lb)
     A * θ ≤ angmax                 (γ_ub)
     g ≥ gmin                       (ρ_lb)
     g ≤ gmax                       (ρ_ub)
     0 ≤ psh                        (μ_lb)
     psh ≤ max(d, 0)                (μ_ub)
     θ[ref] = 0                     (η_ref)
```

# Returns
Vector of KKT residuals (should be zero at optimum):
1. Stationarity w.r.t. θ: B' * ν_bal + (W*A)' * ν_flow + e_ref * η_ref + A'*(γ_ub - γ_lb) = 0
2. Stationarity w.r.t. g: 2*Cq * g + cl - G_inc' * ν_bal - ρ_lb + ρ_ub = 0
3. Stationarity w.r.t. f: τ² * f - ν_flow - λ_lb + λ_ub = 0
4. Stationarity w.r.t. psh: c_shed - ν_bal - μ_lb + μ_ub = 0
5. Complementary slackness for flow bounds
6. Complementary slackness for phase angle difference bounds
7. Complementary slackness for gen bounds
8. Complementary slackness for shedding bounds
9. Primal feasibility: power balance
10. Primal feasibility: flow definition
11. Reference bus constraint
"""
function kkt(z::AbstractVector, prob::DCOPFProblem, d::AbstractVector)
    return kkt(z, prob.network, d)
end

function kkt(z::AbstractVector, net::DCNetwork, d::AbstractVector)
    n, m, k = net.n, net.m, net.k
    vars = unflatten_variables(z, net)

    # Extract variables
    θ, g, f, psh = vars.va, vars.pg, vars.f, vars.psh
    λ_lb, λ_ub = vars.lam_lb, vars.lam_ub
    γ_lb, γ_ub = vars.gamma_lb, vars.gamma_ub
    ρ_lb, ρ_ub = vars.rho_lb, vars.rho_ub
    μ_lb, μ_ub = vars.mu_lb, vars.mu_ub
    ν_bal, ν_flow = vars.nu_bal, vars.nu_flow
    η_ref = vars.η_ref

    # Construct matrices
    W = Diagonal(-net.b .* net.sw)
    B_mat = net.A' * W * net.A
    WA = W * net.A

    # Reference bus indicator
    e_ref = zeros(n)
    e_ref[net.ref_bus] = 1.0

    # KKT conditions
    # 1. Stationarity w.r.t. θ
    K_θ = B_mat' * ν_bal + WA' * ν_flow + e_ref * η_ref + net.A' * (net.sw .* (γ_ub - γ_lb))

    # 2. Stationarity w.r.t. g
    K_g = 2 * Diagonal(net.cq) * g + net.cl - net.G_inc' * ν_bal - ρ_lb + ρ_ub

    # 3. Stationarity w.r.t. f
    K_f = net.tau^2 * f - ν_flow - λ_lb + λ_ub

    # 4. Stationarity w.r.t. psh
    K_psh = net.c_shed - ν_bal - μ_lb + μ_ub

    # 5. Complementary slackness: flow bounds
    K_λ_lb = λ_lb .* (f + net.fmax)
    K_λ_ub = λ_ub .* (net.fmax - f)

    # 6. Complementary slackness: phase angle difference bounds
    Aθ = net.A * θ
    K_γ_lb = γ_lb .* net.sw .* (Aθ - net.angmin)
    K_γ_ub = γ_ub .* net.sw .* (net.angmax - Aθ)

    # 7. Complementary slackness: generation bounds
    K_ρ_lb = ρ_lb .* (g - net.gmin)
    K_ρ_ub = ρ_ub .* (net.gmax - g)

    # 8. Load shedding bounds
    # If max(d[i], 0) == 0, psh[i] is structurally fixed to zero and the
    # upper-bound dual is redundant. Replace the degenerate complementarity
    # pair by psh[i] = 0 and μ_ub[i] = 0 to avoid an exactly singular KKT block.
    K_μ_lb = similar(psh)
    K_μ_ub = similar(psh)
    @inbounds for i in 1:n
        if _is_fixed_zero_shed(d[i])
            K_μ_lb[i] = psh[i]
            K_μ_ub[i] = μ_ub[i]
        else
            K_μ_lb[i] = μ_lb[i] * psh[i]
            K_μ_ub[i] = μ_ub[i] * (_shed_capacity(d[i]) - psh[i])
        end
    end

    # 9. Primal feasibility: power balance (G_inc * g + psh - d = B * θ)
    K_power_bal = net.G_inc * g + psh - d - B_mat * θ

    # 10. Primal feasibility: flow definition
    K_flow_def = f - WA * θ

    # 11. Reference bus
    K_ref = θ[net.ref_bus]

    return vcat(K_θ, K_g, K_f, K_psh, K_λ_lb, K_λ_ub, K_γ_lb, K_γ_ub, K_ρ_lb, K_ρ_ub, K_μ_lb, K_μ_ub, K_power_bal, K_flow_def, [K_ref])
end

# =============================================================================
# KKT Jacobian
# =============================================================================

"""
    calc_kkt_jacobian(prob::DCOPFProblem; sol=nothing)

Compute the sparse Jacobian of the KKT operator analytically.

# Arguments
- `prob`: DCOPFProblem
- `sol`: Optional pre-computed solution. If not provided, ensures the problem is solved (reusing cached solution if available).

# Returns
Sparse matrix ∂K/∂z where z is the flattened variable vector.

This analytical Jacobian is more efficient than ForwardDiff for large problems.
"""
function calc_kkt_jacobian(prob::DCOPFProblem; sol::Union{DCOPFSolution,Nothing}=nothing)
    if isnothing(sol)
        sol = _ensure_solved!(prob)
    end
    return calc_kkt_jacobian(prob.network, prob.d, prob, sol)
end

@inline function _push_csc_entry!(rowval::Vector{Int}, nzval::Vector{Float64}, row::Int, val::Real)
    iszero(val) && return nothing
    push!(rowval, row)
    push!(nzval, Float64(val))
    return nothing
end

# SparseArrays provides fast column iteration for SparseMatrixCSC, but not row
# iteration on the lazy transpose. This cache gives row-wise access without
# materializing sparse(transpose(M)) in the KKT hot path.
function _sparse_row_storage(M::SparseMatrixCSC)
    nrow = size(M, 1)
    rows, cols, vals = findnz(M)
    counts = zeros(Int, nrow)
    @inbounds for row in rows
        counts[row] += 1
    end
    rowptr = Vector{Int}(undef, nrow + 1)
    rowptr[1] = 1
    @inbounds for i in 1:nrow
        rowptr[i + 1] = rowptr[i] + counts[i]
    end
    colind = Vector{Int}(undef, length(rows))
    rownzval = Vector{Float64}(undef, length(rows))
    next = copy(rowptr)
    @inbounds for p in eachindex(rows)
        dest = next[rows[p]]
        colind[dest] = cols[p]
        rownzval[dest] = vals[p]
        next[rows[p]] += 1
    end
    return rowptr, colind, rownzval
end

function calc_kkt_jacobian(net::DCNetwork, d::AbstractVector, prob::DCOPFProblem, sol::DCOPFSolution)
    # Use getfield in this hot path because these types overload getproperty
    # for Unicode aliases, which obscures concrete field types from inference.
    n = getfield(net, :n)
    m = getfield(net, :m)
    k = getfield(net, :k)
    dim = kkt_dims(n, m, k)
    A = getfield(net, :A)
    G_inc = getfield(net, :G_inc)
    b = getfield(net, :b)
    sw = getfield(net, :sw)
    fmax = getfield(net, :fmax)
    gmin = getfield(net, :gmin)
    gmax = getfield(net, :gmax)
    cq = getfield(net, :cq)
    angmin = getfield(net, :angmin)
    angmax = getfield(net, :angmax)
    ref_bus = getfield(net, :ref_bus)
    tau = getfield(net, :tau)
    va = getfield(sol, :va)
    pg = getfield(sol, :pg)
    f = getfield(sol, :f)
    psh = getfield(sol, :psh)
    lam_lb = getfield(sol, :lam_lb)
    lam_ub = getfield(sol, :lam_ub)
    gamma_lb = getfield(sol, :gamma_lb)
    gamma_ub = getfield(sol, :gamma_ub)
    rho_lb = getfield(sol, :rho_lb)
    rho_ub = getfield(sol, :rho_ub)
    mu_lb = getfield(sol, :mu_lb)
    mu_ub = getfield(sol, :mu_ub)

    # Construct matrices
    W = Diagonal(-b .* sw)
    B_mat = sparse(A' * W * A)

    # Build Jacobian blocks using centralized index calculation
    idx = kkt_indices(n, m, k)
    A_rowptr, A_rowcols, A_rowvals = _sparse_row_storage(A)
    G_rowptr, G_rowcols, G_rowvals = _sparse_row_storage(G_inc)
    rowval = Int[]
    nzval = Float64[]
    entry_hint = 4 * nnz(A) + 2 * nnz(B_mat) + 2 * nnz(G_inc) + 5n + 10m + 7k + 1
    sizehint!(rowval, entry_hint)
    sizehint!(nzval, entry_hint)
    colptr = Vector{Int}(undef, dim + 1)
    Aθ = A * va

    @inline function start_col!(col::Int)
        colptr[col] = length(rowval) + 1
        return nothing
    end

    # Column-wise CSC assembly of ∂K/∂z. The groups below mirror the block
    # formulas from the original assignment form while avoiding temporary
    # SparseMatrixCSC diagonals and structural setindex! insertions.

    # va columns:
    # ∂K_γ_lb/∂θ = Diag(γ_lb .* sw) * A
    # ∂K_γ_ub/∂θ = -Diag(γ_ub .* sw) * A
    # ∂K_power_bal/∂θ = -B
    # ∂K_flow_def/∂θ = -W*A, where W = Diag(-b .* sw)
    # ∂K_ref/∂θ_ref = 1
    @inbounds for j in 1:n
        start_col!(idx.va[j])
        for p in nzrange(A, j)
            e = rowvals(A)[p]
            aej = nonzeros(A)[p]
            _push_csc_entry!(rowval, nzval, idx.gamma_lb[e], gamma_lb[e] * sw[e] * aej)
        end
        for p in nzrange(A, j)
            e = rowvals(A)[p]
            aej = nonzeros(A)[p]
            _push_csc_entry!(rowval, nzval, idx.gamma_ub[e], -gamma_ub[e] * sw[e] * aej)
        end
        for p in nzrange(B_mat, j)
            _push_csc_entry!(rowval, nzval, idx.nu_bal[rowvals(B_mat)[p]], -nonzeros(B_mat)[p])
        end
        for p in nzrange(A, j)
            e = rowvals(A)[p]
            _push_csc_entry!(rowval, nzval, idx.nu_flow[e], b[e] * sw[e] * nonzeros(A)[p])
        end
        j == ref_bus && _push_csc_entry!(rowval, nzval, idx.η, 1.0)
    end

    # pg columns:
    # ∂K_g/∂g = 2*Diag(cq)
    # ∂K_ρ_lb/∂g = Diag(ρ_lb)
    # ∂K_ρ_ub/∂g = -Diag(ρ_ub)
    # ∂K_power_bal/∂g = G_inc
    @inbounds for j in 1:k
        start_col!(idx.pg[j])
        _push_csc_entry!(rowval, nzval, idx.pg[j], 2 * cq[j])
        _push_csc_entry!(rowval, nzval, idx.rho_lb[j], rho_lb[j])
        _push_csc_entry!(rowval, nzval, idx.rho_ub[j], -rho_ub[j])
        for p in nzrange(G_inc, j)
            _push_csc_entry!(rowval, nzval, idx.nu_bal[rowvals(G_inc)[p]], nonzeros(G_inc)[p])
        end
    end

    # f columns:
    # ∂K_f/∂f = τ²*I
    # ∂K_λ_lb/∂f = Diag(λ_lb)
    # ∂K_λ_ub/∂f = -Diag(λ_ub)
    # ∂K_flow_def/∂f = I
    @inbounds for e in 1:m
        start_col!(idx.f[e])
        _push_csc_entry!(rowval, nzval, idx.f[e], tau^2)
        _push_csc_entry!(rowval, nzval, idx.lam_lb[e], lam_lb[e])
        _push_csc_entry!(rowval, nzval, idx.lam_ub[e], -lam_ub[e])
        _push_csc_entry!(rowval, nzval, idx.nu_flow[e], 1.0)
    end

    # psh columns:
    # ∂K_μ_lb/∂psh = Diag(μ_lb), except fixed zero-shed buses use I
    # ∂K_μ_ub/∂psh = -Diag(μ_ub), except fixed zero-shed buses omit this term
    # ∂K_power_bal/∂psh = I
    @inbounds for i in 1:n
        start_col!(idx.psh[i])
        if _is_fixed_zero_shed(d[i])
            _push_csc_entry!(rowval, nzval, idx.mu_lb[i], 1.0)
        else
            _push_csc_entry!(rowval, nzval, idx.mu_lb[i], mu_lb[i])
            _push_csc_entry!(rowval, nzval, idx.mu_ub[i], -mu_ub[i])
        end
        _push_csc_entry!(rowval, nzval, idx.nu_bal[i], 1.0)
    end

    # lambda columns:
    # ∂K_f/∂λ_lb = -I
    # ∂K_λ_lb/∂λ_lb = Diag(f + fmax)
    # ∂K_f/∂λ_ub = I
    # ∂K_λ_ub/∂λ_ub = Diag(fmax - f)
    @inbounds for e in 1:m
        start_col!(idx.lam_lb[e])
        _push_csc_entry!(rowval, nzval, idx.f[e], -1.0)
        _push_csc_entry!(rowval, nzval, idx.lam_lb[e], f[e] + fmax[e])
    end
    @inbounds for e in 1:m
        start_col!(idx.lam_ub[e])
        _push_csc_entry!(rowval, nzval, idx.f[e], 1.0)
        _push_csc_entry!(rowval, nzval, idx.lam_ub[e], fmax[e] - f[e])
    end

    # gamma columns:
    # ∂K_θ/∂γ_lb = -A' * Diag(sw)
    # ∂K_γ_lb/∂γ_lb = Diag(sw .* (A*θ - angmin))
    # ∂K_θ/∂γ_ub = A' * Diag(sw)
    # ∂K_γ_ub/∂γ_ub = Diag(sw .* (angmax - A*θ))
    @inbounds for e in 1:m
        start_col!(idx.gamma_lb[e])
        for p in A_rowptr[e]:(A_rowptr[e + 1] - 1)
            _push_csc_entry!(rowval, nzval, idx.va[A_rowcols[p]], -sw[e] * A_rowvals[p])
        end
        _push_csc_entry!(rowval, nzval, idx.gamma_lb[e], sw[e] * (Aθ[e] - angmin[e]))
    end
    @inbounds for e in 1:m
        start_col!(idx.gamma_ub[e])
        for p in A_rowptr[e]:(A_rowptr[e + 1] - 1)
            _push_csc_entry!(rowval, nzval, idx.va[A_rowcols[p]], sw[e] * A_rowvals[p])
        end
        _push_csc_entry!(rowval, nzval, idx.gamma_ub[e], sw[e] * (angmax[e] - Aθ[e]))
    end

    # rho columns:
    # ∂K_g/∂ρ_lb = -I
    # ∂K_ρ_lb/∂ρ_lb = Diag(g - gmin)
    # ∂K_g/∂ρ_ub = I
    # ∂K_ρ_ub/∂ρ_ub = Diag(gmax - g)
    @inbounds for j in 1:k
        start_col!(idx.rho_lb[j])
        _push_csc_entry!(rowval, nzval, idx.pg[j], -1.0)
        _push_csc_entry!(rowval, nzval, idx.rho_lb[j], pg[j] - gmin[j])
    end
    @inbounds for j in 1:k
        start_col!(idx.rho_ub[j])
        _push_csc_entry!(rowval, nzval, idx.pg[j], 1.0)
        _push_csc_entry!(rowval, nzval, idx.rho_ub[j], gmax[j] - pg[j])
    end

    # mu columns:
    # ∂K_psh/∂μ_lb = -I
    # ∂K_μ_lb/∂μ_lb = Diag(psh), except fixed zero-shed buses omit this term
    # ∂K_psh/∂μ_ub = I
    # ∂K_μ_ub/∂μ_ub = Diag(max(d, 0) - psh), except fixed zero-shed buses use I
    @inbounds for i in 1:n
        start_col!(idx.mu_lb[i])
        _push_csc_entry!(rowval, nzval, idx.psh[i], -1.0)
        _is_fixed_zero_shed(d[i]) || _push_csc_entry!(rowval, nzval, idx.mu_lb[i], psh[i])
    end
    @inbounds for i in 1:n
        start_col!(idx.mu_ub[i])
        _push_csc_entry!(rowval, nzval, idx.psh[i], 1.0)
        if _is_fixed_zero_shed(d[i])
            _push_csc_entry!(rowval, nzval, idx.mu_ub[i], 1.0)
        else
            _push_csc_entry!(rowval, nzval, idx.mu_ub[i], _shed_capacity(d[i]) - psh[i])
        end
    end

    # nu_bal columns:
    # ∂K_θ/∂ν_bal = B'
    # ∂K_g/∂ν_bal = -G_inc'
    # ∂K_psh/∂ν_bal = -I
    @inbounds for i in 1:n
        start_col!(idx.nu_bal[i])
        for p in nzrange(B_mat, i)
            _push_csc_entry!(rowval, nzval, idx.va[rowvals(B_mat)[p]], nonzeros(B_mat)[p])
        end
        for p in G_rowptr[i]:(G_rowptr[i + 1] - 1)
            _push_csc_entry!(rowval, nzval, idx.pg[G_rowcols[p]], -G_rowvals[p])
        end
        _push_csc_entry!(rowval, nzval, idx.psh[i], -1.0)
    end

    # nu_flow columns:
    # ∂K_θ/∂ν_flow = (W*A)'
    # ∂K_f/∂ν_flow = -I
    @inbounds for e in 1:m
        start_col!(idx.nu_flow[e])
        for p in A_rowptr[e]:(A_rowptr[e + 1] - 1)
            _push_csc_entry!(rowval, nzval, idx.va[A_rowcols[p]], -b[e] * sw[e] * A_rowvals[p])
        end
        _push_csc_entry!(rowval, nzval, idx.f[e], -1.0)
    end

    # eta column:
    # ∂K_θ/∂η = e_ref
    start_col!(idx.η)
    _push_csc_entry!(rowval, nzval, idx.va[ref_bus], 1.0)
    colptr[dim + 1] = length(rowval) + 1

    return SparseMatrixCSC(dim, dim, colptr, rowval, nzval)
end

"""
    calc_kkt_jacobian_demand(net::DCNetwork, d::AbstractVector, sol::DCOPFSolution)

Compute the Jacobian of KKT conditions with respect to demand ∂K/∂d.

# Returns
Sparse matrix of size (kkt_dims × n).
"""
function calc_kkt_jacobian_demand(net::DCNetwork, d::AbstractVector, sol::DCOPFSolution)
    n = getfield(net, :n)
    m = getfield(net, :m)
    k = getfield(net, :k)
    dim = kkt_dims(n, m, k)
    idx = kkt_indices(n, m, k)
    mu_ub = getfield(sol, :mu_ub)

    colptr = Vector{Int}(undef, n + 1)
    rowval = Int[]
    nzval = Float64[]
    sizehint!(rowval, 2n)
    sizehint!(nzval, 2n)

    # ∂K_power_bal/∂d = -I and ∂K_μ_ub/∂d = Diag(μ_ub .* ∂max(d, 0)/∂d).
    @inbounds for i in 1:n
        colptr[i] = length(rowval) + 1
        shed_capacity_derivative = _shed_capacity_derivative(d[i])
        _push_csc_entry!(rowval, nzval, idx.mu_ub[i], mu_ub[i] * shed_capacity_derivative)
        _push_csc_entry!(rowval, nzval, idx.nu_bal[i], -1.0)
    end
    colptr[n + 1] = length(rowval) + 1

    return SparseMatrixCSC(dim, n, colptr, rowval, nzval)
end

"""
    calc_kkt_jacobian_demand_column(net, sol, j::Int) → Vector{Float64}

Compute column `j` of the KKT parameter Jacobian ∂K/∂d.
Only 1-2 nonzeros: always `nu_bal[j]`, plus `mu_ub[j]` if `d[j] > 0`.
"""
function calc_kkt_jacobian_demand_column(net::DCNetwork, d::AbstractVector, sol::DCOPFSolution, j::Int)
    col = zeros(kkt_dims(net))
    idx = kkt_indices(net)
    col[idx.nu_bal[j]] = -1.0
    col[idx.mu_ub[j]] = sol.mu_ub[j] * _shed_capacity_derivative(d[j])
    return col
end

# =============================================================================
# Topology (Switching) Sensitivity
# =============================================================================

"""
    calc_kkt_jacobian_switching(prob::DCOPFProblem, sol::DCOPFSolution)

Compute the Jacobian of KKT conditions with respect to switching variables ∂K/∂s.

The switching variable s ∈ [0,1]^m affects the susceptance-weighted Laplacian:
- W = Diagonal(-b .* s)
- B = A' * W * A
- Flow definition: f = W * A * θ

# Arguments
- `prob`: DCOPFProblem
- `sol`: Pre-computed solution

# Returns
Sparse matrix of size (kkt_dims × m).

# Notes
The switching variables s relaxes the binary line status to continuous values,
enabling gradient-based optimization for topology control.
"""
function calc_kkt_jacobian_switching(prob::DCOPFProblem, sol::DCOPFSolution)
    net = prob.network
    n, m, k = net.n, net.m, net.k
    dim = kkt_dims(net)

    θ = sol.va

    # Current switching state
    s = net.sw
    b = net.b
    A = net.A

    J_s = spzeros(dim, m)

    # Use centralized index calculation
    idx = kkt_indices(n, m, k)

    # Precompute A * θ once (invariant across branches)
    Aθ = A * θ

    ν_bal = sol.nu_bal
    ν_flow = sol.nu_flow

    for e in 1:m
        A_e_vec = Vector(A[e, :])  # dense once, reuse below
        Aθ_e = Aθ[e]  # scalar: phase angle difference across branch e

        # ∂K_power_bal/∂s_e = b_e * A[e,:]' * (A * θ)[e]
        J_s[idx.nu_bal, e] = b[e] * A_e_vec * Aθ_e

        # ∂K_flow_def/∂s_e: only row e is nonzero
        J_s[idx.nu_flow[e], e] = b[e] * Aθ_e

        # ∂K_θ/∂s_e = ∂B'/∂s_e * ν_bal + ∂(WA')/∂s_e * ν_flow
        Ae_dot_ν = dot(A_e_vec, ν_bal)
        J_s[idx.va, e] = -b[e] * A_e_vec * (Ae_dot_ν + ν_flow[e])

        # ∂K_θ/∂s_e from gated angle-difference bounds
        J_s[idx.va, e] += A_e_vec * (sol.gamma_ub[e] - sol.gamma_lb[e])

        # ∂K_γ/∂s_e from gated complementary slackness
        J_s[idx.gamma_lb[e], e] = sol.gamma_lb[e] * (Aθ_e - net.angmin[e])
        J_s[idx.gamma_ub[e], e] = sol.gamma_ub[e] * (net.angmax[e] - Aθ_e)
    end

    return J_s
end

"""
    _branch_bus_indices(A::AbstractMatrix, e::Int) → (f_bus::Int, t_bus::Int)

Extract the from-bus (+1) and to-bus (-1) indices from row `e` of the incidence matrix.
"""
function _branch_bus_indices(A::AbstractMatrix, e::Int)
    js, vs = findnz(A[e, :])
    f_bus = 0; t_bus = 0
    for (j, v) in zip(js, vs)
        if v > 0; f_bus = j; else; t_bus = j; end
    end
    return f_bus, t_bus
end

"""
    calc_kkt_jacobian_switching_column(prob, sol, e::Int) → Vector{Float64}

Compute column `e` of the KKT parameter Jacobian ∂K/∂sw.
~6 nonzeros from the incidence structure of branch `e`.
"""
function calc_kkt_jacobian_switching_column(prob::DCOPFProblem, sol::DCOPFSolution, e::Int)
    net = prob.network
    col = zeros(kkt_dims(net))
    idx = kkt_indices(net)
    A = net.A; b = net.b; θ = sol.va
    Aθ_e = dot(A[e, :], θ)
    f_bus, t_bus = _branch_bus_indices(A, e)

    # ∂K_power_bal/∂s_e: b[e] * A[e,:] * (A*θ)[e]
    col[idx.nu_bal[f_bus]] += b[e] * Aθ_e
    col[idx.nu_bal[t_bus]] -= b[e] * Aθ_e
    # ∂K_flow_def/∂s_e: only row e
    col[idx.nu_flow[e]] = b[e] * Aθ_e
    # ∂K_θ/∂s_e = -b[e] * A[e,:] * (dot(A[e,:], ν_bal) + ν_flow[e])
    coeff = -b[e] * (sol.nu_bal[f_bus] - sol.nu_bal[t_bus] + sol.nu_flow[e])
    col[idx.va[f_bus]] += coeff
    col[idx.va[t_bus]] -= coeff
    # gated angle-difference stationarity contribution
    coeff_ang = sol.gamma_ub[e] - sol.gamma_lb[e]
    col[idx.va[f_bus]] += coeff_ang
    col[idx.va[t_bus]] -= coeff_ang
    # gated angle-difference complementary slackness
    col[idx.gamma_lb[e]] = sol.gamma_lb[e] * (Aθ_e - net.angmin[e])
    col[idx.gamma_ub[e]] = sol.gamma_ub[e] * (net.angmax[e] - Aθ_e)
    return col
end

"""
    update_switching!(prob::DCOPFProblem, s::AbstractVector)

Update the network switching state, invalidate the sensitivity cache, and
rebuild the JuMP model so that `solve!(prob)` uses the new switching state.

# Arguments
- `prob`: DCOPFProblem to update
- `s`: New switching state vector (length m), values in [0,1]
"""
function update_switching!(prob::DCOPFProblem, s::AbstractVector)
    m = prob.network.m
    length(s) == m || throw(DimensionMismatch("Switching vector length $(length(s)) must match number of branches $m"))
    all(0 .<= s .<= 1) || throw(ArgumentError("Switching values must be in [0,1]"))

    # Invalidate all cached data including topology-dependent b_r_factor
    invalidate_topology!(prob.cache)

    # Update network switching state
    prob.network.sw .= s

    # Rebuild JuMP model with new switching coefficients
    _rebuild_jump_model!(prob)

    return prob
end
