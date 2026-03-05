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

# Tikhonov regularization magnitude for singular KKT Jacobians
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
                @warn "KKT Jacobian is singular; applying Tikhonov perturbation ($TIKHONOV_EPS)"
                J_reg = J_z + TIKHONOV_EPS * sparse(I, size(J_z, 1), size(J_z, 1))
                try
                    lu(J_reg)
                catch e2
                    error("KKT Jacobian remains singular after Tikhonov perturbation: $(e2)")
                end
            else
                rethrow(e)
            end
        end
    end
    return prob.cache.kkt_factor
end

# =============================================================================
# Cached Derivative Computation Functions
# =============================================================================

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
        return dz_dp[idx.nu_bal, :]
    else
        throw(ArgumentError("Unknown operand: $operand"))
    end
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
- Dual (inequality): lam_lb (m), lam_ub (m), rho_lb (k), rho_ub (k), mu_lb (n), mu_ub (n)
- Dual (equality): nu_bal (n), nu_flow (m)
- Reference bus constraint: 1

Total: 5n + 4m + 3k + 1
"""
kkt_dims(prob::DCOPFProblem) = kkt_dims(prob.network)

function kkt_dims(net::DCNetwork)
    n, m, k = net.n, net.m, net.k
    # va(n) + pg(k) + f(m) + psh(n) + lam_lb(m) + lam_ub(m) + rho_lb(k) + rho_ub(k) + mu_lb(n) + mu_ub(n) + nu_bal(n) + nu_flow(m) + ref(1)
    return 5n + 4m + 3k + 1
end

"""
    kkt_indices(n, m, k) → NamedTuple

Compute all KKT variable indices from network dimensions.
Single source of truth for index calculations.

# Variable ordering
[va(n), pg(k), f(m), psh(n), lam_lb(m), lam_ub(m), rho_lb(k), rho_ub(k), mu_lb(n), mu_ub(n), nu_bal(n), nu_flow(m), eta(1)]

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
[va; pg; f; psh; lam_lb; lam_ub; rho_lb; rho_ub; mu_lb; mu_ub; nu_bal; nu_flow; eta]

where eta is the dual for the reference bus constraint (set to 0).
"""
function flatten_variables(sol::DCOPFSolution, prob::DCOPFProblem)
    # Reference bus dual (typically not needed, set to 0)
    η_ref = dual(prob.cons.ref)

    return vcat(
        sol.va,
        sol.pg,
        sol.f,
        sol.psh,
        sol.lam_lb,
        sol.lam_ub,
        sol.rho_lb,
        sol.rho_ub,
        sol.mu_lb,
        sol.mu_ub,
        sol.nu_bal,
        sol.nu_flow,
        [η_ref]
    )
end

"""
    unflatten_variables(z::AbstractVector, prob::DCOPFProblem)

Unflatten KKT variable vector into named components.

# Returns
NamedTuple with fields: va, pg, f, psh, lam_lb, lam_ub, rho_lb, rho_ub, mu_lb, mu_ub, nu_bal, nu_flow, eta
"""
function unflatten_variables(z::AbstractVector, prob::DCOPFProblem)
    return unflatten_variables(z, prob.network)
end

function unflatten_variables(z::AbstractVector, net::DCNetwork)
    idx = kkt_indices(net)
    return (
        va = z[idx.va], pg = z[idx.pg], f = z[idx.f], psh = z[idx.psh],
        lam_lb = z[idx.lam_lb], lam_ub = z[idx.lam_ub],
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
     g ≥ gmin                       (ρ_lb)
     g ≤ gmax                       (ρ_ub)
     0 ≤ psh                        (μ_lb)
     psh ≤ d                        (μ_ub)
     θ[ref] = 0                     (η_ref)
```

# Returns
Vector of KKT residuals (should be zero at optimum):
1. Stationarity w.r.t. θ: B' * ν_bal + (W*A)' * ν_flow + e_ref * η_ref = 0
2. Stationarity w.r.t. g: 2*Cq * g + cl - G_inc' * ν_bal - ρ_lb + ρ_ub = 0
3. Stationarity w.r.t. f: τ² * f - ν_flow - λ_lb + λ_ub = 0
4. Stationarity w.r.t. psh: c_shed - ν_bal - μ_lb + μ_ub = 0
5. Complementary slackness for flow bounds
6. Complementary slackness for gen bounds
7. Complementary slackness for shedding bounds
8. Primal feasibility: power balance
9. Primal feasibility: flow definition
10. Reference bus constraint
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
    K_θ = B_mat' * ν_bal + WA' * ν_flow + e_ref * η_ref

    # 2. Stationarity w.r.t. g
    K_g = 2 * Diagonal(net.cq) * g + net.cl - net.G_inc' * ν_bal - ρ_lb + ρ_ub

    # 3. Stationarity w.r.t. f
    K_f = net.tau^2 * f - ν_flow - λ_lb + λ_ub

    # 4. Stationarity w.r.t. psh
    K_psh = net.c_shed - ν_bal - μ_lb + μ_ub

    # 5. Complementary slackness: flow bounds
    K_λ_lb = λ_lb .* (f + net.fmax)
    K_λ_ub = λ_ub .* (net.fmax - f)

    # 6. Complementary slackness: generation bounds
    K_ρ_lb = ρ_lb .* (g - net.gmin)
    K_ρ_ub = ρ_ub .* (net.gmax - g)

    # 7. Complementary slackness: load shedding bounds
    K_μ_lb = μ_lb .* psh
    K_μ_ub = μ_ub .* (d - psh)

    # 8. Primal feasibility: power balance (G_inc * g + psh - d = B * θ)
    K_power_bal = net.G_inc * g + psh - d - B_mat * θ

    # 9. Primal feasibility: flow definition
    K_flow_def = f - WA * θ

    # 10. Reference bus
    K_ref = θ[net.ref_bus]

    return vcat(K_θ, K_g, K_f, K_psh, K_λ_lb, K_λ_ub, K_ρ_lb, K_ρ_ub, K_μ_lb, K_μ_ub, K_power_bal, K_flow_def, [K_ref])
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

function calc_kkt_jacobian(net::DCNetwork, d::AbstractVector, prob::DCOPFProblem, sol::DCOPFSolution)
    n, m, k = net.n, net.m, net.k
    dim = kkt_dims(net)

    vars = (
        va = sol.va, pg = sol.pg, f = sol.f, psh = sol.psh,
        lam_lb = sol.lam_lb, lam_ub = sol.lam_ub,
        rho_lb = sol.rho_lb, rho_ub = sol.rho_ub,
        mu_lb = sol.mu_lb, mu_ub = sol.mu_ub,
        nu_bal = sol.nu_bal
    )

    # Construct matrices
    W = Diagonal(-net.b .* net.sw)
    B_mat = sparse(net.A' * W * net.A)
    WA = sparse(W * net.A)

    # Reference bus indicator
    e_ref = spzeros(n, 1)
    e_ref[net.ref_bus, 1] = 1.0

    # Build Jacobian blocks using centralized index calculation
    idx = kkt_indices(n, m, k)

    # Cache sparse identity matrices (reused across blocks)
    I_n = sparse(I, n, n)
    I_m = sparse(I, m, m)
    I_k = sparse(I, k, k)

    J = spzeros(dim, dim)

    # ∂K_θ/∂...
    # K_θ = B' * ν_bal + WA' * ν_flow + e_ref * η_ref
    J[idx.va, idx.nu_bal] = B_mat'
    J[idx.va, idx.nu_flow] = WA'
    J[idx.va, idx.η] = e_ref

    # ∂K_g/∂...
    # K_g = 2*Cq * g + cl - G_inc' * ν_bal - ρ_lb + ρ_ub
    J[idx.pg, idx.pg] = 2 * sparse(Diagonal(net.cq))
    J[idx.pg, idx.rho_lb] = -I_k
    J[idx.pg, idx.rho_ub] = I_k
    J[idx.pg, idx.nu_bal] = -net.G_inc'

    # ∂K_f/∂...
    # K_f = τ² * f - ν_flow - λ_lb + λ_ub
    J[idx.f, idx.f] = net.tau^2 * I_m
    J[idx.f, idx.lam_lb] = -I_m
    J[idx.f, idx.lam_ub] = I_m
    J[idx.f, idx.nu_flow] = -I_m

    # ∂K_psh/∂...
    # K_psh = c_shed - ν_bal - μ_lb + μ_ub
    J[idx.psh, idx.nu_bal] = -I_n
    J[idx.psh, idx.mu_lb] = -I_n
    J[idx.psh, idx.mu_ub] = I_n

    # ∂K_λ_lb/∂... (complementary slackness for lower flow bound)
    # K_λ_lb = λ_lb .* (f + fmax)
    J[idx.lam_lb, idx.f] = sparse(Diagonal(vars.lam_lb))
    J[idx.lam_lb, idx.lam_lb] = sparse(Diagonal(vars.f .+ net.fmax))

    # ∂K_λ_ub/∂... (complementary slackness for upper flow bound)
    # K_λ_ub = λ_ub .* (fmax - f)
    J[idx.lam_ub, idx.f] = -sparse(Diagonal(vars.lam_ub))
    J[idx.lam_ub, idx.lam_ub] = sparse(Diagonal(net.fmax .- vars.f))

    # ∂K_ρ_lb/∂... (complementary slackness for lower gen bound)
    # K_ρ_lb = ρ_lb .* (g - gmin)
    J[idx.rho_lb, idx.pg] = sparse(Diagonal(vars.rho_lb))
    J[idx.rho_lb, idx.rho_lb] = sparse(Diagonal(vars.pg .- net.gmin))

    # ∂K_ρ_ub/∂... (complementary slackness for upper gen bound)
    # K_ρ_ub = ρ_ub .* (gmax - g)
    J[idx.rho_ub, idx.pg] = -sparse(Diagonal(vars.rho_ub))
    J[idx.rho_ub, idx.rho_ub] = sparse(Diagonal(net.gmax .- vars.pg))

    # ∂K_μ_lb/∂... (complementary slackness for lower shedding bound)
    # K_μ_lb = μ_lb .* psh
    J[idx.mu_lb, idx.psh] = sparse(Diagonal(vars.mu_lb))
    J[idx.mu_lb, idx.mu_lb] = sparse(Diagonal(vars.psh))

    # ∂K_μ_ub/∂... (complementary slackness for upper shedding bound)
    # K_μ_ub = μ_ub .* (d - psh)
    J[idx.mu_ub, idx.psh] = -sparse(Diagonal(vars.mu_ub))
    J[idx.mu_ub, idx.mu_ub] = sparse(Diagonal(d .- vars.psh))

    # ∂K_power_bal/∂... (primal feasibility: power balance)
    # K_power_bal = G_inc * g + psh - d - B * θ
    J[idx.nu_bal, idx.va] = -B_mat
    J[idx.nu_bal, idx.pg] = net.G_inc
    J[idx.nu_bal, idx.psh] = I_n

    # ∂K_flow_def/∂... (primal feasibility: flow definition)
    # K_flow_def = f - WA * θ
    J[idx.nu_flow, idx.va] = -WA
    J[idx.nu_flow, idx.f] = I_m

    # ∂K_ref/∂θ (reference bus)
    J[idx.η, net.ref_bus] = 1.0

    return J
end

"""
    calc_kkt_jacobian_demand(net::DCNetwork, d::AbstractVector, sol::DCOPFSolution)

Compute the Jacobian of KKT conditions with respect to demand ∂K/∂d.

# Returns
Sparse matrix of size (kkt_dims × n).
"""
function calc_kkt_jacobian_demand(net::DCNetwork, d::AbstractVector, sol::DCOPFSolution)
    n, m, k = net.n, net.m, net.k
    dim = kkt_dims(net)
    idx = kkt_indices(n, m, k)

    J_d = spzeros(dim, n)

    # ∂K_power_bal/∂d = -I (from K_power_bal = G_inc * g + psh - d - B * θ)
    J_d[idx.nu_bal, :] = -sparse(I, n, n)

    # ∂K_μ_ub/∂d = Diag(μ_ub) (from K_μ_ub = μ_ub .* (d - psh))
    J_d[idx.mu_ub, :] = sparse(Diagonal(sol.mu_ub))

    return J_d
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
    end

    return J_s
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

    # Invalidate sensitivity cache since parameters changed
    invalidate!(prob.cache)

    # Update network switching state
    prob.network.sw .= s

    # Rebuild JuMP model with new switching coefficients
    _rebuild_jump_model!(prob)

    return prob
end
