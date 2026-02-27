# =============================================================================
# KKT System for DC OPF
# =============================================================================
#
# Implements KKT conditions for implicit differentiation of DC OPF solutions.

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
function _ensure_kkt_factor!(prob::DCOPFProblem)::LinearAlgebra.LU
    if isnothing(prob.cache.kkt_factor)
        sol = _ensure_solved!(prob)
        J_z = calc_kkt_jacobian(prob; sol=sol)
        prob.cache.kkt_factor = lu(Matrix(J_z))
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
        J_d = calc_kkt_jacobian_demand(prob.network)
        prob.cache.dz_dd = -(kkt_lu \ Matrix(J_d))
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
        prob.cache.dz_dcl = -(kkt_lu \ Matrix(J_cl))
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
        prob.cache.dz_dcq = -(kkt_lu \ Matrix(J_cq))
    end
    return prob.cache.dz_dcq
end

"""
    _get_dz_dz!(prob::DCOPFProblem) → Matrix{Float64}

Get or compute the full KKT derivative matrix w.r.t. switching.
"""
function _get_dz_dz!(prob::DCOPFProblem)::Matrix{Float64}
    if isnothing(prob.cache.dz_dz)
        kkt_lu = _ensure_kkt_factor!(prob)
        sol = _ensure_solved!(prob)
        J_s = calc_kkt_jacobian_switching(prob, sol)
        prob.cache.dz_dz = -(kkt_lu \ Matrix(J_s))
    end
    return prob.cache.dz_dz
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
        prob.cache.dz_dfmax = -(kkt_lu \ Matrix(J_fmax))
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
        prob.cache.dz_db = -(kkt_lu \ Matrix(J_b))
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
- `operand`: Which operand to extract (:va, :pg, :f, :lmp)
"""
function _extract_sensitivity(prob::DCOPFProblem, dz_dp::Matrix{Float64}, operand::Symbol)::Matrix{Float64}
    idx = kkt_indices(prob)

    if operand === :va || operand === :θ
        return Matrix(dz_dp[idx.θ, :])
    elseif operand === :pg || operand === :g
        return Matrix(dz_dp[idx.g, :])
    elseif operand === :f
        return Matrix(dz_dp[idx.f, :])
    elseif operand === :lmp || operand === :ν_bal
        # LMP = ν_bal in B-θ formulation
        return Matrix(dz_dp[idx.ν_bal, :])
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
- Primal: θ (n), g (k), f (m)
- Dual (inequality): λ_lb (m), λ_ub (m), ρ_lb (k), ρ_ub (k)
- Dual (equality): ν_bal (n), ν_flow (m)
- Reference bus constraint: 1

Total: n + k + m + 2m + 2k + n + m + 1 = 2n + 4m + 3k + 1
"""
kkt_dims(prob::DCOPFProblem) = kkt_dims(prob.network)

function kkt_dims(net::DCNetwork)
    n, m, k = net.n, net.m, net.k
    # θ(n) + g(k) + f(m) + λ_lb(m) + λ_ub(m) + ρ_lb(k) + ρ_ub(k) + ν_bal(n) + ν_flow(m) + ref(1)
    return 2n + 4m + 3k + 1
end

"""
    kkt_indices(n, m, k) → NamedTuple

Compute all KKT variable indices from network dimensions.
Single source of truth for index calculations.

# Variable ordering
[θ(n), g(k), f(m), λ_lb(m), λ_ub(m), ρ_lb(k), ρ_ub(k), ν_bal(n), ν_flow(m), η_ref(1)]

# Returns
NamedTuple with index ranges for each variable block.
"""
function kkt_indices(n::Int, m::Int, k::Int)
    i = 0
    idx_θ = (i+1):(i+n); i += n
    idx_g = (i+1):(i+k); i += k
    idx_f = (i+1):(i+m); i += m
    idx_λ_lb = (i+1):(i+m); i += m
    idx_λ_ub = (i+1):(i+m); i += m
    idx_ρ_lb = (i+1):(i+k); i += k
    idx_ρ_ub = (i+1):(i+k); i += k
    idx_ν_bal = (i+1):(i+n); i += n
    idx_ν_flow = (i+1):(i+m); i += m
    idx_η = i + 1

    return (
        θ = idx_θ, g = idx_g, f = idx_f,
        λ_lb = idx_λ_lb, λ_ub = idx_λ_ub,
        ρ_lb = idx_ρ_lb, ρ_ub = idx_ρ_ub,
        ν_bal = idx_ν_bal, ν_flow = idx_ν_flow, η = idx_η
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
[θ; g; f; λ_lb; λ_ub; ρ_lb; ρ_ub; ν_bal; ν_flow; η_ref]

where η_ref is the dual for the reference bus constraint (set to 0).
"""
function flatten_variables(sol::DCOPFSolution, prob::DCOPFProblem)
    # Reference bus dual (typically not needed, set to 0)
    η_ref = dual(prob.cons.ref)

    return vcat(
        sol.θ,
        sol.g,
        sol.f,
        sol.λ_lb,
        sol.λ_ub,
        sol.ρ_lb,
        sol.ρ_ub,
        sol.ν_bal,
        sol.ν_flow,
        [η_ref]
    )
end

"""
    unflatten_variables(z::AbstractVector, prob::DCOPFProblem)

Unflatten KKT variable vector into named components.

# Returns
NamedTuple with fields: θ, g, f, λ_lb, λ_ub, ρ_lb, ρ_ub, ν_bal, ν_flow, η_ref
"""
function unflatten_variables(z::AbstractVector, prob::DCOPFProblem)
    return unflatten_variables(z, prob.network)
end

function unflatten_variables(z::AbstractVector, net::DCNetwork)
    n, m, k = net.n, net.m, net.k

    i = 0
    θ = z[i+1:i+n]; i += n
    g = z[i+1:i+k]; i += k
    f = z[i+1:i+m]; i += m
    λ_lb = z[i+1:i+m]; i += m
    λ_ub = z[i+1:i+m]; i += m
    ρ_lb = z[i+1:i+k]; i += k
    ρ_ub = z[i+1:i+k]; i += k
    ν_bal = z[i+1:i+n]; i += n
    ν_flow = z[i+1:i+m]; i += m
    η_ref = z[i+1]

    return (
        θ = θ,
        g = g,
        f = f,
        λ_lb = λ_lb,
        λ_ub = λ_ub,
        ρ_lb = ρ_lb,
        ρ_ub = ρ_ub,
        ν_bal = ν_bal,
        ν_flow = ν_flow,
        η_ref = η_ref
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
min  g' Cq g + cl' g + (τ²/2) ||f||²
s.t. G_inc * g - d = B * θ     (ν_bal)
     f = W * A * θ              (ν_flow)
     f ≥ -fmax                  (λ_lb)
     f ≤ fmax                   (λ_ub)
     g ≥ gmin                   (ρ_lb)
     g ≤ gmax                   (ρ_ub)
     θ[ref] = 0                 (η_ref)
```

# Returns
Vector of KKT residuals (should be zero at optimum):
1. Stationarity w.r.t. θ: B' * ν_bal + (W*A)' * ν_flow + e_ref * η_ref = 0
2. Stationarity w.r.t. g: 2*Cq * g + cl - G_inc' * ν_bal - ρ_lb + ρ_ub = 0
3. Stationarity w.r.t. f: τ² * f - ν_flow - λ_lb + λ_ub = 0
4. Complementary slackness for flow bounds
5. Complementary slackness for gen bounds
6. Primal feasibility: power balance
7. Primal feasibility: flow definition
8. Reference bus constraint
"""
function kkt(z::AbstractVector, prob::DCOPFProblem, d::AbstractVector)
    return kkt(z, prob.network, d)
end

function kkt(z::AbstractVector, net::DCNetwork, d::AbstractVector)
    n, m, k = net.n, net.m, net.k
    vars = unflatten_variables(z, net)

    # Extract variables
    θ, g, f = vars.θ, vars.g, vars.f
    λ_lb, λ_ub = vars.λ_lb, vars.λ_ub
    ρ_lb, ρ_ub = vars.ρ_lb, vars.ρ_ub
    ν_bal, ν_flow = vars.ν_bal, vars.ν_flow
    η_ref = vars.η_ref

    # Construct matrices
    W = Diagonal(-net.b .* net.z)
    B_mat = net.A' * W * net.A
    WA = W * net.A

    # Reference bus indicator
    e_ref = zeros(n)
    e_ref[net.ref_bus] = 1.0

    # KKT conditions
    # 1. Stationarity w.r.t. θ
    K_θ = B_mat' * ν_bal + WA' * ν_flow + e_ref * η_ref

    # 2. Stationarity w.r.t. g
    # Objective is cq_i * g_i^2 + cl_i * g_i, so ∂obj/∂g_i = 2*cq_i*g_i + cl_i
    K_g = 2 * Diagonal(net.cq) * g + net.cl - net.G_inc' * ν_bal - ρ_lb + ρ_ub

    # 3. Stationarity w.r.t. f
    K_f = net.τ^2 * f - ν_flow - λ_lb + λ_ub

    # 4. Complementary slackness: flow bounds
    K_λ_lb = λ_lb .* (f + net.fmax)
    K_λ_ub = λ_ub .* (net.fmax - f)

    # 5. Complementary slackness: generation bounds
    K_ρ_lb = ρ_lb .* (g - net.gmin)
    K_ρ_ub = ρ_ub .* (net.gmax - g)

    # 6. Primal feasibility: power balance
    K_power_bal = net.G_inc * g - d - B_mat * θ

    # 7. Primal feasibility: flow definition
    K_flow_def = f - WA * θ

    # 8. Reference bus
    K_ref = θ[net.ref_bus]

    return vcat(K_θ, K_g, K_f, K_λ_lb, K_λ_ub, K_ρ_lb, K_ρ_ub, K_power_bal, K_flow_def, [K_ref])
end

# =============================================================================
# KKT Jacobian
# =============================================================================

"""
    calc_kkt_jacobian(prob::DCOPFProblem; sol=nothing)

Compute the sparse Jacobian of the KKT operator analytically.

# Arguments
- `prob`: DCOPFProblem
- `sol`: Optional pre-computed solution. If not provided, calls solve!(prob).

# Returns
Sparse matrix ∂K/∂z where z is the flattened variable vector.

This analytical Jacobian is more efficient than ForwardDiff for large problems.
"""
function calc_kkt_jacobian(prob::DCOPFProblem; sol::Union{DCOPFSolution,Nothing}=nothing)
    if isnothing(sol)
        sol = solve!(prob)
    end
    return calc_kkt_jacobian(prob.network, prob.d, prob, sol)
end

function calc_kkt_jacobian(net::DCNetwork, d::AbstractVector, prob::DCOPFProblem, sol::DCOPFSolution)
    n, m, k = net.n, net.m, net.k
    dim = kkt_dims(net)

    vars = (
        θ = sol.θ, g = sol.g, f = sol.f,
        λ_lb = sol.λ_lb, λ_ub = sol.λ_ub,
        ρ_lb = sol.ρ_lb, ρ_ub = sol.ρ_ub,
        ν_bal = sol.ν_bal
    )

    # Construct matrices
    W = Diagonal(-net.b .* net.z)
    B_mat = sparse(net.A' * W * net.A)
    WA = sparse(W * net.A)

    # Reference bus indicator
    e_ref = spzeros(n, 1)
    e_ref[net.ref_bus, 1] = 1.0

    # Build Jacobian blocks using centralized index calculation
    idx = kkt_indices(n, m, k)

    J = spzeros(dim, dim)

    # ∂K_θ/∂... (row block 1: indices 1:n)
    # K_θ = B' * ν_bal + WA' * ν_flow + e_ref * η_ref
    J[idx.θ, idx.ν_bal] = B_mat'
    J[idx.θ, idx.ν_flow] = WA'
    J[idx.θ, idx.η] = e_ref

    # ∂K_g/∂... (row block 2: indices n+1:n+k)
    # K_g = 2*Cq * g + cl - G_inc' * ν_bal - ρ_lb + ρ_ub
    J[idx.g, idx.g] = 2 * sparse(Diagonal(net.cq))
    J[idx.g, idx.ρ_lb] = -sparse(I, k, k)
    J[idx.g, idx.ρ_ub] = sparse(I, k, k)
    J[idx.g, idx.ν_bal] = -net.G_inc'

    # ∂K_f/∂... (row block 3: indices n+k+1:n+k+m)
    # K_f = τ² * f - ν_flow - λ_lb + λ_ub
    J[idx.f, idx.f] = net.τ^2 * sparse(I, m, m)
    J[idx.f, idx.λ_lb] = -sparse(I, m, m)
    J[idx.f, idx.λ_ub] = sparse(I, m, m)
    J[idx.f, idx.ν_flow] = -sparse(I, m, m)

    # ∂K_λ_lb/∂... (complementary slackness for lower flow bound)
    # K_λ_lb = λ_lb .* (f + fmax)
    J[idx.λ_lb, idx.f] = sparse(Diagonal(vars.λ_lb))
    J[idx.λ_lb, idx.λ_lb] = sparse(Diagonal(vars.f .+ net.fmax))

    # ∂K_λ_ub/∂... (complementary slackness for upper flow bound)
    # K_λ_ub = λ_ub .* (fmax - f)
    J[idx.λ_ub, idx.f] = -sparse(Diagonal(vars.λ_ub))
    J[idx.λ_ub, idx.λ_ub] = sparse(Diagonal(net.fmax .- vars.f))

    # ∂K_ρ_lb/∂... (complementary slackness for lower gen bound)
    # K_ρ_lb = ρ_lb .* (g - gmin)
    J[idx.ρ_lb, idx.g] = sparse(Diagonal(vars.ρ_lb))
    J[idx.ρ_lb, idx.ρ_lb] = sparse(Diagonal(vars.g .- net.gmin))

    # ∂K_ρ_ub/∂... (complementary slackness for upper gen bound)
    # K_ρ_ub = ρ_ub .* (gmax - g)
    J[idx.ρ_ub, idx.g] = -sparse(Diagonal(vars.ρ_ub))
    J[idx.ρ_ub, idx.ρ_ub] = sparse(Diagonal(net.gmax .- vars.g))

    # ∂K_power_bal/∂... (primal feasibility: power balance)
    # K_power_bal = G_inc * g - d - B * θ
    J[idx.ν_bal, idx.θ] = -B_mat
    J[idx.ν_bal, idx.g] = net.G_inc

    # ∂K_flow_def/∂... (primal feasibility: flow definition)
    # K_flow_def = f - WA * θ
    J[idx.ν_flow, idx.θ] = -WA
    J[idx.ν_flow, idx.f] = sparse(I, m, m)

    # ∂K_ref/∂θ (reference bus)
    J[idx.η, net.ref_bus] = 1.0

    return J
end

"""
    calc_kkt_jacobian_demand(net::DCNetwork)

Compute the Jacobian of KKT conditions with respect to demand ∂K/∂d.

# Returns
Sparse matrix of size (kkt_dims × n).
"""
function calc_kkt_jacobian_demand(net::DCNetwork)
    n, m, k = net.n, net.m, net.k
    dim = kkt_dims(net)
    idx = kkt_indices(n, m, k)

    # ∂K/∂d only affects the power balance equation: K_power_bal = G_inc * g - d - B * θ
    # ∂K_power_bal/∂d = -I
    J_d = spzeros(dim, n)
    J_d[idx.ν_bal, :] = -sparse(I, n, n)

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

    θ = sol.θ

    # Current switching state
    s = net.z
    b = net.b
    A = net.A

    J_s = spzeros(dim, m)

    # Use centralized index calculation
    idx = kkt_indices(n, m, k)

    # ∂W/∂s_e = Diagonal with -b_e at position (e,e)
    # ∂B/∂s_e = A' * ∂W/∂s_e * A = -b_e * A[e,:]' * A[e,:]
    # ∂(WA)/∂s_e = ∂W/∂s_e * A = -b_e * e_e * A[e,:]  (only row e changes)

    for e in 1:m
        # For each branch e, compute ∂K/∂s_e

        # 1. ∂K_θ/∂s_e: K_θ = B' * ν_bal + WA' * ν_flow + e_ref * η_ref
        # ∂B'/∂s_e * ν_bal + ∂(WA')/∂s_e * ν_flow
        # Note: B' = B (symmetric), so ∂B'/∂s_e = ∂B/∂s_e
        # ∂B/∂s_e = -b_e * A[e,:]' * A[e,:]
        # ∂(WA')/∂s_e = (∂(WA)/∂s_e)' where ∂(WA)/∂s_e has row e = -b_e * A[e,:]
        # So ∂(WA')/∂s_e has column e = -b_e * A[e,:]'

        # For K_θ: contribution from ν_bal through B depends on current ν_bal values
        # For K_θ: contribution from ν_flow through WA' depends on current ν_flow values
        # These involve ∂K_θ/∂s_e = ∂B/∂s_e * ν_bal + ∂(WA')/∂s_e * ν_flow
        # But we need to evaluate at current solution...

        # Actually, for sensitivity analysis via implicit function theorem,
        # we need ∂K/∂s evaluated at the solution, treating primal/dual vars as fixed.

        # 2. ∂K_power_bal/∂s_e: K_power_bal = G_inc * g - d - B * θ
        # ∂K_power_bal/∂s_e = -∂B/∂s_e * θ = -(-b_e * A[e,:]' * A[e,:]) * θ
        #                    = b_e * (A[e,:]' * (A[e,:] * θ))
        #                    = b_e * A[e,:]' * (A * θ)[e]
        A_e = A[e, :]  # 1×n sparse row
        A_e_vec = Vector(A_e[:])  # Convert to dense vector
        Aθ_e = (A * θ)[e]  # scalar: phase angle difference across branch e
        ∂K_power_bal_∂s_e = b[e] * A_e_vec * Aθ_e  # n×1 vector (scalar times vector)
        J_s[idx.ν_bal, e] = ∂K_power_bal_∂s_e

        # 3. ∂K_flow_def/∂s_e: K_flow_def = f - WA * θ
        # ∂K_flow_def/∂s_e = -∂(WA)/∂s_e * θ
        # ∂(WA)/∂s_e * θ: row e is -b_e * A[e,:] * θ = -b_e * Aθ_e
        # All other rows are 0
        ∂K_flow_def_∂s_e = spzeros(m)
        ∂K_flow_def_∂s_e[e] = b[e] * Aθ_e  # Note: -(-b_e * Aθ_e) = b_e * Aθ_e
        J_s[idx.ν_flow, e] = ∂K_flow_def_∂s_e

        # 4. K_θ also depends on s through B and WA affecting the stationarity conditions
        # K_θ = B' * ν_bal + WA' * ν_flow + e_ref * η_ref
        # But B and WA depend on s, so:
        # ∂K_θ/∂s_e = ∂B'/∂s_e * ν_bal + ∂(WA')/∂s_e * ν_flow
        # However, for implicit differentiation, we treat duals as variables, not functions of s.
        # So ∂K_θ/∂s_e at fixed duals is computed as above.
        ν_bal = sol.ν_bal
        ν_flow = sol.ν_flow

        # ∂B'/∂s_e = -b_e * A[e,:]' * A[e,:]  (this is symmetric, same as ∂B/∂s_e)
        # For the outer product, we need: -b_e * (A[e,:] ⋅ ν_bal) * A[e,:]'
        # Because (A[e,:]' * A[e,:]) * ν_bal = A[e,:]' * (A[e,:] ⋅ ν_bal)
        A_e_vec = Vector(A_e[:])  # Convert to dense vector for computation
        Ae_dot_ν = dot(A_e_vec, ν_bal)  # scalar
        ∂K_θ_from_ν_bal = -b[e] * A_e_vec * Ae_dot_ν  # n×1 vector

        # ∂(WA')/∂s_e affects only column e: column e becomes -b_e * A[e,:]'
        # So ∂(WA')/∂s_e * ν_flow = -b_e * A[e,:]' * ν_flow[e]
        ∂K_θ_from_ν_flow = -b[e] * A_e_vec * ν_flow[e]  # n×1

        J_s[idx.θ, e] = ∂K_θ_from_ν_bal + ∂K_θ_from_ν_flow
    end

    return J_s
end

"""
    update_switching!(prob::DCOPFProblem, s::AbstractVector)

Update the network switching state and invalidate the sensitivity cache.

# Arguments
- `prob`: DCOPFProblem to update
- `s`: New switching state vector (length m), values in [0,1]

# Note
This modifies `prob.network.z` and invalidates cached sensitivities.
The JuMP model is not rebuilt; re-solving will use the new switching state
for KKT-based sensitivity analysis.
"""
function update_switching!(prob::DCOPFProblem, s::AbstractVector)
    m = prob.network.m
    @assert length(s) == m "Switching vector length must match number of branches"
    @assert all(0 .<= s .<= 1) "Switching values must be in [0,1]"

    # Invalidate sensitivity cache since parameters changed
    invalidate!(prob.cache)

    # Update network switching state
    prob.network.z .= s

    # Rebuild the susceptance matrix and update constraints
    W = Diagonal(-prob.network.b .* prob.network.z)
    B_mat = sparse(prob.network.A' * W * prob.network.A)

    # Note: Full problem rebuild would be needed for JuMP model update
    # For now, this updates the network parameters; re-solve will use new values
    return prob
end
