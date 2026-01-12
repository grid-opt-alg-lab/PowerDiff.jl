# Susceptance Sensitivity Analysis for DC OPF
# Uses implicit differentiation via KKT conditions

using LinearAlgebra
using SparseArrays

"""
    calc_sensitivity_susceptance(prob::DCOPFProblem)

Compute sensitivity of DC OPF solution with respect to branch susceptances using implicit differentiation.

Uses the implicit function theorem on KKT conditions:
```
∂z/∂b = -(∂K/∂z)⁻¹ · (∂K/∂b)
```

where K(z, b) = 0 are the KKT conditions and z contains all primal/dual variables.

The susceptance b_e affects:
- Susceptance matrix: B = A' * Diag(-b .* z) * A
- Flow definition: f = W * A * θ where W = Diag(-b .* z)

# Returns
`SusceptanceSensitivity` containing Jacobian matrices:
- `dθ_db`: ∂θ/∂b (n × m) - phase angle sensitivity
- `dg_db`: ∂g/∂b (k × m) - generation sensitivity
- `df_db`: ∂f/∂b (m × m) - flow sensitivity
- `dlmp_db`: ∂LMP/∂b (n × m) - LMP sensitivity

# Example
```julia
prob = DCOPFProblem(net, d)
solve!(prob)
sens = calc_sensitivity_susceptance(prob)
# How does generation change when susceptance of line 1 changes?
dg_db1 = sens.dg_db[:, 1]
```
"""
function calc_sensitivity_susceptance(prob::DCOPFProblem)
    net = prob.network
    n, m, k = net.n, net.m, net.k

    # Compute KKT Jacobian ∂K/∂z
    J_z = calc_kkt_jacobian(prob)

    # Compute KKT Jacobian w.r.t. susceptance ∂K/∂b
    J_b = calc_kkt_jacobian_susceptance(prob)

    # Solve linear system: ∂z/∂b = -J_z⁻¹ * J_b
    dz_db = -(J_z \ Matrix(J_b))

    # Extract individual sensitivities from flattened result
    # Variable order: [θ(n), g(k), f(m), λ_lb(m), λ_ub(m), ρ_lb(k), ρ_ub(k), ν_bal(n), ν_flow(m), η_ref(1)]
    idx_θ = 1:n
    idx_g = n+1:n+k
    idx_f = n+k+1:n+k+m
    idx_ν_bal = n+k+3m+2k+1:2n+k+3m+2k

    dθ_db = dz_db[idx_θ, :]
    dg_db = dz_db[idx_g, :]
    df_db = dz_db[idx_f, :]
    dν_bal_db = dz_db[idx_ν_bal, :]

    # LMP sensitivity: LMP = ν_bal in B-θ formulation
    dlmp_db = dν_bal_db

    return SusceptanceSensitivity(
        Matrix(dθ_db),
        Matrix(dg_db),
        Matrix(df_db),
        Matrix(dlmp_db)
    )
end

"""
    calc_kkt_jacobian_susceptance(prob::DCOPFProblem)

Compute the Jacobian of KKT conditions with respect to susceptances ∂K/∂b.

# Returns
Sparse matrix of size (kkt_dims × m).

# Notes
Susceptance b affects:
- Susceptance matrix: B = A' * Diag(-b .* z) * A
- Weight matrix: W = Diag(-b .* z)
- Flow definition: f = W * A * θ

The affected KKT conditions are:
- K_θ = B' * ν_bal + (WA)' * ν_flow + e_ref * η_ref
- K_power_bal = G_inc * g - d - B * θ
- K_flow_def = f - W * A * θ

Derivatives:
- ∂B/∂b_e = -z_e * A[e,:]' * A[e,:]
- ∂(WA)/∂b_e: row e becomes -z_e * A[e,:]
"""
function calc_kkt_jacobian_susceptance(prob::DCOPFProblem)
    net = prob.network
    n, m, k = net.n, net.m, net.k
    dim = kkt_dims(net)

    # Get current solution
    sol = solve!(prob)
    θ = sol.θ
    ν_bal = sol.ν_bal
    ν_flow = dual.(prob.cons.flow_def)

    # Network parameters
    z = net.z
    b = net.b
    A = net.A

    J_b = spzeros(dim, m)

    # Row indices in KKT system:
    # K_θ (n), K_g (k), K_f (m), K_λ_lb (m), K_λ_ub (m), K_ρ_lb (k), K_ρ_ub (k), K_power_bal (n), K_flow_def (m), K_ref (1)
    idx_θ = 1:n
    idx_power_bal = n + k + 3m + 2k + 1 : n + k + 3m + 2k + n
    idx_flow_def = n + k + 3m + 2k + n + 1 : n + k + 3m + 2k + n + m

    for e in 1:m
        # For each branch e, compute ∂K/∂b_e
        A_e = A[e, :]
        A_e_vec = Vector(A_e[:])  # Convert to dense vector
        Aθ_e = (A * θ)[e]  # Phase angle difference across branch e

        # 1. ∂K_θ/∂b_e: K_θ = B' * ν_bal + (WA)' * ν_flow + e_ref * η_ref
        # ∂B'/∂b_e = -z_e * A[e,:]' * A[e,:] (symmetric)
        # ∂(WA')/∂b_e: column e becomes -z_e * A[e,:]'

        # Contribution from ν_bal through B:
        # ∂B/∂b_e * ν_bal = -z_e * A[e,:]' * (A[e,:] · ν_bal)
        Ae_dot_ν_bal = dot(A_e_vec, ν_bal)
        ∂K_θ_from_ν_bal = -z[e] * A_e_vec * Ae_dot_ν_bal

        # Contribution from ν_flow through WA':
        # ∂(WA')/∂b_e * ν_flow = -z_e * A[e,:]' * ν_flow[e]
        ∂K_θ_from_ν_flow = -z[e] * A_e_vec * ν_flow[e]

        J_b[idx_θ, e] = ∂K_θ_from_ν_bal + ∂K_θ_from_ν_flow

        # 2. ∂K_power_bal/∂b_e: K_power_bal = G_inc * g - d - B * θ
        # ∂K_power_bal/∂b_e = -∂B/∂b_e * θ = -(-z_e * A[e,:]' * A[e,:]) * θ
        #                    = z_e * A[e,:]' * (A[e,:] · θ)
        #                    = z_e * A_e_vec * Aθ_e
        ∂K_power_bal_∂b_e = z[e] * A_e_vec * Aθ_e
        J_b[idx_power_bal, e] = ∂K_power_bal_∂b_e

        # 3. ∂K_flow_def/∂b_e: K_flow_def = f - W * A * θ
        # ∂K_flow_def/∂b_e = -∂(WA)/∂b_e * θ
        # ∂(WA)/∂b_e * θ: row e is -z_e * A[e,:] * θ = -z_e * Aθ_e
        # So ∂K_flow_def/∂b_e: row e is z_e * Aθ_e (note the sign flip)
        ∂K_flow_def_∂b_e = spzeros(m)
        ∂K_flow_def_∂b_e[e] = z[e] * Aθ_e
        J_b[idx_flow_def, e] = ∂K_flow_def_∂b_e
    end

    return J_b
end
