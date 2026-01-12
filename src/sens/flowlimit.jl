# Flow Limit Sensitivity Analysis for DC OPF
# Uses implicit differentiation via KKT conditions
#
# NOTE: Sensitivity at exactly binding constraints can be numerically unstable
# due to near-singular complementary slackness terms. This is a known issue
# in optimization sensitivity analysis. The sensitivities are correct when:
# - Constraints are not binding (sensitivities are zero, as expected)
# - Constraints are strictly interior (no active inequalities)
# For binding constraints, consider using regularization or active-set methods.

using LinearAlgebra
using SparseArrays

"""
    calc_sensitivity_flowlimit(prob::DCOPFProblem) → OPFFlowLimitSens

Compute sensitivity of DC OPF solution with respect to flow limits using implicit differentiation.

Uses the implicit function theorem on KKT conditions:
```
dz/dfmax = -(dK/dz)^-1 * (dK/dfmax)
```

where K(z, fmax) = 0 are the KKT conditions and z contains all primal/dual variables.

# Returns
`OPFFlowLimitSens` containing Jacobian matrices:
- `dva_dfmax`: d(va)/d(fmax) (n x m) - voltage angle sensitivity
- `dg_dfmax`: dg/d(fmax) (k x m) - generation sensitivity
- `df_dfmax`: df/d(fmax) (m x m) - flow sensitivity
- `dlmp_dfmax`: d(LMP)/d(fmax) (n x m) - LMP sensitivity

# Example
```julia
prob = DCOPFProblem(net, d)
solve!(prob)
sens = calc_sensitivity_flowlimit(prob)
# How does generation change when flow limit on line 1 increases?
dg_dfmax1 = sens.dg_dfmax[:, 1]
```
"""
function calc_sensitivity_flowlimit(prob::DCOPFProblem)
    net = prob.network
    n, m, k = net.n, net.m, net.k

    # Solve once and reuse solution
    sol = solve!(prob)

    # Compute KKT Jacobian dK/dz (pass solution to avoid re-solving)
    J_z = calc_kkt_jacobian(prob; sol=sol)

    # Compute KKT Jacobian w.r.t. flow limits dK/dfmax
    J_fmax = calc_kkt_jacobian_flowlimit(prob, sol)

    # Solve linear system: dz/dfmax = -J_z^-1 * J_fmax
    dz_dfmax = -(J_z \ Matrix(J_fmax))

    # Extract individual sensitivities using centralized index calculation
    idx = kkt_indices(n, m, k)

    dva_dfmax = dz_dfmax[idx.θ, :]
    dg_dfmax = dz_dfmax[idx.g, :]
    df_dfmax = dz_dfmax[idx.f, :]
    dν_bal_dfmax = dz_dfmax[idx.ν_bal, :]

    # LMP sensitivity: LMP = ν_bal in B-θ formulation
    dlmp_dfmax = dν_bal_dfmax

    return OPFFlowLimitSens(
        Matrix(dva_dfmax),
        Matrix(dg_dfmax),
        Matrix(df_dfmax),
        Matrix(dlmp_dfmax)
    )
end

"""
    calc_kkt_jacobian_flowlimit(prob::DCOPFProblem, sol::DCOPFSolution)

Compute the Jacobian of KKT conditions with respect to flow limits dK/dfmax.

# Arguments
- `prob`: DCOPFProblem
- `sol`: Pre-computed solution

# Returns
Sparse matrix of size (kkt_dims x m).

# Notes
Flow limits fmax appear in the complementary slackness conditions:
- K_lambda_lb = lambda_lb .* (f + fmax)
- K_lambda_ub = lambda_ub .* (fmax - f)

Therefore:
- dK_lambda_lb/dfmax = Diag(lambda_lb)
- dK_lambda_ub/dfmax = Diag(lambda_ub)
"""
function calc_kkt_jacobian_flowlimit(prob::DCOPFProblem, sol::DCOPFSolution)
    net = prob.network
    n, m, k = net.n, net.m, net.k
    dim = kkt_dims(net)
    idx = kkt_indices(n, m, k)

    λ_lb = sol.λ_lb
    λ_ub = sol.λ_ub

    J_fmax = spzeros(dim, m)

    # ∂K_λ_lb/∂fmax = Diag(λ_lb)
    # K_λ_lb = λ_lb .* (f + fmax), so ∂K_λ_lb/∂fmax_e = λ_lb[e] for row e
    J_fmax[idx.λ_lb, :] = sparse(Diagonal(λ_lb))

    # ∂K_λ_ub/∂fmax = Diag(λ_ub)
    # K_λ_ub = λ_ub .* (fmax - f), so ∂K_λ_ub/∂fmax_e = λ_ub[e] for row e
    J_fmax[idx.λ_ub, :] = sparse(Diagonal(λ_ub))

    return J_fmax
end
