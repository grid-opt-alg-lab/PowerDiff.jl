# Cost Sensitivity Analysis for DC OPF
# Uses implicit differentiation via KKT conditions

using LinearAlgebra
using SparseArrays

"""
    calc_sensitivity_cost(prob::DCOPFProblem) → OPFCostSens

Compute sensitivity of DC OPF solution with respect to cost coefficients using implicit differentiation.

Uses the implicit function theorem on KKT conditions:
```
∂z/∂c = -(∂K/∂z)⁻¹ · (∂K/∂c)
```

where K(z, c) = 0 are the KKT conditions and z contains all primal/dual variables.

# Returns
`OPFCostSens` containing Jacobian matrices:
- `dg_dcq`: ∂g/∂cq (k × k) - generation sensitivity to quadratic cost
- `dg_dcl`: ∂g/∂cl (k × k) - generation sensitivity to linear cost
- `dlmp_dcq`: ∂LMP/∂cq (n × k) - LMP sensitivity to quadratic cost
- `dlmp_dcl`: ∂LMP/∂cl (n × k) - LMP sensitivity to linear cost

# Example
```julia
prob = DCOPFProblem(net, d)
solve!(prob)
sens = calc_sensitivity_cost(prob)
# How does generation change when linear cost of generator 1 increases?
dg_dc1 = sens.dg_dcl[:, 1]
```
"""
function calc_sensitivity_cost(prob::DCOPFProblem)
    net = prob.network
    n, m, k = net.n, net.m, net.k

    # Solve once and reuse solution
    sol = solve!(prob)

    # Compute KKT Jacobian ∂K/∂z (pass solution to avoid re-solving)
    J_z = calc_kkt_jacobian(prob; sol=sol)

    # Compute KKT Jacobians w.r.t. cost coefficients
    J_cl = calc_kkt_jacobian_cost_linear(net)
    J_cq = calc_kkt_jacobian_cost_quadratic(prob, sol)

    # Solve linear systems: ∂z/∂c = -J_z⁻¹ * J_c
    dz_dcl = -(J_z \ Matrix(J_cl))
    dz_dcq = -(J_z \ Matrix(J_cq))

    # Extract individual sensitivities using centralized index calculation
    idx = kkt_indices(n, m, k)

    dg_dcl = dz_dcl[idx.g, :]
    dg_dcq = dz_dcq[idx.g, :]

    dν_bal_dcl = dz_dcl[idx.ν_bal, :]
    dν_bal_dcq = dz_dcq[idx.ν_bal, :]

    dλ_lb_dcl = dz_dcl[idx.λ_lb, :]
    dλ_lb_dcq = dz_dcq[idx.λ_lb, :]
    dλ_ub_dcl = dz_dcl[idx.λ_ub, :]
    dλ_ub_dcq = dz_dcq[idx.λ_ub, :]

    # Compute LMP sensitivity
    # In the B-θ formulation, LMP = ν_bal (the power balance dual already
    # incorporates network topology through the Laplacian constraint)
    # Therefore: ∂LMP/∂c = ∂ν_bal/∂c
    dlmp_dcl = dν_bal_dcl
    dlmp_dcq = dν_bal_dcq

    return OPFCostSens(
        Matrix(dg_dcq),
        Matrix(dg_dcl),
        Matrix(dlmp_dcq),
        Matrix(dlmp_dcl)
    )
end

"""
    calc_kkt_jacobian_cost_linear(net::DCNetwork)

Compute the Jacobian of KKT conditions with respect to linear cost coefficients ∂K/∂cl.

# Returns
Sparse matrix of size (kkt_dims × k).

# Notes
Only the stationarity condition for g depends on cl:
  K_g = Cq * g + cl - G_inc' * ν_bal - ρ_lb + ρ_ub
  ∂K_g/∂cl = I_k (identity matrix)
"""
function calc_kkt_jacobian_cost_linear(net::DCNetwork)
    n, m, k = net.n, net.m, net.k
    dim = kkt_dims(net)
    idx = kkt_indices(n, m, k)

    J_cl = spzeros(dim, k)

    # ∂K_g/∂cl = I_k
    J_cl[idx.g, :] = sparse(I, k, k)

    return J_cl
end

"""
    calc_kkt_jacobian_cost_quadratic(prob::DCOPFProblem, sol::DCOPFSolution)

Compute the Jacobian of KKT conditions with respect to quadratic cost coefficients ∂K/∂cq.

# Arguments
- `prob`: DCOPFProblem
- `sol`: Pre-computed solution

# Returns
Sparse matrix of size (kkt_dims × k).

# Notes
Only the stationarity condition for g depends on cq:
  K_g = Cq * g + cl - G_inc' * ν_bal - ρ_lb + ρ_ub
  ∂K_g/∂cq_i = g_i (element i of generation vector)

So ∂K_g/∂cq = Diagonal(g) evaluated at the solution.
"""
function calc_kkt_jacobian_cost_quadratic(prob::DCOPFProblem, sol::DCOPFSolution)
    net = prob.network
    n, m, k = net.n, net.m, net.k
    dim = kkt_dims(net)
    idx = kkt_indices(n, m, k)

    g = sol.g

    J_cq = spzeros(dim, k)

    # ∂K_g/∂cq = Diagonal(g)
    # The (i, i) entry is g_i: derivative of cq_i * g_i with respect to cq_i is g_i
    J_cq[idx.g, :] = sparse(Diagonal(g))

    return J_cq
end
