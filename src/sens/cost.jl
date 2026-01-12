# Cost Sensitivity Analysis for DC OPF
# Uses implicit differentiation via KKT conditions

using LinearAlgebra
using SparseArrays

"""
    calc_sensitivity_cost(prob::DCOPFProblem)

Compute sensitivity of DC OPF solution with respect to cost coefficients using implicit differentiation.

Uses the implicit function theorem on KKT conditions:
```
∂z/∂c = -(∂K/∂z)⁻¹ · (∂K/∂c)
```

where K(z, c) = 0 are the KKT conditions and z contains all primal/dual variables.

# Returns
`CostSensitivity` containing Jacobian matrices:
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

    # Compute KKT Jacobian ∂K/∂z
    J_z = calc_kkt_jacobian(prob)

    # Compute KKT Jacobians w.r.t. cost coefficients
    J_cl = calc_kkt_jacobian_cost_linear(net)
    J_cq = calc_kkt_jacobian_cost_quadratic(prob)

    # Solve linear systems: ∂z/∂c = -J_z⁻¹ * J_c
    dz_dcl = -(J_z \ Matrix(J_cl))
    dz_dcq = -(J_z \ Matrix(J_cq))

    # Extract individual sensitivities from flattened result
    # Variable order: [θ(n), g(k), f(m), λ_lb(m), λ_ub(m), ρ_lb(k), ρ_ub(k), ν_bal(n), ν_flow(m), η_ref(1)]
    idx_g = n+1:n+k
    idx_ν_bal = n+k+3m+2k+1:2n+k+3m+2k
    idx_λ_lb = n+k+m+1:n+k+2m
    idx_λ_ub = n+k+2m+1:n+k+3m

    dg_dcl = dz_dcl[idx_g, :]
    dg_dcq = dz_dcq[idx_g, :]

    dν_bal_dcl = dz_dcl[idx_ν_bal, :]
    dν_bal_dcq = dz_dcq[idx_ν_bal, :]

    dλ_lb_dcl = dz_dcl[idx_λ_lb, :]
    dλ_lb_dcq = dz_dcq[idx_λ_lb, :]
    dλ_ub_dcl = dz_dcl[idx_λ_ub, :]
    dλ_ub_dcq = dz_dcq[idx_λ_ub, :]

    # Compute LMP sensitivity
    # In the B-θ formulation, LMP = ν_bal (the power balance dual already
    # incorporates network topology through the Laplacian constraint)
    # Therefore: ∂LMP/∂c = ∂ν_bal/∂c
    dlmp_dcl = dν_bal_dcl
    dlmp_dcq = dν_bal_dcq

    return CostSensitivity(
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

    J_cl = spzeros(dim, k)

    # K_g row indices: n+1:n+k (after K_θ which is 1:n)
    idx_g = n+1:n+k

    # ∂K_g/∂cl = I_k
    J_cl[idx_g, :] = sparse(I, k, k)

    return J_cl
end

"""
    calc_kkt_jacobian_cost_quadratic(prob::DCOPFProblem)

Compute the Jacobian of KKT conditions with respect to quadratic cost coefficients ∂K/∂cq.

# Returns
Sparse matrix of size (kkt_dims × k).

# Notes
Only the stationarity condition for g depends on cq:
  K_g = Cq * g + cl - G_inc' * ν_bal - ρ_lb + ρ_ub
  ∂K_g/∂cq_i = g_i (element i of generation vector)

So ∂K_g/∂cq = Diagonal(g) evaluated at the solution.
"""
function calc_kkt_jacobian_cost_quadratic(prob::DCOPFProblem)
    net = prob.network
    n, m, k = net.n, net.m, net.k
    dim = kkt_dims(net)

    # Get current solution for g values
    sol = solve!(prob)
    g = sol.g

    J_cq = spzeros(dim, k)

    # K_g row indices: n+1:n+k
    idx_g = n+1:n+k

    # ∂K_g/∂cq = Diagonal(g)
    # The (i, i) entry is g_i: derivative of cq_i * g_i with respect to cq_i is g_i
    J_cq[idx_g, :] = sparse(Diagonal(g))

    return J_cq
end
