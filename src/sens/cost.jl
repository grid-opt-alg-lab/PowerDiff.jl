# Cost Sensitivity Analysis for DC OPF
# Uses implicit differentiation via KKT conditions
#
# Note: For DCOPFProblem, cost sensitivities are computed via the cached
# KKT system in kkt_dc_opf.jl. This file contains the KKT Jacobian functions
# for cost parameters.

using LinearAlgebra
using SparseArrays

"""
    calc_kkt_jacobian_cost_linear(net::DCNetwork)

Compute the Jacobian of KKT conditions with respect to linear cost coefficients dK/dcl.

# Returns
Sparse matrix of size (kkt_dims x k).

# Notes
Only the stationarity condition for g depends on cl:
  K_g = Cq * g + cl - G_inc' * nu_bal - rho_lb + rho_ub
  dK_g/dcl = I_k (identity matrix)
"""
function calc_kkt_jacobian_cost_linear(net::DCNetwork)
    n, m, k = net.n, net.m, net.k
    dim = kkt_dims(net)
    idx = kkt_indices(n, m, k)

    J_cl = spzeros(dim, k)

    # dK_g/dcl = I_k
    J_cl[idx.g, :] = sparse(I, k, k)

    return J_cl
end

"""
    calc_kkt_jacobian_cost_quadratic(prob::DCOPFProblem, sol::DCOPFSolution)

Compute the Jacobian of KKT conditions with respect to quadratic cost coefficients dK/dcq.

# Arguments
- `prob`: DCOPFProblem
- `sol`: Pre-computed solution

# Returns
Sparse matrix of size (kkt_dims x k).

# Notes
Only the stationarity condition for g depends on cq:
  K_g = Cq * g + cl - G_inc' * nu_bal - rho_lb + rho_ub
  dK_g/dcq_i = g_i (element i of generation vector)

So dK_g/dcq = Diagonal(g) evaluated at the solution.
"""
function calc_kkt_jacobian_cost_quadratic(prob::DCOPFProblem, sol::DCOPFSolution)
    net = prob.network
    n, m, k = net.n, net.m, net.k
    dim = kkt_dims(net)
    idx = kkt_indices(n, m, k)

    g = sol.g

    J_cq = spzeros(dim, k)

    # dK_g/dcq = Diagonal(g)
    # The (i, i) entry is g_i: derivative of cq_i * g_i with respect to cq_i is g_i
    J_cq[idx.g, :] = sparse(Diagonal(g))

    return J_cq
end
