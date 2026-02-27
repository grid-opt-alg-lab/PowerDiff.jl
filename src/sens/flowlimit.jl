# Flow Limit Sensitivity Analysis for DC OPF
# Uses implicit differentiation via KKT conditions
#
# Note: For DCOPFProblem, flow limit sensitivities are computed via the cached
# KKT system in kkt_dc_opf.jl. This file contains the KKT Jacobian function
# for flow limit parameters.
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

    lambda_lb = sol.λ_lb
    lambda_ub = sol.λ_ub

    J_fmax = spzeros(dim, m)

    # dK_lambda_lb/dfmax = Diag(lambda_lb)
    # K_lambda_lb = lambda_lb .* (f + fmax), so dK_lambda_lb/dfmax_e = lambda_lb[e] for row e
    J_fmax[idx.λ_lb, :] = sparse(Diagonal(lambda_lb))

    # dK_lambda_ub/dfmax = Diag(lambda_ub)
    # K_lambda_ub = lambda_ub .* (fmax - f), so dK_lambda_ub/dfmax_e = lambda_ub[e] for row e
    J_fmax[idx.λ_ub, :] = sparse(Diagonal(lambda_ub))

    return J_fmax
end
