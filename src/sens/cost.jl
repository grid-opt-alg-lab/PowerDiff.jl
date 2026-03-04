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

# Cost Sensitivity Analysis for DC OPF
# Uses implicit differentiation via KKT conditions
#
# Note: For DCOPFProblem, cost sensitivities are computed via the cached
# KKT system in kkt_dc_opf.jl. This file contains the KKT Jacobian functions
# for cost parameters.

"""
    calc_kkt_jacobian_cost_linear(net::DCNetwork)

Compute the Jacobian of KKT conditions with respect to linear cost coefficients dK/dcl.

# Returns
Sparse matrix of size (kkt_dims x k).

# Notes
Only the stationarity condition for g depends on cl:
  K_g = 2*Cq * g + cl - G_inc' * nu_bal - rho_lb + rho_ub
  dK_g/dcl = I_k (identity matrix)
"""
function calc_kkt_jacobian_cost_linear(net::DCNetwork)
    n, m, k = net.n, net.m, net.k
    dim = kkt_dims(net)
    idx = kkt_indices(n, m, k)

    J_cl = spzeros(dim, k)

    # dK_g/dcl = I_k
    J_cl[idx.pg, :] = sparse(I, k, k)

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
  K_g = 2*Cq * g + cl - G_inc' * nu_bal - rho_lb + rho_ub
  dK_g/dcq_i = 2*g_i (since objective is cq_i * g_i^2, stationarity has 2*cq_i*g_i)

So dK_g/dcq = 2*Diagonal(g) evaluated at the solution.
"""
function calc_kkt_jacobian_cost_quadratic(prob::DCOPFProblem, sol::DCOPFSolution)
    net = prob.network
    n, m, k = net.n, net.m, net.k
    dim = kkt_dims(net)
    idx = kkt_indices(n, m, k)

    g = sol.pg

    J_cq = spzeros(dim, k)

    # dK_g/dcq = 2*Diagonal(g)
    # Objective is cq_i * g_i^2, stationarity is 2*cq_i*g_i + cl_i - ...
    # So ∂(2*cq_i*g_i)/∂cq_i = 2*g_i
    J_cq[idx.pg, :] = 2 * sparse(Diagonal(g))

    return J_cq
end
