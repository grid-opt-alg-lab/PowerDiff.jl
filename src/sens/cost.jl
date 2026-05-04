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
    # Use getfield because DCNetwork overloads getproperty for field aliases.
    n = getfield(net, :n)
    m = getfield(net, :m)
    k = getfield(net, :k)
    dim = kkt_dims(n, m, k)
    idx = kkt_indices(n, m, k)

    colptr = Vector{Int}(undef, k + 1)
    rowval = Int[]
    nzval = Float64[]
    sizehint!(rowval, k)
    sizehint!(nzval, k)

    # dK_g/dcl = I_k
    @inbounds for j in 1:k
        colptr[j] = length(rowval) + 1
        _push_csc_entry!(rowval, nzval, idx.pg[j], 1.0)
    end
    colptr[k + 1] = length(rowval) + 1

    return SparseMatrixCSC(dim, k, colptr, rowval, nzval)
end

"""
    calc_kkt_jacobian_cost_linear_column(net, j::Int) → Vector{Float64}

Compute column `j` of ∂K/∂cl. Only 1 nonzero: `pg[j] = 1.0`.
"""
function calc_kkt_jacobian_cost_linear_column(net::DCNetwork, j::Int)
    col = zeros(kkt_dims(net))
    idx = kkt_indices(net)
    col[idx.pg[j]] = 1.0
    return col
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
    # Use getfield because DC OPF types overload getproperty for field aliases.
    net = getfield(prob, :network)
    n = getfield(net, :n)
    m = getfield(net, :m)
    k = getfield(net, :k)
    dim = kkt_dims(n, m, k)
    idx = kkt_indices(n, m, k)

    g = getfield(sol, :pg)

    colptr = Vector{Int}(undef, k + 1)
    rowval = Int[]
    nzval = Float64[]
    sizehint!(rowval, k)
    sizehint!(nzval, k)

    # dK_g/dcq = 2*Diagonal(g)
    # Objective is cq_i * g_i^2, stationarity is 2*cq_i*g_i + cl_i - ...
    # So ∂(2*cq_i*g_i)/∂cq_i = 2*g_i
    @inbounds for j in 1:k
        colptr[j] = length(rowval) + 1
        _push_csc_entry!(rowval, nzval, idx.pg[j], 2 * g[j])
    end
    colptr[k + 1] = length(rowval) + 1

    return SparseMatrixCSC(dim, k, colptr, rowval, nzval)
end

"""
    calc_kkt_jacobian_cost_quadratic_column(net, sol, j::Int) → Vector{Float64}

Compute column `j` of ∂K/∂cq. Only 1 nonzero: `pg[j] = 2*g[j]`.
"""
function calc_kkt_jacobian_cost_quadratic_column(net::DCNetwork, sol::DCOPFSolution, j::Int)
    col = zeros(kkt_dims(net))
    idx = kkt_indices(net)
    col[idx.pg[j]] = 2.0 * sol.pg[j]
    return col
end
