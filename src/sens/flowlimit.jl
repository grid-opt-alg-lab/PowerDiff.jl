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
    # Use getfield because DC OPF types overload getproperty for field aliases.
    net = getfield(prob, :network)
    n = getfield(net, :n)
    m = getfield(net, :m)
    k = getfield(net, :k)
    dim, idx = _dc_kkt_layout(prob)

    lambda_lb = getfield(sol, :lam_lb)
    lambda_ub = getfield(sol, :lam_ub)

    colptr = Vector{Int}(undef, m + 1)
    rowval = Int[]
    nzval = Float64[]
    sizehint!(rowval, 2m)
    sizehint!(nzval, 2m)

    # dK_lambda_lb/dfmax and dK_lambda_ub/dfmax.
    @inbounds for e in 1:m
        colptr[e] = length(rowval) + 1
        # K_lambda_lb = lambda_lb .* (f + fmax), so dK_lambda_lb/dfmax_e = lambda_lb[e]
        _push_csc_entry!(rowval, nzval, idx.lam_lb[e], lambda_lb[e])
        # K_lambda_ub = lambda_ub .* (fmax - f), so dK_lambda_ub/dfmax_e = lambda_ub[e]
        _push_csc_entry!(rowval, nzval, idx.lam_ub[e], lambda_ub[e])
    end
    colptr[m + 1] = length(rowval) + 1

    return SparseMatrixCSC(dim, m, colptr, rowval, nzval)
end

"""
    calc_kkt_jacobian_flowlimit_column(net, sol, e::Int) → Vector{Float64}

Compute column `e` of ∂K/∂fmax. Only 2 nonzeros: `lam_lb[e]` and `lam_ub[e]`.
"""
function calc_kkt_jacobian_flowlimit_column(net::DCNetwork, sol::DCOPFSolution, e::Int)
    dim, idx = _dc_kkt_layout(net)
    col = zeros(dim)
    col[idx.lam_lb[e]] = sol.lam_lb[e]
    col[idx.lam_ub[e]] = sol.lam_ub[e]
    return col
end
