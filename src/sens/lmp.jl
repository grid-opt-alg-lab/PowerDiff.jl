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

# Locational Marginal Price (LMP) Computation
#
# In the B-θ DC OPF formulation, the power balance constraint is:
#     G_inc * g + psh - d = B * θ
# where B = A' * Diag(-b .* sw) * A is the susceptance-weighted Laplacian.
#
# The LMP at bus i equals the power balance dual ν_bal[i]. The network topology
# is embedded in the constraint through B, so ν_bal already incorporates both
# energy and congestion effects.
#
# LMP Decomposition (for analysis):
#     LMP = ν_bal = energy_component + congestion_component
# where:
#     congestion_component = B_r⁻¹ A_r' Diag(-b .* sw) (λ_ub_std - λ_lb_std)  (non-ref block)
#     energy_component = ν_bal - congestion_component  (uniform for connected network)
#
# Sign conventions:
#     - Our LMPs are positive (cost increases when demand increases)
#     - PowerModels uses negative LMPs: our_lmp = -pm_lmp
#     - JuMP returns λ_ub < 0 for binding f ≤ fmax constraints
#     - Standard convention has λ_ub_std ≥ 0, so λ_ub_std = -λ_ub_jmp

"""
    calc_lmp(sol::DCOPFSolution, net::DCNetwork)

Compute Locational Marginal Prices from DC OPF solution.

The LMP at bus i is the marginal cost of serving an additional unit of demand
at that bus. In the B-θ formulation, this equals the power balance dual ν_bal[i].

# Returns
Vector of LMPs (length n), one per bus.

# Example
```julia
sol = solve!(prob)
lmps = calc_lmp(sol, prob.network)
```
"""
function calc_lmp(sol::DCOPFSolution, net::DCNetwork)
    return sol.nu_bal
end

"""
    calc_lmp(prob::DCOPFProblem)

Solve the problem (if needed) and compute LMPs.
"""
function calc_lmp(prob::DCOPFProblem)
    sol = solve!(prob)
    return calc_lmp(sol, prob.network)
end

"""
    calc_congestion_component(sol::DCOPFSolution, net::DCNetwork)

Extract the congestion component of LMPs for analysis.

From KKT conditions: B * ν_bal = A' * W * (λ_ub_std - λ_lb_std)
Solving the reduced system (ref bus deleted):
    congestion[non_ref] = B_r \\ (A' W (λ_ub_std - λ_lb_std))[non_ref]

The congestion component captures price differentiation due to binding flow constraints,
with the reference bus congestion component equal to zero.
This represents price differentiation due to binding flow constraints.

# Returns
Vector (length n) of congestion contributions to each bus's LMP.
"""
function calc_congestion_component(sol::DCOPFSolution, net::DCNetwork)
    w = -net.b  # positive weights (b < 0 for inductive lines)
    B = calc_susceptance_matrix(net)
    non_ref = setdiff(1:net.n, net.ref_bus)
    B_r = B[non_ref, non_ref]

    # Convert JuMP duals to standard convention: λ_ub_std = -λ_ub_jmp
    λ_ub_std = -sol.lam_ub
    λ_lb_std = sol.lam_lb
    rhs_full = net.A' * Diagonal(w .* net.sw) * (λ_ub_std - λ_lb_std)

    result = zeros(net.n)
    result[non_ref] = B_r \ rhs_full[non_ref]
    return result
end

"""
    calc_energy_component(sol::DCOPFSolution, net::DCNetwork)

Extract the energy (non-congestion) component of LMPs for analysis.

This is the uniform price component: energy = ν_bal - congestion
For a connected network, this should be approximately constant across all buses.

# Returns
Vector (length n) of energy contributions to each bus's LMP.
"""
function calc_energy_component(sol::DCOPFSolution, net::DCNetwork)
    return sol.nu_bal .- calc_congestion_component(sol, net)
end

