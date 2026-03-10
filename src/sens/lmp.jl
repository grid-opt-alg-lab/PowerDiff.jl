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
#     congestion_component = B_r⁻¹ [A_r' Diag(-b .* sw) (λ_ub - λ_lb) + A_r'(γ_ub - γ_lb)]  (non-ref block)
#     energy_component = ν_bal - congestion_component  (uniform for connected network)
#
# Sign conventions (DC OPF):
#     - Our LMPs are positive (cost increases when demand increases)
#     - PowerModels uses negative LMPs: our_lmp = -pm_lmp
#     - DCOPFSolution stores standard KKT duals (non-negative for inequality constraints)
#       JuMP's sign convention for <= constraints is handled at extraction in solve!
#
# Sign conventions (AC OPF):
#     The power balance constraint is h_P = P_flow + G_s|V|² + P_d - P_g = 0
#     (demand positive). JuMP's Lagrangian L = f - ν · h gives ν_p_bal < 0
#     at optimum (since marginal cost is positive). The LMP is the marginal
#     cost of serving demand: LMP = ∂f*/∂P_d = -ν_p_bal > 0.

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

From the θ-stationarity KKT condition:
    B' * ν_bal + (WA)' * ν_flow + e_ref * η_ref + A'*(γ_ub - γ_lb) = 0

The congestion RHS includes both flow limit duals and angle difference duals:
    congestion[non_ref] = B_r \\ (A' W (λ_ub - λ_lb) + A'(γ_ub - γ_lb))[non_ref]

The congestion component captures price differentiation due to binding flow and angle
constraints, with the reference bus congestion component equal to zero.

# Returns
Vector (length n) of congestion contributions to each bus's LMP.
"""
function calc_congestion_component(sol::DCOPFSolution, net::DCNetwork;
                                   B_r_factor=sol.B_r_factor)
    w = -net.b  # positive weights (b < 0 for inductive lines)
    non_ref = setdiff(1:net.n, net.ref_bus)

    At = net.A'
    rhs_full = At * Diagonal(w .* net.sw) * (sol.lam_ub - sol.lam_lb) + At * (sol.gamma_ub - sol.gamma_lb)

    result = zeros(net.n)
    result[non_ref] = B_r_factor \ rhs_full[non_ref]
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

# =============================================================================
# AC OPF LMP Computation
# =============================================================================

"""
    calc_lmp(sol::ACOPFSolution, prob::ACOPFProblem)

Compute Locational Marginal Prices from AC OPF solution.

The LMP at bus i is the marginal cost of serving an additional unit of active
demand at that bus: LMP_i = ∂f*/∂P_d_i = -ν_p_bal_i.

The sign negation arises because JuMP's dual `ν_p_bal` is negative at optimum
for the standard power balance formulation (see sign derivation in file header).

# Returns
Vector of LMPs (length n), one per bus.
"""
function calc_lmp(sol::ACOPFSolution, prob::ACOPFProblem)
    return -sol.nu_p_bal
end

"""
    calc_lmp(prob::ACOPFProblem)

Solve the AC OPF problem (if needed) and compute LMPs.
"""
function calc_lmp(prob::ACOPFProblem)
    sol = if isnothing(prob.cache.solution)
        solve!(prob)
    else
        prob.cache.solution
    end
    return calc_lmp(sol, prob)
end

