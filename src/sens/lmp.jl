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
#     congestion_component = B_r⁻¹ [A_r' Diag(-b .* sw) (λ_ub - λ_lb) + A_r' Diag(sw) (γ_ub - γ_lb)]  (non-ref block)
#     energy_component = ν_bal - congestion_component  (uniform within each island)
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

From the θ-stationarity KKT condition, where `E_ref` is the `n × n_ref` matrix
that selects the one reference bus per energized island (so `E_ref * η_ref` is a
length-`n` vector aligned with the bus stationarity rows):
    B' * ν_bal + (WA)' * ν_flow + E_ref * η_ref + A' Diag(sw) (γ_ub - γ_lb) = 0

Neglecting the O(τ²) flow regularization (so ν_flow ≈ λ_ub - λ_lb), the congestion
RHS gathers the flow limit and angle difference dual contributions:
    congestion[non_ref] = B_r \\ (A' W (λ_ub - λ_lb) + A' Diag(sw) (γ_ub - γ_lb))[non_ref]

The congestion component captures price differentiation due to binding flow and angle
constraints. Only the non-reference rows are populated, so every energized island
reference bus has a congestion component of exactly zero.

# Returns
Vector (length n) of congestion contributions to each bus's LMP.
"""
function calc_congestion_component(sol::DCOPFSolution, net::DCNetwork;
                                   B_r_factor=sol.B_r_factor)
    w = -net.b  # positive weights (b < 0 for inductive lines)
    non_ref = _non_reference_buses(net)

    At = net.A'
    # Angle difference constraints are gated by `sw` in the model, so their
    # stationarity contribution carries the same `Diag(sw)` factor: the gate is the
    # identity on fully closed branches (sw == 1), scales the contribution on
    # fractional branches (0 < sw < 1), and zeroes the term on open branches
    # (sw == 0), matching the gated angle dual the solver returns.
    rhs_full = At * Diagonal(w .* net.sw) * (sol.lam_ub - sol.lam_lb) +
               At * Diagonal(net.sw) * (sol.gamma_ub - sol.gamma_lb)

    result = zeros(net.n)
    result[non_ref] = B_r_factor \ rhs_full[non_ref]
    return result
end

"""
    calc_energy_component(sol::DCOPFSolution, net::DCNetwork)

Extract the energy (non-congestion) component of LMPs for analysis.

This is the uniform price component: energy = ν_bal - congestion.
It is exactly uniform within each energized island in the unregularized limit;
the O(τ²) flow regularization perturbs that uniformity slightly.

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
    sol = _ensure_ac_solved!(prob)
    return calc_lmp(sol, prob)
end

# =============================================================================
# AC OPF Reactive Power LMP (QLMP) Computation
# =============================================================================

"""
    calc_qlmp(sol::ACOPFSolution, prob::ACOPFProblem)

Compute reactive power Locational Marginal Prices from AC OPF solution.

The QLMP at bus i is the marginal cost of serving an additional unit of reactive
demand at that bus: QLMP_i = ∂f*/∂Q_d_i = -ν_q_bal_i.

# Returns
Vector of QLMPs (length n), one per bus.
"""
function calc_qlmp(sol::ACOPFSolution, prob::ACOPFProblem)
    return -sol.nu_q_bal
end

"""
    calc_qlmp(prob::ACOPFProblem)

Solve the AC OPF problem (if needed) and compute reactive power LMPs.
"""
function calc_qlmp(prob::ACOPFProblem)
    sol = _ensure_ac_solved!(prob)
    return calc_qlmp(sol, prob)
end
