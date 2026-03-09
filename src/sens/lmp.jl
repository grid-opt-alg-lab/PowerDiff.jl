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
# DC OPF (B-θ formulation)
# -------------------------
# The power balance constraint is:
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
# AC OPF (polar formulation)
# ---------------------------
# The active power balance constraint at bus i is (JuMP form):
#     Σ p_arcs == Pg - Pd - Psh
# where the normalized residual h_P = Σ p_arcs - Pg + Pd + Psh.
#
# The Lagrangian uses: L = f - ν_p_bal * h_P  (JuMP/MOI dual sign convention)
# KKT stationarity for Pg at bus i: cost' + ν_p_bal = 0 → ν_p_bal = -cost' < 0
#
# The MATPOWER-convention LMP is the marginal cost of demand (∂f*/∂Pd > 0):
#     LMP = ∂f*/∂Pd = -ν_p_bal > 0
#
# Therefore, for ACOPFSolution:  calc_lmp = -nu_p_bal
# For the sensitivity:           ∂LMP/∂param = -∂(ν_p_bal)/∂param
#
# Sign conventions:
#     - Our LMPs are positive (cost increases when demand increases)
#     - PowerModels/MATPOWER uses negative Lagrange multipliers: our_lmp = -pm_lambda
#     - DCOPFSolution: LMP = ν_bal > 0 (demand with negative sign in KKT residual)
#     - ACOPFSolution: LMP = -ν_p_bal > 0 (demand with positive sign in KKT residual)

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
    calc_lmp(sol::ACOPFSolution, prob::ACOPFProblem)

Compute Locational Marginal Prices from AC OPF solution.

Following the MATPOWER convention, the LMP at bus i is the marginal cost of serving
an additional unit of active power demand at that bus:
    LMP[i] = ∂f*/∂Pd[i]

In the polar AC OPF formulation, the active power balance constraint is:
    Σ p_arcs == Pg - Pd - Psh  (JuMP form, normalized residual h_P = Σp - Pg + Pd + Psh)

The Lagrangian uses L = f - ν_p_bal * h_P, giving KKT stationarity:
    cost' + ν_p_bal = 0  →  ν_p_bal = -cost' < 0

Applying the envelope theorem: LMP = ∂f*/∂Pd = -ν_p_bal > 0.
Hence `LMP = -sol.nu_p_bal` (negation of the JuMP power-balance dual).

# Returns
Vector of LMPs (length n), one per bus, in the same units as the generation cost.

# Example
```julia
prob = ACOPFProblem(pm_data)
sol = solve!(prob)
lmps = calc_lmp(sol, prob)
```
"""
function calc_lmp(sol::ACOPFSolution, prob::ACOPFProblem)
    return -sol.nu_p_bal
end

"""
    calc_lmp(prob::ACOPFProblem)

Ensure the AC OPF is solved and compute LMPs.
Reuses the cached solution when available.
"""
function calc_lmp(prob::ACOPFProblem)
    sol = _ensure_ac_solved!(prob)
    return calc_lmp(sol, prob)
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
function calc_congestion_component(sol::DCOPFSolution, net::DCNetwork)
    w = -net.b  # positive weights (b < 0 for inductive lines)
    B = calc_susceptance_matrix(net)
    non_ref = setdiff(1:net.n, net.ref_bus)
    B_r = B[non_ref, non_ref]

    At = net.A'
    rhs_full = At * Diagonal(w .* net.sw) * (sol.lam_ub - sol.lam_lb) + At * (sol.gamma_ub - sol.gamma_lb)

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

