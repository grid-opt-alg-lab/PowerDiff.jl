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

# =============================================================================
# DC OPF Problem Solving and Operations
# =============================================================================
#
# Functions for solving DC OPF problems and updating parameters.

# Threshold for snapping near-boundary primal/dual values to strict complementarity.
# Interior-point solvers leave psh ≈ ε > 0 and gamma ≈ ε > 0 when a bound is active;
# snapping below this tolerance forces clean KKT structure.
const COMPLEMENTARITY_SNAP_TOL = 1e-6

"""
    _warn_negative_demand(d)

Warn if any demand entries are negative, which makes shedding bounds infeasible.
"""
function _warn_negative_demand(d)
    neg_buses = findall(d .< 0)
    if !isempty(neg_buses)
        @warn "Negative demand at buses $neg_buses; shedding bounds 0 ≤ psh ≤ d will be infeasible at those buses"
    end
end

"""
    solve!(prob::DCOPFProblem)

Solve the DC OPF problem and return a DCOPFSolution.

Invalidates the sensitivity cache since the solution may have changed.

# Returns
DCOPFSolution containing optimal primal and dual variables.

# Throws
Error if optimization does not converge to optimal/locally optimal solution.
"""
function solve!(prob::DCOPFProblem)
    # Invalidate sensitivity cache since we're re-solving
    invalidate!(prob.cache)

    optimize!(prob.model)

    status = termination_status(prob.model)
    status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED) || error("DC OPF failed with status: $status")

    # Extract primal variables
    θ_val = value.(prob.va)
    g_val = value.(prob.pg)
    f_val = value.(prob.f)
    psh_val = value.(prob.psh)

    # Extract dual variables
    # JuMP returns non-positive duals for <= constraints; negate to get standard
    # KKT duals (non-negative for inequality constraints).
    ν_bal = dual.(prob.cons.power_bal)
    ν_flow = dual.(prob.cons.flow_def)
    λ_ub = -dual.(prob.cons.line_ub)
    λ_lb = dual.(prob.cons.line_lb)
    ρ_ub = -dual.(prob.cons.gen_ub)
    ρ_lb = dual.(prob.cons.gen_lb)
    μ_lb = dual.(prob.cons.shed_lb)
    μ_ub = -dual.(prob.cons.shed_ub)
    γ_lb = dual.(prob.cons.phase_diff_lb)
    γ_ub = -dual.(prob.cons.phase_diff_ub)

    # Post-process phase angle difference duals for strict complementarity.
    # Interior-point solvers leave gamma ≈ 1e-8 for non-binding constraints.
    net = prob.network
    Atheta = net.A * θ_val
    TOL = COMPLEMENTARITY_SNAP_TOL
    for e in 1:net.m
        if net.angmax[e] - Atheta[e] > TOL  # upper angle not binding
            γ_ub[e] = 0.0
        else
            γ_ub[e] = max(γ_ub[e], 0.0)  # clamp solver noise on binding side
        end
        if Atheta[e] - net.angmin[e] > TOL  # lower angle not binding
            γ_lb[e] = 0.0
        else
            γ_lb[e] = max(γ_lb[e], 0.0)  # clamp solver noise on binding side
        end
    end

    # Post-process load shedding for strict complementarity.
    # Interior-point solvers give psh ≈ ε > 0 even when shedding is inactive.
    # Snap to strict complementarity for clean KKT sensitivity computation.
    d = prob.d
    for i in eachindex(psh_val)
        if d[i] < -TOL
            @warn "Negative demand at bus $i (d=$(d[i])); psh snap may be unreliable"
        end
        if abs(d[i]) < TOL
            # Degenerate: both bounds collapse to 0 ≤ psh ≤ 0
            psh_val[i] = 0.0
            # Keep both duals from solver (both bounds active)
        elseif psh_val[i] < TOL
            # Lower bound active: psh = 0
            psh_val[i] = 0.0
            μ_ub[i] = 0.0
        elseif d[i] - psh_val[i] < TOL
            # Upper bound active: psh = d
            psh_val[i] = d[i]
            μ_lb[i] = 0.0
        else
            # Strictly interior: both bounds inactive
            μ_lb[i] = 0.0
            μ_ub[i] = 0.0
        end
    end

    obj = objective_value(prob.model)

    # Reuse cached reduced susceptance factorization, or compute and cache it
    if prob.cache.b_r_factor === nothing
        prob.cache.b_r_factor, _ = _factorize_B_r(net)
    end
    B_r_factor = prob.cache.b_r_factor

    sol = DCOPFSolution(θ_val, g_val, f_val, psh_val, ν_bal, ν_flow, λ_ub, λ_lb, ρ_ub, ρ_lb, μ_lb, μ_ub, γ_lb, γ_ub, obj, B_r_factor)

    # Cache the solution for sensitivity computations
    prob.cache.solution = sol

    return sol
end

"""
    update_demand!(prob::DCOPFProblem, d::AbstractVector)

Update the demand parameter in the DC OPF problem.

This modifies the RHS of power balance and shedding upper-bound constraints for re-solving with new demand.
Invalidates the sensitivity cache since parameters have changed.
"""
function update_demand!(prob::DCOPFProblem, d::AbstractVector)
    n = prob.network.n
    length(d) == n || throw(DimensionMismatch("Demand vector length $(length(d)) must match number of buses $n"))

    _warn_negative_demand(d)

    # Invalidate sensitivity cache since parameters changed
    invalidate!(prob.cache)

    # Update stored demand
    prob.d .= d

    # Update constraint RHS
    for i in 1:n
        set_normalized_rhs(prob.cons.power_bal[i], d[i])
        set_normalized_rhs(prob.cons.shed_ub[i], d[i])
    end

    return prob
end
