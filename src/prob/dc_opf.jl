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
# Interior point solvers leave psh ≈ ε > 0 and gamma ≈ ε > 0 when a bound is active;
# snapping below this tolerance forces clean KKT structure.
const COMPLEMENTARITY_SNAP_TOL = 1e-6

"""
    _check_solve_status(model, label::String)

Check JuMP model termination status and throw informative errors for common failure modes.
"""
function _check_solve_status(model::JuMP.Model, label::String)
    status = termination_status(model)
    status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED) && return status
    if status == MOI.ALMOST_LOCALLY_SOLVED
        _SILENCE_WARNINGS[] || @warn "$label converged at acceptable tolerance (ALMOST_LOCALLY_SOLVED)"
        return status
    end
    if status == MOI.INFEASIBLE
        error("$label is infeasible. Check that demand is feasible given generator capacities and network constraints.")
    elseif status == MOI.DUAL_INFEASIBLE
        error("$label is unbounded (dual infeasible). Check cost coefficients and variable bounds.")
    elseif status in (MOI.ITERATION_LIMIT, MOI.TIME_LIMIT)
        error("$label solver reached $status. Try increasing solver limits or simplifying the problem.")
    else
        error("$label failed with status: $status")
    end
end

"""
    _debug_negative_demand(d)

Log if any demand entries are negative. Negative net demand is retained in
power balance, while load shedding is disabled at those buses.
"""
function _debug_negative_demand(d)
    neg_buses = findall(d .< 0)
    if !isempty(neg_buses)
        @debug "Negative demand treated as net injection with psh fixed at zero" buses=neg_buses
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
    invalidate!(prob.cache)

    optimize!(prob.model)

    _check_solve_status(prob.model, "DC OPF")

    θ_val = value.(prob.va)
    g_val = value.(prob.pg)
    f_val = value.(prob.f)
    psh_val = value.(prob.psh)

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
    η_ref = dual(prob.cons.ref)

    # Post-process phase angle difference duals for strict complementarity.
    # Interior point solvers leave gamma ≈ 1e-8 for non binding constraints.
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
    # Interior point solvers give psh ≈ ε > 0 even when shedding is inactive.
    # Snap to strict complementarity for clean KKT sensitivity computation.
    d = prob.d
    for i in eachindex(psh_val)
        shed_capacity = _shed_capacity(d[i])
        if shed_capacity < TOL
            # Degenerate: both bounds collapse to 0 ≤ psh ≤ 0
            psh_val[i] = 0.0
            # Canonicalize the duplicate dual pair so the sensitivity KKT sees
            # one effective lower-bound multiplier and a zero upper-bound dual.
            μ_lb[i] -= μ_ub[i]
            μ_ub[i] = 0.0
        elseif psh_val[i] < TOL
            # Lower bound active: psh = 0
            psh_val[i] = 0.0
            μ_ub[i] = 0.0
        elseif shed_capacity - psh_val[i] < TOL
            # Upper bound active: psh = max(d, 0)
            psh_val[i] = shed_capacity
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

    sol = DCOPFSolution(θ_val, g_val, f_val, psh_val, ν_bal, ν_flow, λ_ub, λ_lb, ρ_ub, ρ_lb, μ_lb, μ_ub, γ_lb, γ_ub, η_ref, obj, B_r_factor)

    # Cache the solution for sensitivity computations
    prob.cache.solution = sol

    return sol
end

"""
    update_demand!(prob::DCOPFProblem, d::AbstractVector)

Rewrite the RHS of the power balance and shedding upper bound constraints to
the new demand `d`, mutate `prob.d` in place, and invalidate the sensitivity
cache. The shedding upper bound is `max(d, 0)` because negative net demand is
an injection rather than curtailable load. Avoids rebuilding the JuMP model.

To keep the JuMP constraints and the sensitivity cache consistent with the
stored demand, always go through this function — do not mutate `prob.d` in
place from the outside.
"""
function update_demand!(prob::DCOPFProblem, d::AbstractVector)
    n = prob.network.n
    length(d) == n || throw(DimensionMismatch("Demand vector length $(length(d)) must match number of buses $n"))

    _debug_negative_demand(d)

    invalidate!(prob.cache)
    prob.d .= d
    for i in 1:n
        set_normalized_rhs(prob.cons.power_bal[i], d[i])
        set_normalized_rhs(prob.cons.shed_ub[i], _shed_capacity(d[i]))
    end

    return prob
end

"""
    update_fmax!(prob::DCOPFProblem, fmax::AbstractVector)

Rewrite the RHS of the `line_lb` (`f ≥ -fmax`) and `line_ub` (`f ≤ fmax`)
constraints to the new branch flow limits `fmax`, mutate `prob.network.fmax`
in place, and invalidate the sensitivity cache. Avoids rebuilding the JuMP
model.

`fmax` does not enter the reduced Laplacian factorization `b_r_factor`
(which depends only on susceptances and switching states), so the cached
factorization is preserved across the update.

To keep the JuMP constraints and the sensitivity cache consistent with the
stored limits, always go through this function — do not mutate
`prob.network.fmax` in place from the outside.

# Arguments
- `prob`: DCOPFProblem to update
- `fmax`: New flow limit vector (length m), non-negative per unit

# Throws
- `DimensionMismatch` if `length(fmax) != network.m`
- `ArgumentError` if any entry of `fmax` is negative or NaN
"""
function update_fmax!(prob::DCOPFProblem, fmax::AbstractVector)
    m = prob.network.m
    length(fmax) == m || throw(DimensionMismatch("Flow limit vector length $(length(fmax)) must match number of branches $m"))
    any(isnan, fmax) && throw(ArgumentError("Flow limits contain NaN"))
    all(fmax .>= 0) || throw(ArgumentError("Flow limits must be non-negative; got $(fmax)"))

    invalidate!(prob.cache)
    prob.network.fmax .= fmax
    for e in 1:m
        set_normalized_rhs(prob.cons.line_lb[e], -fmax[e])
        set_normalized_rhs(prob.cons.line_ub[e],  fmax[e])
    end

    return prob
end
