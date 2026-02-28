# =============================================================================
# DC OPF Problem Solving and Operations
# =============================================================================
#
# Functions for solving DC OPF problems and updating parameters.

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
    @assert status ∈ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED] "Optimization failed with status: $status"

    # Extract primal variables
    θ_val = value.(prob.θ)
    g_val = value.(prob.g)
    f_val = value.(prob.f)
    psh_val = value.(prob.psh)

    # Extract dual variables
    ν_bal = dual.(prob.cons.power_bal)
    ν_flow = dual.(prob.cons.flow_def)
    λ_ub = dual.(prob.cons.line_ub)
    λ_lb = dual.(prob.cons.line_lb)
    ρ_ub = dual.(prob.cons.gen_ub)
    ρ_lb = dual.(prob.cons.gen_lb)
    μ_lb = dual.(prob.cons.shed_lb)
    μ_ub = dual.(prob.cons.shed_ub)

    # Post-process load shedding for strict complementarity.
    # Interior-point solvers give psh ≈ ε > 0 even when shedding is inactive.
    # Snap to strict complementarity for clean KKT sensitivity computation.
    for i in eachindex(psh_val)
        if psh_val[i] < 1e-6
            psh_val[i] = 0.0
            μ_ub[i] = 0.0
        else
            μ_lb[i] = 0.0
        end
    end

    obj = objective_value(prob.model)

    sol = DCOPFSolution(θ_val, g_val, f_val, psh_val, ν_bal, ν_flow, λ_ub, λ_lb, ρ_ub, ρ_lb, μ_lb, μ_ub, obj)

    # Cache the solution for sensitivity computations
    prob.cache.solution = sol

    return sol
end

"""
    update_demand!(prob::DCOPFProblem, d::AbstractVector)

Update the demand parameter in the DC OPF problem.

This modifies the RHS of power balance constraints for re-solving with new demand.
Invalidates the sensitivity cache since parameters have changed.
"""
function update_demand!(prob::DCOPFProblem, d::AbstractVector)
    n = prob.network.n
    @assert length(d) == n "Demand vector length must match number of buses"

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
