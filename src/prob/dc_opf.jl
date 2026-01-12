# =============================================================================
# DC OPF Problem Solving and Operations
# =============================================================================
#
# Functions for solving DC OPF problems and updating parameters.

"""
    solve!(prob::DCOPFProblem)

Solve the DC OPF problem and return a DCOPFSolution.

# Returns
DCOPFSolution containing optimal primal and dual variables.

# Throws
Error if optimization does not converge to optimal/locally optimal solution.
"""
function solve!(prob::DCOPFProblem)
    optimize!(prob.model)

    status = termination_status(prob.model)
    @assert status ∈ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED] "Optimization failed with status: $status"

    # Extract primal variables
    θ_val = value.(prob.θ)
    g_val = value.(prob.g)
    f_val = value.(prob.f)

    # Extract dual variables
    ν_bal = dual.(prob.cons.power_bal)
    ν_flow = dual.(prob.cons.flow_def)
    λ_ub = dual.(prob.cons.line_ub)
    λ_lb = dual.(prob.cons.line_lb)
    ρ_ub = dual.(prob.cons.gen_ub)
    ρ_lb = dual.(prob.cons.gen_lb)

    obj = objective_value(prob.model)

    return DCOPFSolution(θ_val, g_val, f_val, ν_bal, ν_flow, λ_ub, λ_lb, ρ_ub, ρ_lb, obj)
end

"""
    update_demand!(prob::DCOPFProblem, d::AbstractVector)

Update the demand parameter in the DC OPF problem.

This modifies the RHS of power balance constraints for re-solving with new demand.
"""
function update_demand!(prob::DCOPFProblem, d::AbstractVector)
    n = prob.network.n
    @assert length(d) == n "Demand vector length must match number of buses"

    # Update stored demand
    prob.d .= d

    # Update constraint RHS: G_inc * g - d = B * θ  →  G_inc * g - B * θ = d
    # The constraint is stored as: G_inc * g - B * θ - d = 0
    # We need to update the constant term
    W = Diagonal(-prob.network.b .* prob.network.z)
    B_mat = sparse(prob.network.A' * W * prob.network.A)

    for i in 1:n
        set_normalized_rhs(prob.cons.power_bal[i], d[i])
    end

    return prob
end
