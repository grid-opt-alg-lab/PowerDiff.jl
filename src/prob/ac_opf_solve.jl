# =============================================================================
# AC OPF Problem Solving and Operations
# =============================================================================
#
# Functions for solving AC OPF problems and updating parameters.

"""
    solve!(prob::ACOPFProblem)

Solve the AC OPF problem and return an ACOPFSolution.

# Returns
ACOPFSolution containing optimal primal and dual variables.

# Throws
Error if optimization does not converge to optimal/locally optimal solution.
"""
function solve!(prob::ACOPFProblem)
    optimize!(prob.model)

    status = termination_status(prob.model)
    @assert status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED] "AC OPF failed with status: $status"

    return _extract_ac_opf_solution(prob)
end

"""
Extract solution from solved AC OPF problem.
"""
function _extract_ac_opf_solution(prob::ACOPFProblem)
    n = prob.network.n
    m = prob.network.m
    k = prob.n_gen

    # Extract primal variables
    va_val = value.(prob.va)
    vm_val = value.(prob.vm)
    pg_val = value.(prob.pg)
    qg_val = value.(prob.qg)

    p_val = Dict(arc => value(prob.p[arc]) for arc in keys(prob.p))
    q_val = Dict(arc => value(prob.q[arc]) for arc in keys(prob.q))

    # Extract dual variables - power balance
    ν_p_bal = [dual(prob.cons.p_bal[i]) for i in 1:n]
    ν_q_bal = [dual(prob.cons.q_bal[i]) for i in 1:n]

    # Extract dual variables - flow equations
    ν_p_fr = [dual(prob.cons.p_fr[l]) for l in 1:m]
    ν_p_to = [dual(prob.cons.p_to[l]) for l in 1:m]
    ν_q_fr = [dual(prob.cons.q_fr[l]) for l in 1:m]
    ν_q_to = [dual(prob.cons.q_to[l]) for l in 1:m]

    # Extract dual variables - thermal limits
    λ_thermal_fr = [dual(prob.cons.thermal_fr[l]) for l in 1:m]
    λ_thermal_to = [dual(prob.cons.thermal_to[l]) for l in 1:m]

    # Extract dual variables - voltage bounds
    μ_vm_lb = [dual(LowerBoundRef(prob.vm[i])) for i in 1:n]
    μ_vm_ub = [dual(UpperBoundRef(prob.vm[i])) for i in 1:n]

    # Extract dual variables - generation bounds
    ρ_pg_lb = [dual(LowerBoundRef(prob.pg[i])) for i in 1:k]
    ρ_pg_ub = [dual(UpperBoundRef(prob.pg[i])) for i in 1:k]
    ρ_qg_lb = [dual(LowerBoundRef(prob.qg[i])) for i in 1:k]
    ρ_qg_ub = [dual(UpperBoundRef(prob.qg[i])) for i in 1:k]

    obj = objective_value(prob.model)

    return ACOPFSolution(
        va_val, vm_val,
        pg_val, qg_val,
        p_val, q_val,
        ν_p_bal, ν_q_bal,
        ν_p_fr, ν_p_to, ν_q_fr, ν_q_to,
        λ_thermal_fr, λ_thermal_to,
        μ_vm_lb, μ_vm_ub,
        ρ_pg_lb, ρ_pg_ub, ρ_qg_lb, ρ_qg_ub,
        obj
    )
end

"""
    update_switching!(prob::ACOPFProblem, z::AbstractVector)

Update the switching state in the AC OPF problem.

This modifies the network's switching state. The model constraints that depend
on switching are parameterized by network.z, so updating it and re-solving
will use the new values.

# Arguments
- `prob`: ACOPFProblem to update
- `z`: New switching state vector (length m), values in [0,1]

# Note
After calling this function, you need to rebuild the problem to update
the constraint coefficients that depend on z. For now, this just updates
the network state; full constraint rebuild is needed for exact sensitivity.
"""
function update_switching!(prob::ACOPFProblem, z::AbstractVector)
    m = prob.network.m
    @assert length(z) == m "Switching vector length must match number of branches"
    @assert all(0 .<= z .<= 1) "Switching values must be in [0,1]"

    # Update network switching state
    prob.network.z .= z

    # Note: For proper sensitivity analysis, the model would need to be rebuilt
    # since z appears in constraint coefficients. This is a limitation of the
    # current approach - a more sophisticated implementation would use
    # parameterized constraints or constraint modification.

    return prob
end
