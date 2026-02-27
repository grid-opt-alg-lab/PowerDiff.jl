# =============================================================================
# AC OPF Problem Solving and Operations
# =============================================================================
#
# Functions for solving AC OPF problems and updating parameters.

"""
    solve!(prob::ACOPFProblem)

Solve the AC OPF problem and return an ACOPFSolution.

Invalidates the sensitivity cache since the solution may have changed.

# Returns
ACOPFSolution containing optimal primal and dual variables.

# Throws
Error if optimization does not converge to optimal/locally optimal solution.
"""
function solve!(prob::ACOPFProblem)
    # Invalidate sensitivity cache since we're re-solving
    invalidate!(prob.cache)

    optimize!(prob.model)

    status = termination_status(prob.model)
    @assert status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED] "AC OPF failed with status: $status"

    sol = _extract_ac_opf_solution(prob)

    # Cache the solution for sensitivity computations
    prob.cache.solution = sol

    return sol
end

"""
Extract solution from solved AC OPF problem.
"""
function _extract_ac_opf_solution(prob::ACOPFProblem)
    n = prob.network.n
    m = prob.network.m
    k = prob.n_gen
    ref = prob.ref

    # Extract primal variables
    va_val = value.(prob.va)
    vm_val = value.(prob.vm)
    pg_val = value.(prob.pg)
    qg_val = value.(prob.qg)

    p_val = Dict(arc => value(prob.p[arc]) for arc in keys(prob.p))
    q_val = Dict(arc => value(prob.q[arc]) for arc in keys(prob.q))

    # Extract dual variables - power balance (equality)
    ν_p_bal = [dual(prob.cons.p_bal[i]) for i in 1:n]
    ν_q_bal = [dual(prob.cons.q_bal[i]) for i in 1:n]

    # Extract dual variables - reference bus (equality)
    ref_bus_keys = sort(collect(keys(ref[:ref_buses])))
    ν_ref_bus = [dual(prob.cons.ref_bus[i]) for i in ref_bus_keys]

    # Extract dual variables - flow definition equations (equality)
    ν_p_fr = [dual(prob.cons.p_fr[l]) for l in 1:m]
    ν_p_to = [dual(prob.cons.p_to[l]) for l in 1:m]
    ν_q_fr = [dual(prob.cons.q_fr[l]) for l in 1:m]
    ν_q_to = [dual(prob.cons.q_to[l]) for l in 1:m]

    # Extract dual variables - thermal limits (inequality)
    λ_thermal_fr = [dual(prob.cons.thermal_fr[l]) for l in 1:m]
    λ_thermal_to = [dual(prob.cons.thermal_to[l]) for l in 1:m]

    # Extract dual variables - angle difference limits (inequality)
    λ_angle_lb = [dual(prob.cons.angle_diff[l][1]) for l in 1:m]
    λ_angle_ub = [dual(prob.cons.angle_diff[l][2]) for l in 1:m]

    # Extract dual variables - voltage bounds (inequality)
    μ_vm_lb = [dual(LowerBoundRef(prob.vm[i])) for i in 1:n]
    μ_vm_ub = [dual(UpperBoundRef(prob.vm[i])) for i in 1:n]

    # Extract dual variables - generation bounds (inequality)
    ρ_pg_lb = [dual(LowerBoundRef(prob.pg[i])) for i in 1:k]
    ρ_pg_ub = [dual(UpperBoundRef(prob.pg[i])) for i in 1:k]
    ρ_qg_lb = [dual(LowerBoundRef(prob.qg[i])) for i in 1:k]
    ρ_qg_ub = [dual(UpperBoundRef(prob.qg[i])) for i in 1:k]

    # Extract dual variables - flow variable bounds (inequality)
    σ_p_fr_lb = zeros(m)
    σ_p_fr_ub = zeros(m)
    σ_q_fr_lb = zeros(m)
    σ_q_fr_ub = zeros(m)
    σ_p_to_lb = zeros(m)
    σ_p_to_ub = zeros(m)
    σ_q_to_lb = zeros(m)
    σ_q_to_ub = zeros(m)

    for l in 1:m
        branch = ref[:branch][l]
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        f_idx = (l, f_bus, t_bus)
        t_idx = (l, t_bus, f_bus)

        σ_p_fr_lb[l] = dual(LowerBoundRef(prob.p[f_idx]))
        σ_p_fr_ub[l] = dual(UpperBoundRef(prob.p[f_idx]))
        σ_q_fr_lb[l] = dual(LowerBoundRef(prob.q[f_idx]))
        σ_q_fr_ub[l] = dual(UpperBoundRef(prob.q[f_idx]))
        σ_p_to_lb[l] = dual(LowerBoundRef(prob.p[t_idx]))
        σ_p_to_ub[l] = dual(UpperBoundRef(prob.p[t_idx]))
        σ_q_to_lb[l] = dual(LowerBoundRef(prob.q[t_idx]))
        σ_q_to_ub[l] = dual(UpperBoundRef(prob.q[t_idx]))
    end

    obj = objective_value(prob.model)

    return ACOPFSolution(
        va_val, vm_val,
        pg_val, qg_val,
        p_val, q_val,
        ν_p_bal, ν_q_bal,
        ν_ref_bus,
        ν_p_fr, ν_p_to, ν_q_fr, ν_q_to,
        λ_thermal_fr, λ_thermal_to,
        λ_angle_lb, λ_angle_ub,
        μ_vm_lb, μ_vm_ub,
        ρ_pg_lb, ρ_pg_ub, ρ_qg_lb, ρ_qg_ub,
        σ_p_fr_lb, σ_p_fr_ub, σ_q_fr_lb, σ_q_fr_ub,
        σ_p_to_lb, σ_p_to_ub, σ_q_to_lb, σ_q_to_ub,
        obj
    )
end

"""
    update_switching!(prob::ACOPFProblem, sw::AbstractVector)

Update the network switching state and invalidate the sensitivity cache.

# Arguments
- `prob`: ACOPFProblem to update
- `sw`: New switching state vector (length m), values in [0,1]

# Warning
This modifies `prob.network.sw` and invalidates cached sensitivities.
The JuMP model constraints embed the previous sw values as coefficients,
so the problem **must** be rebuilt for exact re-optimization. However, the
KKT-based sensitivity analysis (which uses its own `ac_kkt` function
with the current `network.sw`) works correctly **not** requiring a model rebuild.
See GitHub issue #5.
"""
function update_switching!(prob::ACOPFProblem, sw::AbstractVector)
    m = prob.network.m
    @assert length(sw) == m "Switching vector length must match number of branches"
    @assert all(0 .<= sw .<= 1) "Switching values must be in [0,1]"

    # Invalidate sensitivity cache since parameters changed
    invalidate!(prob.cache)

    # Update network switching state
    prob.network.sw .= sw

    # Note: The JuMP model constraints embed the previous sw values as coefficients.
    # After calling this, the problem must be rebuilt for exact re-optimization.
    # However, the KKT-based sensitivity analysis (which uses its own ac_kkt function
    # with the current network.sw) works correctly without a model rebuild.

    return prob
end
