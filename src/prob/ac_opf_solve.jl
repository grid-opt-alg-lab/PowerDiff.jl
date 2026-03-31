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
# AC OPF Problem Solving and Operations
# =============================================================================
#
# Functions for solving AC OPF problems and updating parameters.

function _check_solve_status(stats, label::String)
    status = getproperty(stats, :status)
    status == :first_order && return status
    if status == :acceptable
        @warn "$label converged at acceptable tolerance"
        return status
    end
    if status == :infeasible
        error("$label is infeasible. Check that demand is feasible given generator capacities and network constraints.")
    elseif status == :max_iter
        error("$label solver reached iteration limit.")
    elseif status == :max_time
        error("$label solver reached time limit.")
    else
        error("$label failed with status: $status")
    end
end

"""
    solve!(prob::ACOPFProblem)

Solve the AC OPF problem and return an ACOPFSolution.

Invalidates the sensitivity cache since the solution may have changed.

# Returns
ACOPFSolution containing optimal primal and dual variables.

# Throws
Error if optimization does not converge to optimal/locally optimal solution.
"""
function solve!(prob::ACOPFProblem{JuMPBackend})
    # Invalidate sensitivity cache since we're re-solving
    invalidate!(prob.cache)

    optimize!(prob.model)

    _check_solve_status(prob.model, "AC OPF")

    sol = _extract_ac_opf_solution(prob)
    prob.cache.solution = sol
    return sol
end

function solve!(prob::ACOPFProblem{ExaBackend})
    # Invalidate sensitivity cache since we're re-solving
    invalidate!(prob.cache)

    result = NLPModelsIpopt.ipopt(prob.model; print_level = prob._silent ? 0 : 5)

    _check_solve_status(result, "AC OPF")

    sol = _extract_ac_opf_solution(prob, result)
    prob.cache.solution = sol

    return sol
end

"""
Extract solution from solved AC OPF problem.
"""
function _extract_ac_opf_solution(prob::ACOPFProblem{JuMPBackend})
    n = prob.network.n
    m = prob.network.m
    k = prob.n_gen

    va_val = value.(prob.va)
    vm_val = value.(prob.vm)
    pg_val = value.(prob.pg)
    qg_val = value.(prob.qg)
    p_arr = value.(prob.p)
    q_arr = value.(prob.q)

    p_val = Dict(prob.data.arcs[i] => p_arr[i] for i in eachindex(prob.data.arcs))
    q_val = Dict(prob.data.arcs[i] => q_arr[i] for i in eachindex(prob.data.arcs))

    ν_p_bal = dual.(prob.cons.p_bal)
    ν_q_bal = dual.(prob.cons.q_bal)
    ν_ref_bus = dual.(prob.cons.ref_bus)
    ν_p_fr = dual.(prob.cons.p_fr)
    ν_p_to = dual.(prob.cons.p_to)
    ν_q_fr = dual.(prob.cons.q_fr)
    ν_q_to = dual.(prob.cons.q_to)
    λ_thermal_fr = dual.(prob.cons.thermal_fr)
    λ_thermal_to = dual.(prob.cons.thermal_to)
    λ_angle_lb = dual.(prob.cons.angle_diff_lb)
    λ_angle_ub = dual.(prob.cons.angle_diff_ub)
    μ_vm_lb = [dual(LowerBoundRef(prob.vm[i])) for i in 1:n]
    μ_vm_ub = [dual(UpperBoundRef(prob.vm[i])) for i in 1:n]
    ρ_pg_lb = [dual(LowerBoundRef(prob.pg[i])) for i in 1:k]
    ρ_pg_ub = [dual(UpperBoundRef(prob.pg[i])) for i in 1:k]
    ρ_qg_lb = [dual(LowerBoundRef(prob.qg[i])) for i in 1:k]
    ρ_qg_ub = [dual(UpperBoundRef(prob.qg[i])) for i in 1:k]

    σ_p_fr_lb = zeros(m)
    σ_p_fr_ub = zeros(m)
    σ_q_fr_lb = zeros(m)
    σ_q_fr_ub = zeros(m)
    σ_p_to_lb = zeros(m)
    σ_p_to_ub = zeros(m)
    σ_q_to_lb = zeros(m)
    σ_q_to_ub = zeros(m)
    for l in 1:m
        f_idx = prob.data.arc_from_idx[l]
        t_idx = prob.data.arc_to_idx[l]
        σ_p_fr_lb[l] = dual(LowerBoundRef(prob.p[f_idx]))
        σ_p_fr_ub[l] = dual(UpperBoundRef(prob.p[f_idx]))
        σ_q_fr_lb[l] = dual(LowerBoundRef(prob.q[f_idx]))
        σ_q_fr_ub[l] = dual(UpperBoundRef(prob.q[f_idx]))
        σ_p_to_lb[l] = dual(LowerBoundRef(prob.p[t_idx]))
        σ_p_to_ub[l] = dual(UpperBoundRef(prob.p[t_idx]))
        σ_q_to_lb[l] = dual(LowerBoundRef(prob.q[t_idx]))
        σ_q_to_ub[l] = dual(UpperBoundRef(prob.q[t_idx]))
    end

    return ACOPFSolution(
        va = va_val, vm = vm_val,
        pg = pg_val, qg = qg_val,
        p = p_val, q = q_val,
        nu_p_bal = ν_p_bal, nu_q_bal = ν_q_bal,
        nu_ref_bus = ν_ref_bus,
        nu_p_fr = ν_p_fr, nu_p_to = ν_p_to, nu_q_fr = ν_q_fr, nu_q_to = ν_q_to,
        lam_thermal_fr = λ_thermal_fr, lam_thermal_to = λ_thermal_to,
        lam_angle_lb = λ_angle_lb, lam_angle_ub = λ_angle_ub,
        mu_vm_lb = μ_vm_lb, mu_vm_ub = μ_vm_ub,
        rho_pg_lb = ρ_pg_lb, rho_pg_ub = ρ_pg_ub, rho_qg_lb = ρ_qg_lb, rho_qg_ub = ρ_qg_ub,
        sig_p_fr_lb = σ_p_fr_lb, sig_p_fr_ub = σ_p_fr_ub,
        sig_q_fr_lb = σ_q_fr_lb, sig_q_fr_ub = σ_q_fr_ub,
        sig_p_to_lb = σ_p_to_lb, sig_p_to_ub = σ_p_to_ub,
        sig_q_to_lb = σ_q_to_lb, sig_q_to_ub = σ_q_to_ub,
        objective = objective_value(prob.model)
    )
end

function _extract_ac_opf_solution(prob::ACOPFProblem{ExaBackend}, result)
    m = prob.network.m

    va_val = ExaModels.solution(result, prob.va)
    vm_val = ExaModels.solution(result, prob.vm)
    pg_val = ExaModels.solution(result, prob.pg)
    qg_val = ExaModels.solution(result, prob.qg)
    p_arr = ExaModels.solution(result, prob.p)
    q_arr = ExaModels.solution(result, prob.q)

    p_val = Dict(prob.data.arcs[i] => p_arr[i] for i in eachindex(prob.data.arcs))
    q_val = Dict(prob.data.arcs[i] => q_arr[i] for i in eachindex(prob.data.arcs))

    ν_p_bal = ExaModels.multipliers(result, prob.cons.p_bal)
    ν_q_bal = ExaModels.multipliers(result, prob.cons.q_bal)
    ν_ref_bus = ExaModels.multipliers(result, prob.cons.ref_bus)
    ν_p_fr = ExaModels.multipliers(result, prob.cons.p_fr)
    ν_p_to = ExaModels.multipliers(result, prob.cons.p_to)
    ν_q_fr = ExaModels.multipliers(result, prob.cons.q_fr)
    ν_q_to = ExaModels.multipliers(result, prob.cons.q_to)
    λ_thermal_fr = ExaModels.multipliers(result, prob.cons.thermal_fr)
    λ_thermal_to = ExaModels.multipliers(result, prob.cons.thermal_to)
    λ_angle_lb = ExaModels.multipliers(result, prob.cons.angle_diff_lb)
    λ_angle_ub = ExaModels.multipliers(result, prob.cons.angle_diff_ub)
    μ_vm_lb = ExaModels.multipliers_L(result, prob.vm)
    μ_vm_ub = ExaModels.multipliers_U(result, prob.vm)
    ρ_pg_lb = ExaModels.multipliers_L(result, prob.pg)
    ρ_pg_ub = ExaModels.multipliers_U(result, prob.pg)
    ρ_qg_lb = ExaModels.multipliers_L(result, prob.qg)
    ρ_qg_ub = ExaModels.multipliers_U(result, prob.qg)

    p_lb = ExaModels.multipliers_L(result, prob.p)
    p_ub = ExaModels.multipliers_U(result, prob.p)
    q_lb = ExaModels.multipliers_L(result, prob.q)
    q_ub = ExaModels.multipliers_U(result, prob.q)
    σ_p_fr_lb = p_lb[prob.data.arc_from_idx]
    σ_p_fr_ub = p_ub[prob.data.arc_from_idx]
    σ_q_fr_lb = q_lb[prob.data.arc_from_idx]
    σ_q_fr_ub = q_ub[prob.data.arc_from_idx]
    σ_p_to_lb = p_lb[prob.data.arc_to_idx]
    σ_p_to_ub = p_ub[prob.data.arc_to_idx]
    σ_q_to_lb = q_lb[prob.data.arc_to_idx]
    σ_q_to_ub = q_ub[prob.data.arc_to_idx]

    return ACOPFSolution(
        va = va_val, vm = vm_val,
        pg = pg_val, qg = qg_val,
        p = p_val, q = q_val,
        nu_p_bal = ν_p_bal, nu_q_bal = ν_q_bal,
        nu_ref_bus = ν_ref_bus,
        nu_p_fr = ν_p_fr, nu_p_to = ν_p_to, nu_q_fr = ν_q_fr, nu_q_to = ν_q_to,
        lam_thermal_fr = λ_thermal_fr, lam_thermal_to = λ_thermal_to,
        lam_angle_lb = λ_angle_lb, lam_angle_ub = λ_angle_ub,
        mu_vm_lb = μ_vm_lb, mu_vm_ub = μ_vm_ub,
        rho_pg_lb = ρ_pg_lb, rho_pg_ub = ρ_pg_ub, rho_qg_lb = ρ_qg_lb, rho_qg_ub = ρ_qg_ub,
        sig_p_fr_lb = σ_p_fr_lb, sig_p_fr_ub = σ_p_fr_ub,
        sig_q_fr_lb = σ_q_fr_lb, sig_q_fr_ub = σ_q_fr_ub,
        sig_p_to_lb = σ_p_to_lb, sig_p_to_ub = σ_p_to_ub,
        sig_q_to_lb = σ_q_to_lb, sig_q_to_ub = σ_q_to_ub,
        objective = result.objective
    )
end

"""
    update_switching!(prob::ACOPFProblem, sw::AbstractVector)

Update the network switching state, invalidate the sensitivity cache, and
rebuild the JuMP model so that `solve!(prob)` uses the new switching state.

# Arguments
- `prob`: ACOPFProblem to update
- `sw`: New switching state vector (length m), values in [0,1]
"""
function update_switching!(prob::ACOPFProblem, sw::AbstractVector)
    m = prob.network.m
    length(sw) == m || throw(DimensionMismatch("Switching vector length $(length(sw)) must match number of branches $m"))
    all(0 .<= sw .<= 1) || throw(ArgumentError("Switching values must be in [0,1]"))

    # Invalidate sensitivity cache since parameters changed
    invalidate!(prob.cache)

    # Update network switching state
    prob.network.sw .= sw

    _rebuild_model!(prob)

    return prob
end
