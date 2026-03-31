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
# Pretty-printing (Base.show) for public types and caches
# =============================================================================

# =============================================================================
# Property aliases: Unicode → ASCII (backward compatibility after field rename)
# =============================================================================

function _alias_getproperty(obj, aliases::Dict{Symbol,Symbol}, s::Symbol)
    s = get(aliases, s, s)
    return getfield(obj, s)
end

const _DC_NETWORK_ALIASES = Dict{Symbol,Symbol}(
    :Δθ_max => :angmax, :Δθ_min => :angmin, :τ => :tau,
)

const _DC_PF_STATE_ALIASES = Dict{Symbol,Symbol}(
    :θ => :va, :g => :pg,
)

const _DC_OPF_SOLUTION_ALIASES = Dict{Symbol,Symbol}(
    :θ => :va, :g => :pg,
    :ν_bal => :nu_bal, :ν_flow => :nu_flow,
    :λ_ub => :lam_ub, :λ_lb => :lam_lb,
    :ρ_ub => :rho_ub, :ρ_lb => :rho_lb,
    :μ_lb => :mu_lb, :μ_ub => :mu_ub,
    :γ_lb => :gamma_lb, :γ_ub => :gamma_ub,
)

const _DC_OPF_PROBLEM_ALIASES = Dict{Symbol,Symbol}(
    :θ => :va, :g => :pg,
)

const _AC_OPF_SOLUTION_ALIASES = Dict{Symbol,Symbol}(
    :ν_p_bal => :nu_p_bal, :ν_q_bal => :nu_q_bal, :ν_ref_bus => :nu_ref_bus,
    :ν_p_fr => :nu_p_fr, :ν_p_to => :nu_p_to, :ν_q_fr => :nu_q_fr, :ν_q_to => :nu_q_to,
    :λ_thermal_fr => :lam_thermal_fr, :λ_thermal_to => :lam_thermal_to,
    :λ_angle_lb => :lam_angle_lb, :λ_angle_ub => :lam_angle_ub,
    :μ_vm_lb => :mu_vm_lb, :μ_vm_ub => :mu_vm_ub,
    :ρ_pg_lb => :rho_pg_lb, :ρ_pg_ub => :rho_pg_ub,
    :ρ_qg_lb => :rho_qg_lb, :ρ_qg_ub => :rho_qg_ub,
    :σ_p_fr_lb => :sig_p_fr_lb, :σ_p_fr_ub => :sig_p_fr_ub,
    :σ_q_fr_lb => :sig_q_fr_lb, :σ_q_fr_ub => :sig_q_fr_ub,
    :σ_p_to_lb => :sig_p_to_lb, :σ_p_to_ub => :sig_p_to_ub,
    :σ_q_to_lb => :sig_q_to_lb, :σ_q_to_ub => :sig_q_to_ub,
)

Base.getproperty(net::DCNetwork, s::Symbol) = _alias_getproperty(net, _DC_NETWORK_ALIASES, s)
Base.getproperty(state::DCPowerFlowState, s::Symbol) = _alias_getproperty(state, _DC_PF_STATE_ALIASES, s)
Base.getproperty(sol::DCOPFSolution, s::Symbol) = _alias_getproperty(sol, _DC_OPF_SOLUTION_ALIASES, s)
Base.getproperty(prob::DCOPFProblem, s::Symbol) = _alias_getproperty(prob, _DC_OPF_PROBLEM_ALIASES, s)
Base.getproperty(sol::ACOPFSolution, s::Symbol) = _alias_getproperty(sol, _AC_OPF_SOLUTION_ALIASES, s)

# =============================================================================
# DCNetwork
# =============================================================================

function Base.show(io::IO, net::DCNetwork)
    print(io, "DCNetwork($(net.n) buses, $(net.m) branches, $(net.k) gens)")
end

function Base.show(io::IO, ::MIME"text/plain", net::DCNetwork)
    println(io, "DCNetwork ($(net.n) buses, $(net.m) branches, $(net.k) gens)")
    println(io, "  Reference bus: $(net.ref_bus)")
    n_open = count(x -> x < 1.0, net.sw)
    println(io, "  Open branches: $n_open/$(net.m)")
    println(io, "  Flow limits:   [$(round(minimum(net.fmax); digits=2)), $(round(maximum(net.fmax); digits=2))]")
    println(io, "  Gen capacity:  [$(round(minimum(net.gmin); digits=2)), $(round(maximum(net.gmax); digits=2))]")
    print(io, "  tau = $(net.tau)")
end

# =============================================================================
# ACNetwork
# =============================================================================

function Base.show(io::IO, net::ACNetwork)
    print(io, "ACNetwork($(net.n) buses, $(net.m) branches)")
end

function Base.show(io::IO, ::MIME"text/plain", net::ACNetwork)
    println(io, "ACNetwork ($(net.n) buses, $(net.m) branches)")
    println(io, "  Slack bus:  $(net.idx_slack)")
    println(io, "  Vm limits:  [$(round(minimum(net.vm_min); digits=2)), $(round(maximum(net.vm_max); digits=2))]")
    n_sw = count(net.is_switchable)
    print(io, "  Switchable: $n_sw/$(net.m) branches")
end

# =============================================================================
# DCPowerFlowState
# =============================================================================

function Base.show(io::IO, state::DCPowerFlowState)
    print(io, "DCPowerFlowState($(state.net.n) buses, $(state.net.m) branches)")
end

function Base.show(io::IO, ::MIME"text/plain", state::DCPowerFlowState)
    println(io, "DCPowerFlowState ($(state.net.n) buses, $(state.net.m) branches)")
    println(io, "  max|va|: $(round(maximum(abs, state.va); digits=4)) rad")
    println(io, "  max|f|: $(round(maximum(abs, state.f); digits=4)) p.u.")
    print(io, "  Total demand: $(round(sum(state.d); digits=2)) p.u.")
end

# =============================================================================
# ACPowerFlowState
# =============================================================================

function Base.show(io::IO, state::ACPowerFlowState)
    print(io, "ACPowerFlowState($(state.n) buses, $(state.m) branches)")
end

function Base.show(io::IO, ::MIME"text/plain", state::ACPowerFlowState)
    println(io, "ACPowerFlowState ($(state.n) buses, $(state.m) branches)")
    vm = abs.(state.v)
    va = angle.(state.v)
    println(io, "  |V| range:  [$(round(minimum(vm); digits=2)), $(round(maximum(vm); digits=2))]")
    println(io, "  ∠V range:   [$(round(minimum(va); digits=2)), $(round(maximum(va); digits=2))] rad")
    print(io, "  Slack bus:   $(state.idx_slack)")
end

# =============================================================================
# DCOPFSolution
# =============================================================================

function Base.show(io::IO, sol::DCOPFSolution)
    print(io, "DCOPFSolution(obj=$(round(sol.objective; digits=2)))")
end

function Base.show(io::IO, ::MIME"text/plain", sol::DCOPFSolution)
    println(io, "DCOPFSolution (objective = $(round(sol.objective; digits=2)))")
    println(io, "  Generators: $(length(sol.pg))  (range: [$(round(minimum(sol.pg); digits=2)), $(round(maximum(sol.pg); digits=2))])")
    println(io, "  Flows:      $(length(sol.f)) (max |f| = $(round(maximum(abs, sol.f); digits=2)))")
    print(io, "  Shedding:   $(round(sum(sol.psh); digits=2)) p.u. total")
end

# =============================================================================
# ACOPFSolution
# =============================================================================

function Base.show(io::IO, sol::ACOPFSolution)
    print(io, "ACOPFSolution(obj=$(round(sol.objective; digits=2)))")
end

function Base.show(io::IO, ::MIME"text/plain", sol::ACOPFSolution)
    println(io, "ACOPFSolution (objective = $(round(sol.objective; digits=2)))")
    println(io, "  |V| range: [$(round(minimum(sol.vm); digits=2)), $(round(maximum(sol.vm); digits=2))]")
    println(io, "  Pg range:  [$(round(minimum(sol.pg); digits=2)), $(round(maximum(sol.pg); digits=2))]")
    print(io, "  Qg range:  [$(round(minimum(sol.qg); digits=2)), $(round(maximum(sol.qg); digits=2))]")
end

# =============================================================================
# DCOPFProblem
# =============================================================================

function Base.show(io::IO, prob::DCOPFProblem)
    status = _problem_status_str(prob.model)
    print(io, "DCOPFProblem($(prob.network.n) buses, $status)")
end

function Base.show(io::IO, ::MIME"text/plain", prob::DCOPFProblem)
    net = prob.network
    println(io, "DCOPFProblem ($(net.n) buses, $(net.m) branches, $(net.k) gens)")
    println(io, "  Status:    $(_problem_status_str(prob.model))")
    if !isnothing(prob.cache.solution)
        println(io, "  Objective: $(round(prob.cache.solution.objective; digits=2))")
    end
    cached = _dc_cache_list(prob.cache)
    print(io, "  Cached:    $(isempty(cached) ? "none" : join(cached, ", "))")
end

# =============================================================================
# ACOPFProblem
# =============================================================================

function Base.show(io::IO, prob::ACOPFProblem)
    status = _problem_status_str(prob.model)
    print(io, "ACOPFProblem($(prob.network.n) buses, $status)")
end

function Base.show(io::IO, ::MIME"text/plain", prob::ACOPFProblem)
    net = prob.network
    println(io, "ACOPFProblem ($(net.n) buses, $(net.m) branches)")
    println(io, "  Status:    $(_problem_status_str(prob.model))")
    if !isnothing(prob.cache.solution)
        println(io, "  Objective: $(round(prob.cache.solution.objective; digits=2))")
    end
    cached = _ac_cache_list(prob.cache)
    print(io, "  Cached:    $(isempty(cached) ? "none" : join(cached, ", "))")
end

# =============================================================================
# DCSensitivityCache
# =============================================================================

const _DC_CACHE_FIELDS = (:solution, :kkt_factor, :dz_dd, :dz_dcl, :dz_dcq, :dz_dsw, :dz_dfmax, :dz_db)

function Base.show(io::IO, cache::DCSensitivityCache)
    n = count(f -> !isnothing(getfield(cache, f)), _DC_CACHE_FIELDS)
    print(io, "DCSensitivityCache($n/$(length(_DC_CACHE_FIELDS)) cached)")
end

function Base.show(io::IO, ::MIME"text/plain", cache::DCSensitivityCache)
    println(io, "DCSensitivityCache")
    for (i, f) in enumerate(_DC_CACHE_FIELDS)
        _show_cache_field(io, string(f), getfield(cache, f);
                          last=(i == length(_DC_CACHE_FIELDS)))
    end
end

# =============================================================================
# ACSensitivityCache
# =============================================================================

const _AC_CACHE_FIELDS = (:solution, :kkt_factor, :kkt_constants, :dz_dsw, :dz_dd, :dz_dqd, :dz_dcq, :dz_dcl, :dz_dfmax)

function Base.show(io::IO, cache::ACSensitivityCache)
    n = count(f -> !isnothing(getfield(cache, f)), _AC_CACHE_FIELDS)
    print(io, "ACSensitivityCache($n/$(length(_AC_CACHE_FIELDS)) cached)")
end

function Base.show(io::IO, ::MIME"text/plain", cache::ACSensitivityCache)
    println(io, "ACSensitivityCache")
    for (i, f) in enumerate(_AC_CACHE_FIELDS)
        _show_cache_field(io, string(f), getfield(cache, f);
                          last=(i == length(_AC_CACHE_FIELDS)))
    end
end

# =============================================================================
# Helpers (private)
# =============================================================================

function _problem_status_str(model::JuMP.Model)
    try
        status = JuMP.termination_status(model)
        return string(status)
    catch
        return "unknown (status query failed)"
    end
end

_problem_status_str(model::ExaModels.AbstractExaModel) = "ExaModel"
_problem_status_str(model) = "unknown"

function _dc_cache_list(cache::DCSensitivityCache)
    [string(f) for f in _DC_CACHE_FIELDS if !isnothing(getfield(cache, f))]
end

function _ac_cache_list(cache::ACSensitivityCache)
    [string(f) for f in _AC_CACHE_FIELDS if !isnothing(getfield(cache, f))]
end

function _show_cache_field(io::IO, name::String, value; last::Bool=false)
    mark = isnothing(value) ? "✗" : "✓"
    padded = rpad(name * ":", 16)
    if last
        print(io, "  $padded $mark")
    else
        println(io, "  $padded $mark")
    end
end
