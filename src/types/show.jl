# =============================================================================
# Pretty-printing (Base.show) for public types and caches
# =============================================================================

# =============================================================================
# Property aliases: .g → .pg (mirrors :g → :pg alias in sensitivity API)
# =============================================================================

function Base.getproperty(state::DCPowerFlowState, s::Symbol)
    s === :g && return getfield(state, :pg)
    return getfield(state, s)
end

function Base.getproperty(sol::DCOPFSolution, s::Symbol)
    s === :g && return getfield(sol, :pg)
    return getfield(sol, s)
end

function Base.getproperty(prob::DCOPFProblem, s::Symbol)
    s === :g && return getfield(prob, :pg)
    return getfield(prob, s)
end

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

function Base.show(io::IO, cache::DCSensitivityCache)
    n = _dc_cache_count(cache)
    print(io, "DCSensitivityCache($n/8 cached)")
end

function Base.show(io::IO, ::MIME"text/plain", cache::DCSensitivityCache)
    println(io, "DCSensitivityCache")
    _show_cache_field(io, "solution",   cache.solution)
    _show_cache_field(io, "kkt_factor", cache.kkt_factor)
    _show_cache_field(io, "dz_dd",      cache.dz_dd)
    _show_cache_field(io, "dz_dcl",     cache.dz_dcl)
    _show_cache_field(io, "dz_dcq",     cache.dz_dcq)
    _show_cache_field(io, "dz_dsw",     cache.dz_dsw)
    _show_cache_field(io, "dz_dfmax",   cache.dz_dfmax)
    _show_cache_field(io, "dz_db",      cache.dz_db; last=true)
end

# =============================================================================
# ACSensitivityCache
# =============================================================================

function Base.show(io::IO, cache::ACSensitivityCache)
    n = count(!isnothing, (cache.solution, cache.dz_dsw))
    print(io, "ACSensitivityCache($n/2 cached)")
end

function Base.show(io::IO, ::MIME"text/plain", cache::ACSensitivityCache)
    println(io, "ACSensitivityCache")
    _show_cache_field(io, "solution", cache.solution)
    _show_cache_field(io, "dz_dsw",  cache.dz_dsw; last=true)
end

# =============================================================================
# Helpers (private)
# =============================================================================

function _problem_status_str(model::JuMP.Model)
    try
        status = JuMP.termination_status(model)
        return string(status)
    catch
        return "not solved"
    end
end

function _dc_cache_count(cache::DCSensitivityCache)
    return count(!isnothing, (
        cache.solution, cache.kkt_factor,
        cache.dz_dd, cache.dz_dcl, cache.dz_dcq,
        cache.dz_dsw, cache.dz_dfmax, cache.dz_db
    ))
end

function _dc_cache_list(cache::DCSensitivityCache)
    names = String[]
    !isnothing(cache.solution)   && push!(names, "solution")
    !isnothing(cache.kkt_factor) && push!(names, "kkt_factor")
    !isnothing(cache.dz_dd)      && push!(names, "dz_dd")
    !isnothing(cache.dz_dcl)     && push!(names, "dz_dcl")
    !isnothing(cache.dz_dcq)     && push!(names, "dz_dcq")
    !isnothing(cache.dz_dsw)     && push!(names, "dz_dsw")
    !isnothing(cache.dz_dfmax)   && push!(names, "dz_dfmax")
    !isnothing(cache.dz_db)      && push!(names, "dz_db")
    return names
end

function _ac_cache_list(cache::ACSensitivityCache)
    names = String[]
    !isnothing(cache.solution) && push!(names, "solution")
    !isnothing(cache.dz_dsw)   && push!(names, "dz_dsw")
    return names
end

function _show_cache_field(io::IO, name::String, value; last::Bool=false)
    mark = isnothing(value) ? "✗" : "✓"
    padded = rpad(name * ":", 12)
    if last
        print(io, "  $padded $mark")
    else
        println(io, "  $padded $mark")
    end
end
