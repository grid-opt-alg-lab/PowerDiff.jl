# =============================================================================
# ACOPFProblem: AC OPF Problem Type and Constructors
# =============================================================================
#
# Polar coordinate formulation of AC OPF wrapped around a JuMP model.
# Design mirrors DCOPFProblem for API consistency.

"""
    ACOPFSolution <: AbstractOPFSolution

Solution container for AC OPF problem, storing primal and dual variables.

# Fields
## Primal Variables
- `va`: Voltage angles at each bus (n)
- `vm`: Voltage magnitudes at each bus (n)
- `pg`: Active power generation (k)
- `qg`: Reactive power generation (k)
- `p`: Active branch flows Dict{(l,i,j) => Float64}
- `q`: Reactive branch flows Dict{(l,i,j) => Float64}

## Dual Variables (Equality Constraints)
- `ν_p_bal`: Active power balance duals (n) - used for LMP
- `ν_q_bal`: Reactive power balance duals (n)
- `ν_ref_bus`: Reference bus constraint duals (n_ref, usually 1)
- `ν_p_fr`, `ν_p_to`: Active flow definition duals (m each)
- `ν_q_fr`, `ν_q_to`: Reactive flow definition duals (m each)

## Dual Variables (Inequality Constraints)
- `λ_thermal_fr`, `λ_thermal_to`: Thermal limit duals (m each)
- `λ_angle_lb`, `λ_angle_ub`: Angle difference limit duals (m each)
- `μ_vm_lb`, `μ_vm_ub`: Voltage magnitude bound duals (n each)
- `ρ_pg_lb`, `ρ_pg_ub`: Active gen bound duals (k each)
- `ρ_qg_lb`, `ρ_qg_ub`: Reactive gen bound duals (k each)
- `σ_p_fr_lb`, `σ_p_fr_ub`: From-side active flow bound duals (m each)
- `σ_q_fr_lb`, `σ_q_fr_ub`: From-side reactive flow bound duals (m each)
- `σ_p_to_lb`, `σ_p_to_ub`: To-side active flow bound duals (m each)
- `σ_q_to_lb`, `σ_q_to_ub`: To-side reactive flow bound duals (m each)

## Objective
- `objective`: Optimal objective value
"""
struct ACOPFSolution <: AbstractOPFSolution
    # Primal - voltages
    va::Vector{Float64}
    vm::Vector{Float64}

    # Primal - generation
    pg::Vector{Float64}
    qg::Vector{Float64}

    # Primal - flows (stored as dicts for arc indexing compatibility)
    p::Dict{Tuple{Int,Int,Int}, Float64}
    q::Dict{Tuple{Int,Int,Int}, Float64}

    # Dual - power balance (equality)
    ν_p_bal::Vector{Float64}
    ν_q_bal::Vector{Float64}

    # Dual - reference bus (equality)
    ν_ref_bus::Vector{Float64}

    # Dual - flow definition equations (equality, used in full-space stationarity)
    ν_p_fr::Vector{Float64}
    ν_p_to::Vector{Float64}
    ν_q_fr::Vector{Float64}
    ν_q_to::Vector{Float64}

    # Dual - thermal limits (inequality)
    λ_thermal_fr::Vector{Float64}
    λ_thermal_to::Vector{Float64}

    # Dual - angle difference limits (inequality)
    λ_angle_lb::Vector{Float64}
    λ_angle_ub::Vector{Float64}

    # Dual - voltage bounds (inequality)
    μ_vm_lb::Vector{Float64}
    μ_vm_ub::Vector{Float64}

    # Dual - generation bounds (inequality)
    ρ_pg_lb::Vector{Float64}
    ρ_pg_ub::Vector{Float64}
    ρ_qg_lb::Vector{Float64}
    ρ_qg_ub::Vector{Float64}

    # Dual - flow variable bounds (inequality, reduced-space)
    σ_p_fr_lb::Vector{Float64}
    σ_p_fr_ub::Vector{Float64}
    σ_q_fr_lb::Vector{Float64}
    σ_q_fr_ub::Vector{Float64}
    σ_p_to_lb::Vector{Float64}
    σ_p_to_ub::Vector{Float64}
    σ_q_to_lb::Vector{Float64}
    σ_q_to_ub::Vector{Float64}

    # Objective
    objective::Float64
end

# =============================================================================
# ACSensitivityCache
# =============================================================================

"""
    ACSensitivityCache

Mutable cache for storing computed AC OPF sensitivity data to avoid redundant
KKT solves and ForwardDiff Jacobian evaluations.

# Fields
- `solution`: Cached ACOPFSolution (or nothing if not yet solved)
- `dz_dsw`: Full ∂z/∂sw derivative matrix w.r.t. switching (or nothing)
"""
mutable struct ACSensitivityCache
    solution::Union{Nothing, ACOPFSolution}
    dz_dsw::Union{Nothing, Matrix{Float64}}
end

"""
    ACSensitivityCache()

Create an empty AC sensitivity cache with all fields set to nothing.
"""
ACSensitivityCache() = ACSensitivityCache(nothing, nothing)

"""
    invalidate!(cache::ACSensitivityCache)

Clear all cached AC sensitivity data. Called when problem parameters change.
"""
function invalidate!(cache::ACSensitivityCache)
    cache.solution = nothing
    cache.dz_dsw = nothing
end

# =============================================================================
# ACOPFProblem
# =============================================================================

"""
    ACOPFProblem <: AbstractOPFProblem

Polar coordinate AC OPF wrapped around a JuMP model.

# Fields
- `model`: JuMP Model
- `network`: ACNetwork data
- `va`, `vm`: Variable references for voltage angles and magnitudes
- `pg`, `qg`: Variable references for active and reactive generation
- `p`, `q`: Dict of branch flow variable references (keyed by arc tuple)
- `cons`: Named tuple of constraint references
- `ref`: PowerModels-style reference dictionary
- `gen_buses`: Generator bus indices (maps generator index to bus index)
- `n_gen`: Number of generators
- `cache`: ACSensitivityCache for caching KKT derivatives
- `_optimizer`: Optimizer factory for model rebuilds (internal)
- `_silent`: Whether to suppress solver output (internal)
"""
mutable struct ACOPFProblem <: AbstractOPFProblem
    model::JuMP.Model
    network::ACNetwork
    va::Vector{VariableRef}
    vm::Vector{VariableRef}
    pg::Vector{VariableRef}
    qg::Vector{VariableRef}
    p::Dict{Tuple{Int,Int,Int}, VariableRef}
    q::Dict{Tuple{Int,Int,Int}, VariableRef}
    cons::NamedTuple
    ref::Dict{Symbol, Any}
    gen_buses::Vector{Int}
    n_gen::Int
    cache::ACSensitivityCache
    _optimizer::Any
    _silent::Bool
end

# =============================================================================
# ACOPFProblem Constructors
# =============================================================================

"""
    ACOPFProblem(network::ACNetwork, pm_data::Dict; optimizer=Ipopt.Optimizer, silent=true)

Build a polar AC OPF problem for the given network.

Uses PowerModels-style data dict for load/gen/branch parameters.

# Arguments
- `network`: ACNetwork containing topology and admittances
- `pm_data`: PowerModels data dictionary (basic network)
- `optimizer`: JuMP-compatible optimizer (default: Ipopt)
- `silent`: Suppress solver output (default: true)

# Example
```julia
pm_data = PowerModels.make_basic_network(PowerModels.parse_file("case5.m"))
net = ACNetwork(pm_data)
prob = ACOPFProblem(net, pm_data)
solve!(prob)
```
"""
function ACOPFProblem(
    network::ACNetwork,
    pm_data::Dict;
    optimizer=Ipopt.Optimizer,
    silent::Bool=true
)
    @assert haskey(pm_data, "basic_network") && pm_data["basic_network"] == true "Network must be a basic network"

    if optimizer === Clarabel.Optimizer
        @warn "Clarabel is a convex solver and cannot solve nonconvex AC OPF. " *
              "Use Ipopt.Optimizer (default) or another NLP solver."
    end

    # Build PowerModels reference structure
    PM.standardize_cost_terms!(pm_data, order=2)
    PM.calc_thermal_limits!(pm_data)
    ref = PM.build_ref(pm_data)[:it][:pm][:nw][0]

    n_gen = length(ref[:gen])
    gen_buses = [ref[:gen][i]["gen_bus"] for i in 1:n_gen]

    # Convert ref to Symbol-keyed dict for storage
    ref_sym = Dict{Symbol, Any}(
        :bus => ref[:bus],
        :gen => ref[:gen],
        :branch => ref[:branch],
        :load => ref[:load],
        :shunt => ref[:shunt],
        :arcs => ref[:arcs],
        :arcs_from => ref[:arcs_from],
        :arcs_to => ref[:arcs_to],
        :bus_arcs => ref[:bus_arcs],
        :bus_gens => ref[:bus_gens],
        :bus_loads => ref[:bus_loads],
        :bus_shunts => ref[:bus_shunts],
        :ref_buses => ref[:ref_buses]
    )

    prob = ACOPFProblem(
        JuMP.Model(), network,
        VariableRef[], VariableRef[], VariableRef[], VariableRef[],
        Dict{Tuple{Int,Int,Int}, VariableRef}(), Dict{Tuple{Int,Int,Int}, VariableRef}(),
        (;), ref_sym, gen_buses, n_gen, ACSensitivityCache(), optimizer, silent
    )
    _rebuild_jump_model!(prob)
    return prob
end

"""
    _rebuild_jump_model!(prob::ACOPFProblem)

Build (or rebuild) the JuMP model from current network parameters.
Called by the constructor and by `update_switching!` after mutating `network.sw`.
"""
function _rebuild_jump_model!(prob::ACOPFProblem)
    network = prob.network
    ref = prob.ref
    n, m = network.n, network.m
    n_gen = prob.n_gen

    # Create model
    model = JuMP.Model(prob._optimizer)
    prob._silent && set_silent(model)
    set_optimizer_attribute(model, "tol", 1e-6)

    # Voltage variables
    @variable(model, va[i in 1:n])
    @variable(model, ref[:bus][i]["vmin"] <= vm[i in 1:n] <= ref[:bus][i]["vmax"], start=1.0)

    # Generation variables
    @variable(model, ref[:gen][i]["pmin"] <= pg[i in 1:n_gen] <= ref[:gen][i]["pmax"])
    @variable(model, ref[:gen][i]["qmin"] <= qg[i in 1:n_gen] <= ref[:gen][i]["qmax"])

    # Branch flow variables
    p = Dict{Tuple{Int,Int,Int}, VariableRef}()
    q = Dict{Tuple{Int,Int,Int}, VariableRef}()
    for (l, i, j) in ref[:arcs]
        p[(l,i,j)] = @variable(model, base_name="p[$l,$i,$j]")
        q[(l,i,j)] = @variable(model, base_name="q[$l,$i,$j]")
        set_lower_bound(p[(l,i,j)], -ref[:branch][l]["rate_a"])
        set_upper_bound(p[(l,i,j)], ref[:branch][l]["rate_a"])
        set_lower_bound(q[(l,i,j)], -ref[:branch][l]["rate_a"])
        set_upper_bound(q[(l,i,j)], ref[:branch][l]["rate_a"])
    end

    # Objective: minimize generation cost (quadratic)
    @objective(model, Min,
        sum(gen["cost"][1]*pg[i]^2 + gen["cost"][2]*pg[i] + gen["cost"][3]
            for (i, gen) in ref[:gen])
    )

    # Reference bus constraint
    ref_bus_con = @constraint(model, [i in keys(ref[:ref_buses])], va[i] == 0)

    # Nodal power balance constraints
    p_bal_cons = Vector{ConstraintRef}(undef, n)
    q_bal_cons = Vector{ConstraintRef}(undef, n)

    for (i, bus) in ref[:bus]
        bus_loads = [ref[:load][l] for l in ref[:bus_loads][i]]
        bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]

        # Active power balance
        p_bal_cons[i] = @constraint(model,
            sum(p[a] for a in ref[:bus_arcs][i]) ==
            sum(pg[g] for g in ref[:bus_gens][i]) -
            sum(load["pd"] for load in bus_loads) -
            sum(shunt["gs"] for shunt in bus_shunts) * vm[i]^2
        )

        # Reactive power balance
        q_bal_cons[i] = @constraint(model,
            sum(q[a] for a in ref[:bus_arcs][i]) ==
            sum(qg[g] for g in ref[:bus_gens][i]) -
            sum(load["qd"] for load in bus_loads) +
            sum(shunt["bs"] for shunt in bus_shunts) * vm[i]^2
        )
    end

    # Branch power flow constraints and thermal limits
    p_fr_cons = Dict{Int, ConstraintRef}()
    q_fr_cons = Dict{Int, ConstraintRef}()
    p_to_cons = Dict{Int, ConstraintRef}()
    q_to_cons = Dict{Int, ConstraintRef}()
    thermal_fr_cons = Dict{Int, ConstraintRef}()
    thermal_to_cons = Dict{Int, ConstraintRef}()
    angle_diff_cons = Dict{Int, Vector{ConstraintRef}}()

    for (l, branch) in ref[:branch]
        f_idx = (l, branch["f_bus"], branch["t_bus"])
        t_idx = (l, branch["t_bus"], branch["f_bus"])

        p_fr = p[f_idx]
        q_fr = q[f_idx]
        p_to = p[t_idx]
        q_to = q[t_idx]

        vm_fr = vm[branch["f_bus"]]
        vm_to = vm[branch["t_bus"]]
        va_fr = va[branch["f_bus"]]
        va_to = va[branch["t_bus"]]

        # Branch parameters (incorporating switching state sw)
        g_br, b_br = PM.calc_branch_y(branch)
        tr, ti = PM.calc_branch_t(branch)
        g_fr_shunt = branch["g_fr"]
        b_fr_shunt = branch["b_fr"]
        g_to_shunt = branch["g_to"]
        b_to_shunt = branch["b_to"]
        tm = branch["tap"]^2

        # Scale by switching state
        sw_l = network.sw[l]

        # AC Power Flow Constraints (from side)
        p_fr_cons[l] = @constraint(model,
            p_fr == sw_l * ((g_br + g_fr_shunt)/tm * vm_fr^2 +
                    (-g_br*tr + b_br*ti)/tm * (vm_fr * vm_to * cos(va_fr - va_to)) +
                    (-b_br*tr - g_br*ti)/tm * (vm_fr * vm_to * sin(va_fr - va_to)))
        )

        q_fr_cons[l] = @constraint(model,
            q_fr == sw_l * (-(b_br + b_fr_shunt)/tm * vm_fr^2 -
                    (-b_br*tr - g_br*ti)/tm * (vm_fr * vm_to * cos(va_fr - va_to)) +
                    (-g_br*tr + b_br*ti)/tm * (vm_fr * vm_to * sin(va_fr - va_to)))
        )

        # AC Power Flow Constraints (to side)
        p_to_cons[l] = @constraint(model,
            p_to == sw_l * ((g_br + g_to_shunt) * vm_to^2 +
                    (-g_br*tr - b_br*ti)/tm * (vm_to * vm_fr * cos(va_to - va_fr)) +
                    (-b_br*tr + g_br*ti)/tm * (vm_to * vm_fr * sin(va_to - va_fr)))
        )

        q_to_cons[l] = @constraint(model,
            q_to == sw_l * (-(b_br + b_to_shunt) * vm_to^2 -
                    (-b_br*tr + g_br*ti)/tm * (vm_to * vm_fr * cos(va_fr - va_to)) +
                    (-g_br*tr - b_br*ti)/tm * (vm_to * vm_fr * sin(va_to - va_fr)))
        )

        # Angle difference limits
        angle_diff_cons[l] = [
            @constraint(model, va_fr - va_to >= branch["angmin"]),
            @constraint(model, va_fr - va_to <= branch["angmax"])
        ]

        # Thermal limits (apparent power)
        thermal_fr_cons[l] = @constraint(model, p_fr^2 + q_fr^2 <= branch["rate_a"]^2)
        thermal_to_cons[l] = @constraint(model, p_to^2 + q_to^2 <= branch["rate_a"]^2)
    end

    prob.model = model
    prob.va = collect(va)
    prob.vm = collect(vm)
    prob.pg = collect(pg)
    prob.qg = collect(qg)
    prob.p = p
    prob.q = q
    prob.cons = (
        ref_bus = ref_bus_con,
        p_bal = p_bal_cons,
        q_bal = q_bal_cons,
        p_fr = p_fr_cons,
        q_fr = q_fr_cons,
        p_to = p_to_cons,
        q_to = q_to_cons,
        thermal_fr = thermal_fr_cons,
        thermal_to = thermal_to_cons,
        angle_diff = angle_diff_cons
    )

    return nothing
end

"""
    ACOPFProblem(pm_data::Dict; kwargs...)

Convenience constructor: build ACOPFProblem directly from PowerModels dict.
"""
function ACOPFProblem(pm_data::Dict; kwargs...)
    if !haskey(pm_data, "basic_network") || !pm_data["basic_network"]
        pm_data = PM.make_basic_network(pm_data)
    end
    network = ACNetwork(pm_data)
    return ACOPFProblem(network, pm_data; kwargs...)
end
