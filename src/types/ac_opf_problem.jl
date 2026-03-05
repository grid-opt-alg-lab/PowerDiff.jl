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
# ACOPFProblem: AC OPF Problem Type and Constructors
# =============================================================================
#
# Polar coordinate formulation of AC OPF wrapped around a JuMP model.
# Design mirrors DCOPFProblem for API consistency.

"""
    ACOPFSolution <: AbstractOPFSolution

Solution container for AC OPF problem, storing primal and dual variables.

# Fields
**Primal Variables**
- `va`: Voltage angles at each bus (n)
- `vm`: Voltage magnitudes at each bus (n)
- `pg`: Active power generation (k)
- `qg`: Reactive power generation (k)
- `p`: Active branch flows Dict{(l,i,j) => Float64}
- `q`: Reactive branch flows Dict{(l,i,j) => Float64}

**Dual Variables (Equality Constraints)**
- `nu_p_bal`: Active power balance duals (n) - used for LMP
- `nu_q_bal`: Reactive power balance duals (n)
- `nu_ref_bus`: Reference bus constraint duals (n_ref, usually 1)
- `nu_p_fr`, `nu_p_to`: Active flow definition duals (m each)
- `nu_q_fr`, `nu_q_to`: Reactive flow definition duals (m each)

**Dual Variables (Inequality Constraints)**
- `lam_thermal_fr`, `lam_thermal_to`: Thermal limit duals (m each)
- `lam_angle_lb`, `lam_angle_ub`: Angle difference limit duals (m each)
- `mu_vm_lb`, `mu_vm_ub`: Voltage magnitude bound duals (n each)
- `rho_pg_lb`, `rho_pg_ub`: Active gen bound duals (k each)
- `rho_qg_lb`, `rho_qg_ub`: Reactive gen bound duals (k each)
- `sig_p_fr_lb`, `sig_p_fr_ub`: From-side active flow bound duals (m each)
- `sig_q_fr_lb`, `sig_q_fr_ub`: From-side reactive flow bound duals (m each)
- `sig_p_to_lb`, `sig_p_to_ub`: To-side active flow bound duals (m each)
- `sig_q_to_lb`, `sig_q_to_ub`: To-side reactive flow bound duals (m each)

**Objective**
- `objective`: Optimal objective value
"""
Base.@kwdef struct ACOPFSolution <: AbstractOPFSolution
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
    nu_p_bal::Vector{Float64}
    nu_q_bal::Vector{Float64}

    # Dual - reference bus (equality)
    nu_ref_bus::Vector{Float64}

    # Dual - flow definition equations (equality, used in full-space stationarity)
    nu_p_fr::Vector{Float64}
    nu_p_to::Vector{Float64}
    nu_q_fr::Vector{Float64}
    nu_q_to::Vector{Float64}

    # Dual - thermal limits (inequality)
    lam_thermal_fr::Vector{Float64}
    lam_thermal_to::Vector{Float64}

    # Dual - angle difference limits (inequality)
    lam_angle_lb::Vector{Float64}
    lam_angle_ub::Vector{Float64}

    # Dual - voltage bounds (inequality)
    mu_vm_lb::Vector{Float64}
    mu_vm_ub::Vector{Float64}

    # Dual - generation bounds (inequality)
    rho_pg_lb::Vector{Float64}
    rho_pg_ub::Vector{Float64}
    rho_qg_lb::Vector{Float64}
    rho_qg_ub::Vector{Float64}

    # Dual - flow variable bounds (inequality, reduced-space)
    sig_p_fr_lb::Vector{Float64}
    sig_p_fr_ub::Vector{Float64}
    sig_q_fr_lb::Vector{Float64}
    sig_q_fr_ub::Vector{Float64}
    sig_p_to_lb::Vector{Float64}
    sig_p_to_ub::Vector{Float64}
    sig_q_to_lb::Vector{Float64}
    sig_q_to_ub::Vector{Float64}

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

AC OPF supports 6 parameter types (`:sw`, `:d`, `:qd`, `:cq`, `:cl`, `:fmax`),
each producing a separate `dz_d*` full-derivative matrix. All share one KKT LU
factorization (`kkt_factor`), so after the first parameter type is queried the
factorization is reused for subsequent parameters. Different operand queries
(e.g. `:va` vs `:pg` vs `:lmp`) for the *same* parameter type all extract
rows from the same cached `dz_d*` matrix — no recomputation needed.

# Fields
- `solution`: Cached ACOPFSolution (or nothing if not yet solved)
- `kkt_factor`: Cached LU factorization of KKT Jacobian (or nothing)
- `kkt_constants`: Cached KKT constants NamedTuple (or nothing)
- `dz_dsw`: Full KKT derivative w.r.t. switching (or nothing)
- `dz_dd`: Full KKT derivative w.r.t. active demand (or nothing)
- `dz_dqd`: Full KKT derivative w.r.t. reactive demand (or nothing)
- `dz_dcq`: Full KKT derivative w.r.t. quadratic cost (or nothing)
- `dz_dcl`: Full KKT derivative w.r.t. linear cost (or nothing)
- `dz_dfmax`: Full KKT derivative w.r.t. flow limits (or nothing)
"""
mutable struct ACSensitivityCache
    solution::Union{Nothing, ACOPFSolution}
    kkt_factor::Union{Nothing, Factorization}
    kkt_constants::Union{Nothing, NamedTuple}
    dz_dsw::Union{Nothing, Matrix{Float64}}
    dz_dd::Union{Nothing, Matrix{Float64}}
    dz_dqd::Union{Nothing, Matrix{Float64}}
    dz_dcq::Union{Nothing, Matrix{Float64}}
    dz_dcl::Union{Nothing, Matrix{Float64}}
    dz_dfmax::Union{Nothing, Matrix{Float64}}
end

"""
    ACSensitivityCache()

Create an empty AC sensitivity cache with all fields set to nothing.
"""
ACSensitivityCache() = ACSensitivityCache(nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)

"""
    invalidate!(cache::ACSensitivityCache)

Clear all cached AC sensitivity data. Called when problem parameters change.
"""
function invalidate!(cache::ACSensitivityCache)
    cache.solution = nothing
    cache.kkt_factor = nothing
    cache.kkt_constants = nothing
    cache.dz_dsw = nothing
    cache.dz_dd = nothing
    cache.dz_dqd = nothing
    cache.dz_dcq = nothing
    cache.dz_dcl = nothing
    cache.dz_dfmax = nothing
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
- `ref`: PowerModels-style reference dictionary (sequential keys)
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
    ACOPFProblem(network::ACNetwork; optimizer=Ipopt.Optimizer, silent=true)

Build a polar AC OPF problem from an ACNetwork that was constructed from a
PowerModels dictionary (i.e., `network.ref` must not be nothing).

Accepts both basic and non-basic networks; internally remaps to sequential indices.

# Arguments
- `network`: ACNetwork containing topology, admittances, and stored `ref`
- `optimizer`: JuMP-compatible optimizer (default: Ipopt)
- `silent`: Suppress solver output (default: true)

# Example
```julia
pm_data = PowerModels.parse_file("case5.m")
prob = ACOPFProblem(pm_data)
solve!(prob)
```
"""
function ACOPFProblem(
    network::ACNetwork;
    optimizer=Ipopt.Optimizer,
    silent::Bool=true
)
    if optimizer === Clarabel.Optimizer
        throw(ArgumentError(
            "Clarabel is a convex solver and cannot solve nonconvex AC OPF. " *
            "Use Ipopt.Optimizer (default) or another NLP solver."
        ))
    end

    isnothing(network.ref) && error(
        "ACOPFProblem requires an ACNetwork constructed from a PowerModels dict " *
        "(network.ref must not be nothing). Use ACOPFProblem(pm_data::Dict) instead.")

    id_map = network.id_map

    # Remap ref to sequential keys so JuMP/KKT code can iterate 1:n, 1:m, 1:k
    seq_ref = _remap_ref_to_sequential(network.ref, id_map)

    n_gen = length(seq_ref[:gen])
    gen_buses = [seq_ref[:gen][i]["gen_bus"] for i in 1:n_gen]

    prob = ACOPFProblem(
        JuMP.Model(), network,
        VariableRef[], VariableRef[], VariableRef[], VariableRef[],
        Dict{Tuple{Int,Int,Int}, VariableRef}(), Dict{Tuple{Int,Int,Int}, VariableRef}(),
        (;), seq_ref, gen_buses, n_gen, ACSensitivityCache(), optimizer, silent
    )
    _rebuild_jump_model!(prob)
    return prob
end

"""
Remap element dict entries to sequential indices, updating bus ID fields.
"""
function _remap_element_dict(ref_dict, to_idx::Dict{Int,Int}, bus_to_idx::Dict{Int,Int}, bus_fields::Vector{String})
    seq = Dict{Int, Any}()
    for (orig_id, elem) in ref_dict
        idx = to_idx[orig_id]
        new_elem = deepcopy(elem)
        new_elem["index"] = idx
        for f in bus_fields
            haskey(new_elem, f) && (new_elem[f] = bus_to_idx[new_elem[f]])
        end
        seq[idx] = new_elem
    end
    return seq
end

"""
Remap a `build_ref` result to sequential 1-based keys so that the JuMP model
and KKT code can iterate `for i in 1:n` without change.
"""
function _remap_ref_to_sequential(ref::Dict, id_map::IDMapping)
    seq_bus = _remap_element_dict(ref[:bus], id_map.bus_to_idx, id_map.bus_to_idx, ["bus_i"])
    seq_branch = _remap_element_dict(ref[:branch], id_map.branch_to_idx, id_map.bus_to_idx, ["f_bus", "t_bus"])
    seq_gen = _remap_element_dict(ref[:gen], id_map.gen_to_idx, id_map.bus_to_idx, ["gen_bus"])
    seq_load = _remap_element_dict(ref[:load], id_map.load_to_idx, id_map.bus_to_idx, ["load_bus"])
    seq_shunt = _remap_element_dict(ref[:shunt], id_map.shunt_to_idx, id_map.bus_to_idx, ["shunt_bus"])

    # Remap arcs: (l, i, j) → (seq_l, seq_i, seq_j)
    function _remap_arc(arc)
        (l, i, j) = arc
        return (id_map.branch_to_idx[l], id_map.bus_to_idx[i], id_map.bus_to_idx[j])
    end

    seq_arcs = [_remap_arc(a) for a in ref[:arcs]]
    seq_arcs_from = [_remap_arc(a) for a in ref[:arcs_from]]
    seq_arcs_to = [_remap_arc(a) for a in ref[:arcs_to]]

    # Remap bus_arcs, bus_gens, bus_loads, bus_shunts to sequential bus keys
    seq_bus_arcs = Dict{Int, Vector{Tuple{Int,Int,Int}}}()
    for (orig_bus, arcs) in ref[:bus_arcs]
        seq_bus_arcs[id_map.bus_to_idx[orig_bus]] = [_remap_arc(a) for a in arcs]
    end

    seq_bus_gens = Dict{Int, Vector{Int}}()
    for (orig_bus, gen_ids) in ref[:bus_gens]
        seq_bus_gens[id_map.bus_to_idx[orig_bus]] = [id_map.gen_to_idx[g] for g in gen_ids]
    end

    seq_bus_loads = Dict{Int, Vector{Int}}()
    for (orig_bus, load_ids) in ref[:bus_loads]
        seq_bus_loads[id_map.bus_to_idx[orig_bus]] = [id_map.load_to_idx[l] for l in load_ids]
    end

    seq_bus_shunts = Dict{Int, Vector{Int}}()
    for (orig_bus, shunt_ids) in ref[:bus_shunts]
        seq_bus_shunts[id_map.bus_to_idx[orig_bus]] = [id_map.shunt_to_idx[s] for s in shunt_ids]
    end

    # Remap ref_buses
    seq_ref_buses = Dict{Int, Any}()
    for (orig_bus, val) in ref[:ref_buses]
        seq_ref_buses[id_map.bus_to_idx[orig_bus]] = val
    end

    return Dict{Symbol, Any}(
        :bus => seq_bus,
        :gen => seq_gen,
        :branch => seq_branch,
        :load => seq_load,
        :shunt => seq_shunt,
        :arcs => seq_arcs,
        :arcs_from => seq_arcs_from,
        :arcs_to => seq_arcs_to,
        :bus_arcs => seq_bus_arcs,
        :bus_gens => seq_bus_gens,
        :bus_loads => seq_bus_loads,
        :bus_shunts => seq_bus_shunts,
        :ref_buses => seq_ref_buses,
    )
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

Accepts both basic and non-basic networks.
"""
function ACOPFProblem(pm_data::Dict; kwargs...)
    network = ACNetwork(pm_data)
    return ACOPFProblem(network; kwargs...)
end
