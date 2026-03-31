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
KKT solves and KKT Jacobian evaluations.

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
    cache.dz_dsw = nothing
    cache.dz_dd = nothing
    cache.dz_dqd = nothing
    cache.dz_dcq = nothing
    cache.dz_dcl = nothing
    cache.dz_dfmax = nothing
    return nothing
end

abstract type AbstractACOPFBackend end
struct JuMPBackend <: AbstractACOPFBackend end
struct ExaBackend <: AbstractACOPFBackend end

struct ACOPFData{C}
    arcs::Vector{NTuple{3,Int}}
    arc_from_idx::Vector{Int}
    arc_to_idx::Vector{Int}
    bus_arc_idxs::Vector{Vector{Int}}
    bus_gen_idxs::Vector{Vector{Int}}
    ref_bus_keys::Vector{Int}
    constants::C
end

struct ACJuMPConstraints
    ref_bus::Vector{ConstraintRef}
    p_bal::Vector{ConstraintRef}
    q_bal::Vector{ConstraintRef}
    p_fr::Vector{ConstraintRef}
    q_fr::Vector{ConstraintRef}
    p_to::Vector{ConstraintRef}
    q_to::Vector{ConstraintRef}
    thermal_fr::Vector{ConstraintRef}
    thermal_to::Vector{ConstraintRef}
    angle_diff_lb::Vector{ConstraintRef}
    angle_diff_ub::Vector{ConstraintRef}
end

struct ACExaConstraints{RB,PB,QB,PF,QF,PT,QT,TF,TT,AL,AU}
    ref_bus::RB
    p_bal::PB
    q_bal::QB
    p_fr::PF
    q_fr::QF
    p_to::PT
    q_to::QT
    thermal_fr::TF
    thermal_to::TT
    angle_diff_lb::AL
    angle_diff_ub::AU
end

"""
    ACOPFProblem <: AbstractOPFProblem

Polar coordinate AC OPF backed by either a JuMP model or an ExaModels model.
"""
mutable struct ACOPFProblem{B<:AbstractACOPFBackend,M,D,VA,VM,PG,QG,P,Q,C,O} <: AbstractOPFProblem
    model::M
    network::ACNetwork
    data::D
    va::VA
    vm::VM
    pg::PG
    qg::QG
    p::P
    q::Q
    cons::C
    gen_buses::Vector{Int}
    n_gen::Int
    cache::ACSensitivityCache
    _backend::B
    _optimizer::O
    _silent::Bool
end

# =============================================================================
# ACOPFProblem Constructors
# =============================================================================

"""
    ACOPFProblem(network::ACNetwork; optimizer=Ipopt.Optimizer, silent=true)

Build a polar AC OPF problem from an ACNetwork.

Accepts both basic and non-basic networks; internally remaps to sequential indices.

# Arguments
- `network`: ACNetwork containing topology, admittances, and typed branch/gen data
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
    backend::Symbol=:jump,
    optimizer=Ipopt.Optimizer,
    silent::Bool=true
)
    backend_tag = _ac_backend_tag(backend)
    data = _build_acopf_data(network)
    return _acopf_problem(network, data, backend_tag; optimizer=optimizer, silent=silent)
end

function _ac_backend_tag(backend::Symbol)
    backend == :jump && return JuMPBackend()
    backend == :exa && return ExaBackend()
    throw(ArgumentError("unsupported ACOPF backend :$backend (expected :jump or :exa)"))
end

function _build_acopf_data(network::ACNetwork)
    n, m = network.n, network.m
    k = length(network.gen_bus)
    g_br = copy(network.g)
    b_br = copy(network.b)
    tr = network.tap .* cos.(network.shift)
    ti = network.tap .* sin.(network.shift)
    g_fr = copy(network.g_fr)
    b_fr = copy(network.b_fr)
    g_to = copy(network.g_to)
    b_to = copy(network.b_to)
    tm = copy(network.tm)
    f_bus = copy(network.f_bus)
    t_bus = copy(network.t_bus)
    angmin = copy(network.angmin)
    angmax = copy(network.angmax)
    fmax = copy(network.rate_a)
    arcs = Vector{NTuple{3,Int}}(undef, 2m)
    arc_from_idx = Vector{Int}(undef, m)
    arc_to_idx = Vector{Int}(undef, m)
    bus_arc_idxs = [Int[] for _ in 1:n]

    for l in 1:m
        from_idx = 2l - 1
        to_idx = 2l
        arc_from_idx[l] = from_idx
        arc_to_idx[l] = to_idx
        arcs[from_idx] = (l, f_bus[l], t_bus[l])
        arcs[to_idx] = (l, t_bus[l], f_bus[l])
        push!(bus_arc_idxs[f_bus[l]], from_idx)
        push!(bus_arc_idxs[t_bus[l]], to_idx)
    end

    vmin = copy(network.vm_min)
    vmax = copy(network.vm_max)
    gs = copy(network.gs)
    bs = copy(network.bs)
    pd = copy(network.pd)
    qd = copy(network.qd)
    ref_bus_keys = copy(network.ref_bus_keys)
    pmin = copy(network.pmin)
    pmax = copy(network.pmax)
    qmin = copy(network.qmin)
    qmax = copy(network.qmax)
    gen_bus = copy(network.gen_bus)
    cq = copy(network.cq)
    cl = copy(network.cl)
    cc = copy(network.cc)
    bus_gen_idxs = [Int[] for _ in 1:n]

    for i in 1:k
        push!(bus_gen_idxs[gen_bus[i]], i)
    end

    constants = (
        g_br = g_br, b_br = b_br, tr = tr, ti = ti,
        g_fr = g_fr, b_fr = b_fr, g_to = g_to, b_to = b_to,
        tm = tm, f_bus = f_bus, t_bus = t_bus,
        angmin = angmin, angmax = angmax,
        vmin = vmin, vmax = vmax,
        pmin = pmin, pmax = pmax, qmin = qmin, qmax = qmax,
        gen_bus = gen_bus, cq = cq, cl = cl, cc = cc,
        fmax = fmax, gs = gs, bs = bs, pd = pd, qd = qd,
        ref_bus_keys = ref_bus_keys,
    )

    return ACOPFData(arcs, arc_from_idx, arc_to_idx, bus_arc_idxs, bus_gen_idxs, ref_bus_keys, constants)
end

"""
    _rebuild_jump_model!(prob::ACOPFProblem)

Build (or rebuild) the JuMP model from current network parameters.
Called by the constructor and by `update_switching!` after mutating `network.sw`.
"""
function _acopf_problem(network::ACNetwork, data::ACOPFData, ::JuMPBackend; optimizer, silent::Bool)
    model, va, vm, pg, qg, p, q, cons = _build_jump_model(network, data, optimizer, silent)
    cache = ACSensitivityCache()
    cache.kkt_constants = data.constants
    gen_buses = copy(data.constants.gen_bus)
    return ACOPFProblem(model, network, data, va, vm, pg, qg, p, q, cons,
        gen_buses, length(gen_buses), cache, JuMPBackend(), optimizer, silent)
end

function _acopf_problem(network::ACNetwork, data::ACOPFData, ::ExaBackend; optimizer, silent::Bool)
    model, va, vm, pg, qg, p, q, cons = _build_examodel(network, data, optimizer, silent)
    cache = ACSensitivityCache()
    cache.kkt_constants = data.constants
    gen_buses = copy(data.constants.gen_bus)
    return ACOPFProblem(model, network, data, va, vm, pg, qg, p, q, cons,
        gen_buses, length(gen_buses), cache, ExaBackend(), optimizer, silent)
end

function _rebuild_model!(prob::ACOPFProblem{JuMPBackend})
    data = _build_acopf_data(prob.network)
    model, va, vm, pg, qg, p, q, cons = _build_jump_model(prob.network, data, prob._optimizer, prob._silent)
    prob.data = data
    prob.model = model
    prob.va = va
    prob.vm = vm
    prob.pg = pg
    prob.qg = qg
    prob.p = p
    prob.q = q
    prob.cons = cons
    prob.gen_buses = copy(data.constants.gen_bus)
    prob.n_gen = length(prob.gen_buses)
    prob.cache.kkt_constants = data.constants
    return nothing
end

function _rebuild_model!(prob::ACOPFProblem{ExaBackend})
    data = _build_acopf_data(prob.network)
    model, va, vm, pg, qg, p, q, cons = _build_examodel(prob.network, data, prob._optimizer, prob._silent)
    prob.data = data
    prob.model = model
    prob.va = va
    prob.vm = vm
    prob.pg = pg
    prob.qg = qg
    prob.p = p
    prob.q = q
    prob.cons = cons
    prob.gen_buses = copy(data.constants.gen_bus)
    prob.n_gen = length(prob.gen_buses)
    prob.cache.kkt_constants = data.constants
    return nothing
end

function _build_jump_model(network::ACNetwork, data::ACOPFData, optimizer, silent::Bool)
    constants = data.constants
    n, m = network.n, network.m
    n_gen = length(constants.gen_bus)
    arc_fmax = [constants.fmax[arc[1]] for arc in data.arcs]

    model = JuMP.Model(optimizer)
    silent && set_silent(model)
    set_optimizer_attribute(model, "tol", 1e-6)

    @variable(model, va[1:n])
    @variable(model, constants.vmin[i] <= vm[i in 1:n] <= constants.vmax[i], start=1.0)
    @variable(model, constants.pmin[i] <= pg[i in 1:n_gen] <= constants.pmax[i])
    @variable(model, constants.qmin[i] <= qg[i in 1:n_gen] <= constants.qmax[i])
    @variable(model, -arc_fmax[i] <= p[i in 1:length(data.arcs)] <= arc_fmax[i])
    @variable(model, -arc_fmax[i] <= q[i in 1:length(data.arcs)] <= arc_fmax[i])

    @objective(model, Min,
        sum(constants.cq[i] * pg[i]^2 + constants.cl[i] * pg[i] + constants.cc[i] for i in 1:n_gen)
    )

    ref_bus = [@constraint(model, va[i] == 0) for i in data.ref_bus_keys]
    p_bal = Vector{ConstraintRef}(undef, n)
    q_bal = Vector{ConstraintRef}(undef, n)
    for i in 1:n
        p_bal[i] = @constraint(model,
            sum(p[j] for j in data.bus_arc_idxs[i]) ==
            sum(pg[g] for g in data.bus_gen_idxs[i]) - constants.pd[i] - constants.gs[i] * vm[i]^2
        )
        q_bal[i] = @constraint(model,
            sum(q[j] for j in data.bus_arc_idxs[i]) ==
            sum(qg[g] for g in data.bus_gen_idxs[i]) - constants.qd[i] + constants.bs[i] * vm[i]^2
        )
    end

    p_fr = Vector{ConstraintRef}(undef, m)
    q_fr = Vector{ConstraintRef}(undef, m)
    p_to = Vector{ConstraintRef}(undef, m)
    q_to = Vector{ConstraintRef}(undef, m)
    thermal_fr = Vector{ConstraintRef}(undef, m)
    thermal_to = Vector{ConstraintRef}(undef, m)
    angle_diff_lb = Vector{ConstraintRef}(undef, m)
    angle_diff_ub = Vector{ConstraintRef}(undef, m)

    for l in 1:m
        f_idx = data.arc_from_idx[l]
        t_idx = data.arc_to_idx[l]
        fb = constants.f_bus[l]
        tb = constants.t_bus[l]
        sw_l = network.sw[l]

        p_fr[l] = @constraint(model,
            p[f_idx] == sw_l * ((constants.g_br[l] + constants.g_fr[l]) / constants.tm[l] * vm[fb]^2 +
                (-constants.g_br[l] * constants.tr[l] + constants.b_br[l] * constants.ti[l]) / constants.tm[l] *
                (vm[fb] * vm[tb] * cos(va[fb] - va[tb])) +
                (-constants.b_br[l] * constants.tr[l] - constants.g_br[l] * constants.ti[l]) / constants.tm[l] *
                (vm[fb] * vm[tb] * sin(va[fb] - va[tb])))
        )
        q_fr[l] = @constraint(model,
            q[f_idx] == sw_l * (-(constants.b_br[l] + constants.b_fr[l]) / constants.tm[l] * vm[fb]^2 -
                (-constants.b_br[l] * constants.tr[l] - constants.g_br[l] * constants.ti[l]) / constants.tm[l] *
                (vm[fb] * vm[tb] * cos(va[fb] - va[tb])) +
                (-constants.g_br[l] * constants.tr[l] + constants.b_br[l] * constants.ti[l]) / constants.tm[l] *
                (vm[fb] * vm[tb] * sin(va[fb] - va[tb])))
        )
        p_to[l] = @constraint(model,
            p[t_idx] == sw_l * ((constants.g_br[l] + constants.g_to[l]) * vm[tb]^2 +
                (-constants.g_br[l] * constants.tr[l] - constants.b_br[l] * constants.ti[l]) / constants.tm[l] *
                (vm[tb] * vm[fb] * cos(va[tb] - va[fb])) +
                (-constants.b_br[l] * constants.tr[l] + constants.g_br[l] * constants.ti[l]) / constants.tm[l] *
                (vm[tb] * vm[fb] * sin(va[tb] - va[fb])))
        )
        q_to[l] = @constraint(model,
            q[t_idx] == sw_l * (-(constants.b_br[l] + constants.b_to[l]) * vm[tb]^2 -
                (-constants.b_br[l] * constants.tr[l] + constants.g_br[l] * constants.ti[l]) / constants.tm[l] *
                (vm[tb] * vm[fb] * cos(va[fb] - va[tb])) +
                (-constants.g_br[l] * constants.tr[l] - constants.b_br[l] * constants.ti[l]) / constants.tm[l] *
                (vm[tb] * vm[fb] * sin(va[tb] - va[fb])))
        )
        angle_diff_lb[l] = @constraint(model, sw_l * (va[fb] - va[tb]) >= sw_l * constants.angmin[l])
        angle_diff_ub[l] = @constraint(model, sw_l * (va[fb] - va[tb]) <= sw_l * constants.angmax[l])
        thermal_fr[l] = @constraint(model, p[f_idx]^2 + q[f_idx]^2 <= constants.fmax[l]^2)
        thermal_to[l] = @constraint(model, p[t_idx]^2 + q[t_idx]^2 <= constants.fmax[l]^2)
    end

    cons = ACJuMPConstraints(ref_bus, p_bal, q_bal, p_fr, q_fr, p_to, q_to,
        thermal_fr, thermal_to, angle_diff_lb, angle_diff_ub)
    return model, collect(va), collect(vm), collect(pg), collect(qg), collect(p), collect(q), cons
end

function _build_examodel(network::ACNetwork, data::ACOPFData, optimizer, silent::Bool)
    constants = data.constants
    n, m = network.n, network.m
    n_gen = length(constants.gen_bus)
    arc_fmax = [constants.fmax[arc[1]] for arc in data.arcs]

    core = ExaModels.ExaCore()
    va = ExaModels.variable(core, n)
    vm = ExaModels.variable(core, n; start=ones(n), lvar=constants.vmin, uvar=constants.vmax)
    pg = ExaModels.variable(core, n_gen; lvar=constants.pmin, uvar=constants.pmax)
    qg = ExaModels.variable(core, n_gen; lvar=constants.qmin, uvar=constants.qmax)
    p = ExaModels.variable(core, length(data.arcs); lvar=-arc_fmax, uvar=arc_fmax)
    q = ExaModels.variable(core, length(data.arcs); lvar=-arc_fmax, uvar=arc_fmax)

    ExaModels.objective(core,
        constants.cq[i] * pg[i]^2 + constants.cl[i] * pg[i] + constants.cc[i] for i in 1:n_gen)

    ref_bus = ExaModels.constraint(core, va[i] for i in data.ref_bus_keys)
    p_fr = ExaModels.constraint(core,
        p[data.arc_from_idx[l]] -
        network.sw[l] * ((constants.g_br[l] + constants.g_fr[l]) / constants.tm[l] * vm[constants.f_bus[l]]^2 +
        (-constants.g_br[l] * constants.tr[l] + constants.b_br[l] * constants.ti[l]) / constants.tm[l] *
        (vm[constants.f_bus[l]] * vm[constants.t_bus[l]] * cos(va[constants.f_bus[l]] - va[constants.t_bus[l]])) +
        (-constants.b_br[l] * constants.tr[l] - constants.g_br[l] * constants.ti[l]) / constants.tm[l] *
        (vm[constants.f_bus[l]] * vm[constants.t_bus[l]] * sin(va[constants.f_bus[l]] - va[constants.t_bus[l]])))
        for l in 1:m)
    q_fr = ExaModels.constraint(core,
        q[data.arc_from_idx[l]] -
        network.sw[l] * (-(constants.b_br[l] + constants.b_fr[l]) / constants.tm[l] * vm[constants.f_bus[l]]^2 -
        (-constants.b_br[l] * constants.tr[l] - constants.g_br[l] * constants.ti[l]) / constants.tm[l] *
        (vm[constants.f_bus[l]] * vm[constants.t_bus[l]] * cos(va[constants.f_bus[l]] - va[constants.t_bus[l]])) +
        (-constants.g_br[l] * constants.tr[l] + constants.b_br[l] * constants.ti[l]) / constants.tm[l] *
        (vm[constants.f_bus[l]] * vm[constants.t_bus[l]] * sin(va[constants.f_bus[l]] - va[constants.t_bus[l]])))
        for l in 1:m)
    p_to = ExaModels.constraint(core,
        p[data.arc_to_idx[l]] -
        network.sw[l] * ((constants.g_br[l] + constants.g_to[l]) * vm[constants.t_bus[l]]^2 +
        (-constants.g_br[l] * constants.tr[l] - constants.b_br[l] * constants.ti[l]) / constants.tm[l] *
        (vm[constants.t_bus[l]] * vm[constants.f_bus[l]] * cos(va[constants.t_bus[l]] - va[constants.f_bus[l]])) +
        (-constants.b_br[l] * constants.tr[l] + constants.g_br[l] * constants.ti[l]) / constants.tm[l] *
        (vm[constants.t_bus[l]] * vm[constants.f_bus[l]] * sin(va[constants.t_bus[l]] - va[constants.f_bus[l]])))
        for l in 1:m)
    q_to = ExaModels.constraint(core,
        q[data.arc_to_idx[l]] -
        network.sw[l] * (-(constants.b_br[l] + constants.b_to[l]) * vm[constants.t_bus[l]]^2 -
        (-constants.b_br[l] * constants.tr[l] + constants.g_br[l] * constants.ti[l]) / constants.tm[l] *
        (vm[constants.t_bus[l]] * vm[constants.f_bus[l]] * cos(va[constants.f_bus[l]] - va[constants.t_bus[l]])) +
        (-constants.g_br[l] * constants.tr[l] - constants.b_br[l] * constants.ti[l]) / constants.tm[l] *
        (vm[constants.t_bus[l]] * vm[constants.f_bus[l]] * sin(va[constants.t_bus[l]] - va[constants.f_bus[l]])))
        for l in 1:m)
    angle_diff_lb = ExaModels.constraint(core,
        network.sw[l] * constants.angmin[l] - network.sw[l] * va[constants.f_bus[l]] + network.sw[l] * va[constants.t_bus[l]]
        for l in 1:m; lcon=fill(-Inf, m), ucon=zeros(m))
    angle_diff_ub = ExaModels.constraint(core,
        network.sw[l] * va[constants.f_bus[l]] - network.sw[l] * va[constants.t_bus[l]] - network.sw[l] * constants.angmax[l]
        for l in 1:m; lcon=fill(-Inf, m), ucon=zeros(m))
    thermal_fr = ExaModels.constraint(core,
        p[data.arc_from_idx[l]]^2 + q[data.arc_from_idx[l]]^2 - constants.fmax[l]^2
        for l in 1:m; lcon=fill(-Inf, m), ucon=zeros(m))
    thermal_to = ExaModels.constraint(core,
        p[data.arc_to_idx[l]]^2 + q[data.arc_to_idx[l]]^2 - constants.fmax[l]^2
        for l in 1:m; lcon=fill(-Inf, m), ucon=zeros(m))

    p_bal = ExaModels.constraint(core,
        constants.pd[i] + constants.gs[i] * vm[i]^2 for i in 1:n)
    for i in 1:n
        for arc_idx in data.bus_arc_idxs[i]
            ExaModels.constraint!(core, p_bal, i => p[arc_idx])
        end
        for g in data.bus_gen_idxs[i]
            ExaModels.constraint!(core, p_bal, i => -pg[g])
        end
    end

    q_bal = ExaModels.constraint(core,
        constants.qd[i] - constants.bs[i] * vm[i]^2 for i in 1:n)
    for i in 1:n
        for arc_idx in data.bus_arc_idxs[i]
            ExaModels.constraint!(core, q_bal, i => q[arc_idx])
        end
        for g in data.bus_gen_idxs[i]
            ExaModels.constraint!(core, q_bal, i => -qg[g])
        end
    end

    model = ExaModels.ExaModel(core; prod=true)
    cons = ACExaConstraints(ref_bus, p_bal, q_bal, p_fr, q_fr, p_to, q_to,
        thermal_fr, thermal_to, angle_diff_lb, angle_diff_ub)
    return model, va, vm, pg, qg, p, q, cons
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

function ACOPFProblem(data::ParsedCase; kwargs...)
    network = ACNetwork(data)
    return ACOPFProblem(network; kwargs...)
end
