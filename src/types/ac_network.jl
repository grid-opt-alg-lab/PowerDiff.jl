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
# ACNetwork: AC Network Data Structure
# =============================================================================
#
# Unified AC network representation with vectorized admittance for differentiation.
# Analogous to DCNetwork for DC OPF.

"""
    ACNetwork <: AbstractPowerNetwork

AC network data with vectorized admittance representation.

Provides a unified interface for AC power flow and sensitivity analysis,
analogous to `DCNetwork` for DC formulations. Uses edge-based conductance
and susceptance vectors for differentiable admittance matrix construction.

The admittance matrix is reconstructed as:
    Y = A' * Diag(g + j*b) * A + Diag(g_shunt + j*b_shunt)

For switching-aware formulation:
    Y(sw) = A' * Diag((g + j*b) .* sw) * A + Diag(g_shunt + j*b_shunt)

# Fields
- `n`: Number of buses
- `m`: Number of branches
- `A`: Branch-bus incidence matrix (m × n)
- `incidences`: Edge list [(i,j), ...] for each branch (sequential indices)
- `g`: Branch conductances
- `b`: Branch susceptances (note: typically negative for inductive lines)
- `g_shunt`: Shunt conductances per bus (from shunts + line charging)
- `b_shunt`: Shunt susceptances per bus
- `sw`: Branch switching states ∈ [0,1]^m
- `is_switchable`: Which branches can be switched
- `idx_slack`: Slack bus index (sequential)
- `vm_min`, `vm_max`: Voltage magnitude limits per bus
- `i_max`: Branch current magnitude limits
- `id_map`: Bidirectional mapping between original and sequential element IDs
- typed branch, bus, and generator arrays used by PF/OPF constructors
"""
struct ACNetwork <: AbstractPowerNetwork
    # Dimensions
    n::Int
    m::Int

    # Topology
    A::SparseMatrixCSC{Float64,Int}
    incidences::Vector{Tuple{Int,Int}}

    # Admittance (vectorized, edge-based)
    g::Vector{Float64}
    b::Vector{Float64}
    g_shunt::Vector{Float64}
    b_shunt::Vector{Float64}

    # Switching
    sw::Vector{Float64}
    is_switchable::BitVector

    # Reference
    idx_slack::Int

    # Limits
    vm_min::Vector{Float64}
    vm_max::Vector{Float64}
    i_max::Vector{Float64}

    # ID mapping
    id_map::IDMapping

    # Branch parameters
    f_bus::Vector{Int}
    t_bus::Vector{Int}
    br_r::Vector{Float64}
    br_x::Vector{Float64}
    br_b::Vector{Float64}
    g_fr::Vector{Float64}
    b_fr::Vector{Float64}
    g_to::Vector{Float64}
    b_to::Vector{Float64}
    tap::Vector{Float64}
    shift::Vector{Float64}
    tm::Vector{Float64}
    angmin::Vector{Float64}
    angmax::Vector{Float64}
    rate_a::Vector{Float64}

    # Bus injections and shunts
    pd::Vector{Float64}
    qd::Vector{Float64}
    gs::Vector{Float64}
    bs::Vector{Float64}

    # Generator data
    pg::Vector{Float64}
    qg::Vector{Float64}
    gen_bus::Vector{Int}
    pmin::Vector{Float64}
    pmax::Vector{Float64}
    qmin::Vector{Float64}
    qmax::Vector{Float64}
    cq::Vector{Float64}
    cl::Vector{Float64}
    cc::Vector{Float64}
    ref_bus_keys::Vector{Int}
end

# =============================================================================
# AC Power Flow State Type
# =============================================================================

"""
    ACPowerFlowState <: AbstractPowerFlowState

AC power flow solution with full injection tracking.

Provides a common interface for AC sensitivity computations, analogous to
`DCPowerFlowState` for DC power flow. Can be constructed from a PowerModels
network or from raw voltage/admittance data.

# Fields
- `net`: ACNetwork reference (optional, provides access to edge-level data)
- `v`: Complex voltage phasors at all buses
- `Y`: Bus admittance matrix
- `p`: Net real power injection (p = pg - pd)
- `q`: Net reactive power injection (q = qg - qd)
- `pg`: Real power generation per bus
- `pd`: Real power demand per bus
- `qg`: Reactive power generation per bus
- `qd`: Reactive power demand per bus
- `branch_data`: Branch dictionary with sequential indices (required for :im sensitivity)
- `idx_slack`: Index of the slack (reference) bus
- `n`: Number of buses
- `m`: Number of branches

# Constructors
- `ACPowerFlowState(v, Y; ...)`: From voltage phasors and admittance matrix
- `ACPowerFlowState(net::ACNetwork, v; ...)`: From ACNetwork and voltage solution
- `ACPowerFlowState(pm_net::Dict)`: From solved PowerModels network
"""
struct ACPowerFlowState <: AbstractPowerFlowState
    net::Union{ACNetwork, Nothing}
    v::Vector{ComplexF64}
    Y::SparseMatrixCSC{ComplexF64,Int}
    p::Vector{Float64}
    q::Vector{Float64}
    pg::Vector{Float64}
    pd::Vector{Float64}
    qg::Vector{Float64}
    qd::Vector{Float64}
    branch_data::Union{Dict{String,Any},Nothing}
    idx_slack::Int
    n::Int
    m::Int
end

"""
    ACPowerFlowState(v, Y; idx_slack=1, branch_data=nothing, pg=nothing, pd=nothing, qg=nothing, qd=nothing)

Construct from voltage phasors and admittance matrix.

Note: When constructing from just Y, the `net` field will be `nothing`.
For full network access, use the constructor that takes an ACNetwork.
"""
function ACPowerFlowState(
    v::AbstractVector{ComplexF64},
    Y::AbstractMatrix{ComplexF64};
    idx_slack::Int=1,
    branch_data::Union{Dict,Nothing}=nothing,
    pg::Union{Vector{Float64},Nothing}=nothing,
    pd::Union{Vector{Float64},Nothing}=nothing,
    qg::Union{Vector{Float64},Nothing}=nothing,
    qd::Union{Vector{Float64},Nothing}=nothing
)
    n = length(v)
    m = isnothing(branch_data) ? 0 : length(branch_data)
    Y_sparse = Y isa SparseMatrixCSC ? Y : sparse(Y)

    # Default to zeros if not provided
    pg_vec = isnothing(pg) ? zeros(n) : pg
    pd_vec = isnothing(pd) ? zeros(n) : pd
    qg_vec = isnothing(qg) ? zeros(n) : qg
    qd_vec = isnothing(qd) ? zeros(n) : qd

    p = pg_vec - pd_vec
    q = qg_vec - qd_vec

    ACPowerFlowState(nothing, Vector(v), Y_sparse, p, q, pg_vec, pd_vec, qg_vec, qd_vec,
                     branch_data, idx_slack, n, m)
end

# =============================================================================
# ACNetwork Constructors
# =============================================================================

"""
    ACNetwork(net::Dict; idx_slack=nothing)

Construct ACNetwork from a PowerModels network dictionary.

Accepts both basic and non-basic networks. Non-basic networks (with arbitrary
bus/branch/gen IDs) are automatically translated to sequential indices internally.
The original IDs are preserved in `id_map` for result interpretation.

# Arguments
- `net`: PowerModels network dictionary (basic or non-basic)
- `idx_slack`: Slack bus index (if not specified, uses reference bus from data)
"""
function ACNetwork(net::Dict{String,<:Any}; idx_slack::Union{Nothing,Int}=nothing)
    pm_data, id_map = _prepare_network_data(net)

    n_bus = length(id_map.bus_ids)
    n_branch = length(id_map.branch_ids)
    n_gen = length(id_map.gen_ids)
    branch_tbl = pm_data["branch"]
    bus_tbl = pm_data["bus"]
    gen_tbl = pm_data["gen"]

    # Build incidence matrix and edge list
    A = spzeros(n_branch, n_bus)
    incidences = Vector{Tuple{Int,Int}}(undef, n_branch)
    f_bus = Vector{Int}(undef, n_branch)
    t_bus = Vector{Int}(undef, n_branch)
    br_r = Vector{Float64}(undef, n_branch)
    br_x = Vector{Float64}(undef, n_branch)
    br_b = Vector{Float64}(undef, n_branch)
    g_fr = Vector{Float64}(undef, n_branch)
    b_fr = Vector{Float64}(undef, n_branch)
    g_to = Vector{Float64}(undef, n_branch)
    b_to = Vector{Float64}(undef, n_branch)
    tap = Vector{Float64}(undef, n_branch)
    shift = Vector{Float64}(undef, n_branch)
    tm = Vector{Float64}(undef, n_branch)
    angmin = Vector{Float64}(undef, n_branch)
    angmax = Vector{Float64}(undef, n_branch)
    rate_a = Vector{Float64}(undef, n_branch)

    for orig_id in id_map.branch_ids
        br = branch_tbl[string(orig_id)]
        ix = id_map.branch_to_idx[orig_id]
        f_idx = id_map.bus_to_idx[br["f_bus"]]
        t_idx = id_map.bus_to_idx[br["t_bus"]]
        A[ix, f_idx] = 1.0
        A[ix, t_idx] = -1.0
        incidences[ix] = (f_idx, t_idx)
        f_bus[ix] = f_idx
        t_bus[ix] = t_idx
        br_r[ix] = br["br_r"]
        br_x[ix] = br["br_x"]
        br_b[ix] = get(br, "br_b", 0.0)
        g_fr[ix] = get(br, "g_fr", 0.0)
        b_fr[ix] = get(br, "b_fr", br_b[ix] / 2.0)
        g_to[ix] = get(br, "g_to", 0.0)
        b_to[ix] = get(br, "b_to", br_b[ix] / 2.0)
        tap[ix] = get(br, "tap", 1.0)
        shift[ix] = get(br, "shift", 0.0)
        tm[ix] = tap[ix]^2
        angmin[ix] = get(br, "angmin", -π)
        angmax[ix] = get(br, "angmax", π)
        rate_a[ix] = get(br, "rate_a", Inf)
    end

    # Compute individual branch admittances from impedance
    g = zeros(n_branch)
    b = zeros(n_branch)

    for ix in 1:n_branch
        r = br_r[ix]
        x = br_x[ix]

        # Branch admittance: y = 1/(r + jx) = (r - jx)/(r² + x²)
        z2 = r^2 + x^2
        if z2 > 1e-10
            g[ix] = r / z2
            b[ix] = -x / z2
        else
            _SILENCE_WARNINGS[] || @warn "Branch $(id_map.branch_ids[ix]) has near-zero impedance (|z|² = $(z2)); treating as open (zero admittance)."
        end
    end

    # Shunt admittances: use PM.calc_admittance_matrix to get the full Y matrix,
    # then extract shunts from diagonal minus branch contributions
    am = PM.calc_admittance_matrix(pm_data)

    g_shunt = zeros(n_bus)
    b_shunt = zeros(n_bus)

    for i in 1:n_bus
        orig_bus_id = id_map.bus_ids[i]
        # Find PM's internal index for this bus
        pm_idx = am.bus_to_idx[orig_bus_id]

        # Y_ii = sum of all admittances connected to bus i
        y_sum = am.matrix[pm_idx, pm_idx]

        # Subtract branch contributions
        for br_idx in 1:n_branch
            if f_bus[br_idx] == i || t_bus[br_idx] == i
                y_sum -= g[br_idx] + im * b[br_idx]
            end
        end

        g_shunt[i] = real(y_sum)
        b_shunt[i] = imag(y_sum)
    end

    # All branches active by default
    sw = ones(n_branch)
    is_switchable = trues(n_branch)

    # Find slack bus
    if isnothing(idx_slack)
        ref_bus_keys = [i for i in 1:n_bus if get(bus_tbl[string(id_map.bus_ids[i])], "bus_type", 1) == 3]
        idx_slack = isempty(ref_bus_keys) ? 1 : ref_bus_keys[1]
    end

    # Voltage limits (iterate in sequential order)
    vm_min = [get(bus_tbl[string(id_map.bus_ids[i])], "vmin", 0.9) for i in 1:n_bus]
    vm_max = [get(bus_tbl[string(id_map.bus_ids[i])], "vmax", 1.1) for i in 1:n_bus]

    # Current limits (from rate_a if available)
    i_max = copy(rate_a)
    pd = zeros(n_bus)
    qd = zeros(n_bus)
    if haskey(pm_data, "load")
        for load_orig_id in id_map.load_ids
            load = pm_data["load"][string(load_orig_id)]
            bus_idx = id_map.bus_to_idx[load["load_bus"]]
            pd[bus_idx] += get(load, "pd", 0.0)
            qd[bus_idx] += get(load, "qd", 0.0)
        end
    else
        for i in 1:n_bus
            pd[i] = get(bus_tbl[string(id_map.bus_ids[i])], "pd", 0.0)
            qd[i] = get(bus_tbl[string(id_map.bus_ids[i])], "qd", 0.0)
        end
    end
    gs = zeros(n_bus)
    bs = zeros(n_bus)
    if haskey(pm_data, "shunt")
        for shunt_orig_id in id_map.shunt_ids
            shunt = pm_data["shunt"][string(shunt_orig_id)]
            bus_idx = id_map.bus_to_idx[shunt["shunt_bus"]]
            gs[bus_idx] += get(shunt, "gs", 0.0)
            bs[bus_idx] += get(shunt, "bs", 0.0)
        end
    else
        for i in 1:n_bus
            gs[i] = get(bus_tbl[string(id_map.bus_ids[i])], "gs", 0.0)
            bs[i] = get(bus_tbl[string(id_map.bus_ids[i])], "bs", 0.0)
        end
    end
    pg = zeros(n_bus)
    qg = zeros(n_bus)
    gen_bus = Vector{Int}(undef, n_gen)
    pmin = Vector{Float64}(undef, n_gen)
    pmax = Vector{Float64}(undef, n_gen)
    qmin = Vector{Float64}(undef, n_gen)
    qmax = Vector{Float64}(undef, n_gen)
    cq = Vector{Float64}(undef, n_gen)
    cl = Vector{Float64}(undef, n_gen)
    cc = Vector{Float64}(undef, n_gen)
    for i in 1:n_gen
        gen = gen_tbl[string(id_map.gen_ids[i])]
        bus_idx = id_map.bus_to_idx[gen["gen_bus"]]
        gen_bus[i] = bus_idx
        pg_val = get(gen, "pg", (gen["pmin"] + gen["pmax"]) / 2)
        qg_val = get(gen, "qg", 0.0)
        pg[bus_idx] += pg_val
        qg[bus_idx] += qg_val
        pmin[i] = gen["pmin"]
        pmax[i] = gen["pmax"]
        qmin[i] = gen["qmin"]
        qmax[i] = gen["qmax"]
        cq[i] = get(gen["cost"], 1, 0.0)
        cl[i] = get(gen["cost"], 2, 0.0)
        cc[i] = get(gen["cost"], 3, 0.0)
    end
    ref_bus_keys = [i for i in 1:n_bus if get(bus_tbl[string(id_map.bus_ids[i])], "bus_type", 1) == 3]

    return ACNetwork(
        n_bus, n_branch,
        sparse(A), incidences,
        g, b, g_shunt, b_shunt,
        sw, is_switchable,
        idx_slack,
        vm_min, vm_max, i_max,
        id_map,
        f_bus, t_bus, br_r, br_x, br_b, g_fr, b_fr, g_to, b_to, tap, shift, tm,
        angmin, angmax, rate_a,
        pd, qd, gs, bs,
        pg, qg, gen_bus, pmin, pmax, qmin, qmax, cq, cl, cc, ref_bus_keys
    )
end

function ACNetwork(data::ParsedCase; idx_slack::Union{Nothing,Int}=nothing)
    return ACNetwork(_parsedcase_to_pm_data(data); idx_slack=idx_slack)
end

"""
    ACNetwork(Y::AbstractMatrix{<:Complex}; idx_slack=1)

Construct ACNetwork from a complex admittance matrix.

Extracts edge-based representation from the full admittance matrix.
Useful for direct construction without PowerModels.
"""
function ACNetwork(Y::AbstractMatrix{<:Complex}; idx_slack::Int=1)
    n = size(Y, 1)

    # Build incidence matrix and extract off-diagonal admittances
    edges = Tuple{Int,Int}[]
    g = Float64[]
    b = Float64[]

    for i in 1:n
        for j in i+1:n
            if abs(Y[i,j]) > 1e-10
                push!(edges, (i, j))
                push!(g, -real(Y[i,j]))  # Off-diagonal is negative of branch admittance
                push!(b, -imag(Y[i,j]))
            end
        end
    end

    m = length(edges)

    # Build incidence matrix
    A = spzeros(m, n)
    for (e, (i, j)) in enumerate(edges)
        A[e, i] = 1.0
        A[e, j] = -1.0
    end

    # Shunt admittances (diagonal minus contributions from branches)
    g_shunt = real.(diag(Y))
    b_shunt = imag.(diag(Y))

    for (e, (i, j)) in enumerate(edges)
        g_shunt[i] -= g[e]
        g_shunt[j] -= g[e]
        b_shunt[i] -= b[e]
        b_shunt[j] -= b[e]
    end

    sw = ones(m)
    is_switchable = trues(m)
    vm_min = fill(0.9, n)
    vm_max = fill(1.1, n)
    i_max = fill(Inf, m)
    return ACNetwork(
        n, m,
        A, edges,
        g, b, g_shunt, b_shunt,
        sw, is_switchable,
        idx_slack,
        vm_min, vm_max, i_max,
        IDMapping(n, m, 0, 0),
        [edge[1] for edge in edges], [edge[2] for edge in edges],
        zeros(m), zeros(m), zeros(m), zeros(m), zeros(m), zeros(m), zeros(m),
        ones(m), zeros(m), ones(m), fill(-π, m), fill(π, m), fill(Inf, m),
        zeros(n), zeros(n), zeros(n), zeros(n),
        zeros(n), zeros(n), Int[], Float64[], Float64[], Float64[],
        Float64[], Float64[], Float64[], Float64[], [idx_slack]
    )
end

# =============================================================================
# Admittance Matrix Reconstruction
# =============================================================================

"""
    admittance_matrix(net::ACNetwork) → SparseMatrixCSC{ComplexF64}

Reconstruct the bus admittance matrix Y from vectorized representation.

    Y = A' * Diag(g + j*b) * A + Diag(g_shunt + j*b_shunt)
"""
function admittance_matrix(net::ACNetwork)
    W = Diagonal(net.g .+ im .* net.b)
    return transpose(net.A) * W * net.A + Diagonal(net.g_shunt .+ im .* net.b_shunt)
end

"""
    admittance_matrix(net::ACNetwork, sw::AbstractVector) → SparseMatrixCSC{ComplexF64}

Reconstruct admittance matrix with switching states.

    Y(sw) = A' * Diag((g + j*b) .* sw) * A + Diag(g_shunt + j*b_shunt)
"""
function admittance_matrix(net::ACNetwork, sw::AbstractVector)
    W = Diagonal((net.g .+ im .* net.b) .* sw)
    return transpose(net.A) * W * net.A + Diagonal(net.g_shunt .+ im .* net.b_shunt)
end

# =============================================================================
# Power Flow Equations as Functions on ACNetwork
# =============================================================================

"""
    p(net::ACNetwork, v::AbstractVector{<:Complex}) → Vector{Float64}

Real power injection at each bus: P = Re(diag(v̄) * Y * v)
"""
function p(net::ACNetwork, v::AbstractVector{<:Complex})
    Y = admittance_matrix(net)
    return real.(Diagonal(conj.(v)) * Y * v)
end

"""
    q(net::ACNetwork, v::AbstractVector{<:Complex}) → Vector{Float64}

Reactive power injection at each bus: Q = Im(diag(v̄) * Y * v)
"""
function q(net::ACNetwork, v::AbstractVector{<:Complex})
    Y = admittance_matrix(net)
    return imag.(Diagonal(conj.(v)) * Y * v)
end

"""
    p(net::ACNetwork, v_re::AbstractVector, v_im::AbstractVector) → Vector{Float64}

Real power injection from rectangular voltage coordinates.
"""
p(net::ACNetwork, v_re::AbstractVector, v_im::AbstractVector) =
    p(net, v_re .+ im .* v_im)

"""
    q(net::ACNetwork, v_re::AbstractVector, v_im::AbstractVector) → Vector{Float64}

Reactive power injection from rectangular voltage coordinates.
"""
q(net::ACNetwork, v_re::AbstractVector, v_im::AbstractVector) =
    q(net, v_re .+ im .* v_im)

"""
    p_polar(net::ACNetwork, vm::AbstractVector, δ::AbstractVector) → Vector{Float64}

Real power injection from polar voltage coordinates.
"""
p_polar(net::ACNetwork, vm::AbstractVector, δ::AbstractVector) =
    p(net, vm .* cis.(δ))

"""
    q_polar(net::ACNetwork, vm::AbstractVector, δ::AbstractVector) → Vector{Float64}

Reactive power injection from polar voltage coordinates.
"""
q_polar(net::ACNetwork, vm::AbstractVector, δ::AbstractVector) =
    q(net, vm .* cis.(δ))

"""
    branch_current(net::ACNetwork, v::AbstractVector{<:Complex}) → Vector{ComplexF64}

Complex branch currents: I_branch = Diag(y) * A * v
"""
function branch_current(net::ACNetwork, v::AbstractVector{<:Complex})
    W = Diagonal(net.g .+ im .* net.b)
    return W * net.A * v
end

"""
    branch_power(net::ACNetwork, v::AbstractVector{<:Complex}) → Vector{ComplexF64}

Complex branch power flows: S_branch = diag(A*v) * conj(I_branch)
"""
function branch_power(net::ACNetwork, v::AbstractVector{<:Complex})
    I = branch_current(net, v)
    ΔV = net.A * v  # Voltage difference across each branch
    return ΔV .* conj.(I)
end

# =============================================================================
# ACPowerFlowState Constructor from ACNetwork
# =============================================================================

"""
    ACPowerFlowState(net::ACNetwork, v::AbstractVector{<:Complex}; kwargs...)

Construct ACPowerFlowState from ACNetwork and voltage solution.

# Arguments
- `net`: ACNetwork containing topology and admittances
- `v`: Complex voltage phasors from power flow solution

# Keyword Arguments
- `pg`, `pd`, `qg`, `qd`: Generation and demand vectors (default to zeros)
"""
function ACPowerFlowState(
    net::ACNetwork,
    v::AbstractVector{<:Complex};
    pg::Union{Vector{Float64},Nothing}=nothing,
    pd::Union{Vector{Float64},Nothing}=nothing,
    qg::Union{Vector{Float64},Nothing}=nothing,
    qd::Union{Vector{Float64},Nothing}=nothing
)
    n = net.n
    m = net.m

    # Build admittance matrix from network
    Y = admittance_matrix(net)

    # Default to zeros if not provided
    pg_vec = isnothing(pg) ? zeros(n) : pg
    pd_vec = isnothing(pd) ? zeros(n) : pd
    qg_vec = isnothing(qg) ? zeros(n) : qg
    qd_vec = isnothing(qd) ? zeros(n) : qd

    p_net = pg_vec - pd_vec
    q_net = qg_vec - qd_vec

    return ACPowerFlowState(
        net, Vector{ComplexF64}(v), Y,
        p_net, q_net,
        pg_vec, pd_vec, qg_vec, qd_vec,
        nothing, net.idx_slack, n, m
    )
end

"""
    ACPowerFlowState(pm_net::Dict)

Construct ACPowerFlowState from a solved PowerModels network.

Accepts both basic and non-basic networks. Extracts voltage solution and
injection data. Creates an ACNetwork internally for access to edge-level data.
The network must have a solved power flow.
"""
function ACPowerFlowState(pm_net::Dict)
    # Create ACNetwork from the PowerModels data (handles basic/non-basic)
    net = ACNetwork(pm_net)

    n = net.n
    m = net.m

    # Extract bus voltages in sequential order
    v = Vector{ComplexF64}(undef, n)
    for i in 1:n
        bus = pm_net["bus"][string(net.id_map.bus_ids[i])]
        vm_val = get(bus, "vm", 1.0)
        va_val = get(bus, "va", 0.0)
        v[i] = vm_val * cis(va_val)
    end

    Y = admittance_matrix(net)
    pg = copy(net.pg)
    pd = copy(net.pd)
    qg = copy(net.qg)
    qd = copy(net.qd)
    p_net = pg - pd
    q_net = qg - qd
    seq_branch = _branch_data_dict(net)

    return ACPowerFlowState(
        net, v, Y,
        p_net, q_net,
        pg, pd, qg, qd,
        seq_branch, net.idx_slack, n, m
    )
end

function _branch_data_dict(net::ACNetwork)
    branch_data = Dict{String,Any}()
    for l in 1:net.m
        branch_data[string(l)] = Dict{String,Any}(
            "index" => l,
            "f_bus" => net.f_bus[l],
            "t_bus" => net.t_bus[l],
        )
    end
    return branch_data
end

"""
    calc_voltage_power_sensitivities(net::ACNetwork, v::AbstractVector{<:Complex}; full=true)

Compute voltage-power sensitivities from ACNetwork and voltage solution.
"""
function calc_voltage_power_sensitivities(
    net::ACNetwork,
    v::AbstractVector{<:Complex};
    full::Bool=true
)
    Y = admittance_matrix(net)
    return calc_voltage_power_sensitivities(Vector{ComplexF64}(v), Y;
        idx_slack=net.idx_slack, full=full)
end
