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
analogous to `DCNetwork` for DC formulations. Each branch contributes its
from/from, from/to, to/from, and to/to pi-model coefficients, including line
charging, transformer taps, phase shifts, switching, and parallel lines.

# Fields
- `n`: Number of buses
- `m`: Number of branches
- `A`: Branch-bus incidence matrix (m × n)
- `incidences`: Edge list [(i,j), ...] for each branch (sequential indices)
- `g`: Branch conductances
- `b`: Branch susceptances (note: typically negative for inductive lines)
- `g_shunt`: Shunt conductances per bus
- `b_shunt`: Shunt susceptances per bus
- `sw`: Branch switching states ∈ [0,1]^m
- `is_switchable`: Which branches can be switched
- `idx_slack`: Slack bus index (sequential)
- `vm_min`, `vm_max`: Voltage magnitude limits per bus
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
`DCPowerFlowState` for DC power flow. Can be constructed from an `ACNetwork`
and externally solved voltages, or from raw voltage/admittance data.

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
- `branch_data`: Optional branch dictionary for raw voltage/admittance states
- `idx_slack`: Index of the slack (reference) bus
- `n`: Number of buses
- `m`: Number of branches

# Constructors
- `ACPowerFlowState(v, Y; ...)`: From voltage phasors and admittance matrix
- `ACPowerFlowState(net::ACNetwork, v; ...)`: From ACNetwork and voltage solution
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

Reject the removed dictionary API with a migration hint.
"""
function ACNetwork(net::Dict{String,<:Any}; idx_slack::Union{Nothing,Int}=nothing)
    throw(ArgumentError("dictionary constructors were removed; parse a network file with PowerDiff.parse_file"))
end

ACNetwork(net::PowerIO.Network; idx_slack::Union{Nothing,Int}=nothing) =
    ACNetwork(_network_data(net); idx_slack=idx_slack)

# Build from PowerDiff network tables (see `_network_data`). The `PowerIO.Network`
# method runs PowerDiff's modeling deltas; this assumes the tables are already
# normalized, so programmatic callers can supply ready values directly.
function ACNetwork(data::NamedTuple; idx_slack::Union{Nothing,Int}=nothing)
    id_map = IDMapping(data)
    n_bus = length(id_map.bus_ids)
    n_branch = length(id_map.branch_ids)
    n_gen = length(id_map.gen_ids)
    bus_tbl = Dict(bus.bus_i => bus for bus in data.bus)
    branch_tbl = Dict(branch.index => branch for branch in data.branch)
    gen_tbl = Dict(gen.index => gen for gen in data.gen)

    A = spzeros(n_branch, n_bus)
    incidences = Vector{Tuple{Int,Int}}(undef, n_branch)
    f_bus = Vector{Int}(undef, n_branch)
    t_bus = Vector{Int}(undef, n_branch)
    br_r = Vector{Float64}(undef, n_branch)
    br_x = Vector{Float64}(undef, n_branch)
    br_b = Vector{Float64}(undef, n_branch)
    g_fr = zeros(n_branch)
    b_fr = Vector{Float64}(undef, n_branch)
    g_to = zeros(n_branch)
    b_to = Vector{Float64}(undef, n_branch)
    tap = Vector{Float64}(undef, n_branch)
    shift = Vector{Float64}(undef, n_branch)
    tm = Vector{Float64}(undef, n_branch)
    angmin = Vector{Float64}(undef, n_branch)
    angmax = Vector{Float64}(undef, n_branch)
    rate_a = Vector{Float64}(undef, n_branch)
    g = zeros(n_branch)
    b = zeros(n_branch)

    for orig_id in id_map.branch_ids
        branch = branch_tbl[orig_id]
        l = id_map.branch_to_idx[orig_id]
        fb = id_map.bus_to_idx[branch.f_bus]
        tb = id_map.bus_to_idx[branch.t_bus]
        A[l, fb] = 1.0
        A[l, tb] = -1.0
        incidences[l] = (fb, tb)
        f_bus[l] = fb
        t_bus[l] = tb
        br_r[l] = branch.br_r
        br_x[l] = branch.br_x
        br_b[l] = branch.br_b
        # MATPOWER models line charging as a single symmetric susceptance with no
        # charging conductance, and `_network_data` already folded the two PowerIO
        # sides into `br_b`. Split it evenly and leave g_fr/g_to at zero (initialized
        # above). Asymmetric b_fr != b_to or nonzero g_fr/g_to from a non MATPOWER
        # source would be averaged/dropped here; thread the per-side values through
        # if that fidelity is ever needed.
        b_fr[l] = branch.br_b / 2
        b_to[l] = branch.br_b / 2
        tap[l] = iszero(branch.tap) ? 1.0 : branch.tap
        shift[l] = branch.shift
        tm[l] = tap[l]^2
        angmin[l] = branch.angmin
        angmax[l] = branch.angmax
        rate_a[l] = branch.rate_a
        z2 = branch.br_r^2 + branch.br_x^2
        if z2 > 1e-10
            g[l] = branch.br_r / z2
            b[l] = -branch.br_x / z2
        else
            _SILENCE_WARNINGS[] || @warn "Branch $orig_id has near-zero impedance; treating as open."
        end
    end

    g_shunt = zeros(n_bus)
    b_shunt = zeros(n_bus)
    pd = zeros(n_bus)
    qd = zeros(n_bus)
    gs = zeros(n_bus)
    bs = zeros(n_bus)
    # to_powerdata aggregates loads/shunts into per-bus values (per-unit).
    for bus in data.bus
        i = id_map.bus_to_idx[bus.bus_i]
        pd[i] += bus.pd
        qd[i] += bus.qd
        gs[i] += bus.gs
        bs[i] += bus.bs
        g_shunt[i] += bus.gs
        b_shunt[i] += bus.bs
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
    for orig_id in id_map.gen_ids
        gen = gen_tbl[orig_id]
        j = id_map.gen_to_idx[orig_id]
        i = id_map.bus_to_idx[gen.gen_bus]
        gen_bus[j] = i
        pg[i] += gen.pg
        qg[i] += gen.qg
        pmin[j] = gen.pmin
        pmax[j] = gen.pmax
        qmin[j] = gen.qmin
        qmax[j] = gen.qmax
        cq[j], cl[j], cc[j] = gen.cost
    end

    if !isnothing(idx_slack)
        1 <= idx_slack <= n_bus || throw(ArgumentError(
            "idx_slack=$idx_slack is not a valid bus index (1:$n_bus)"))
        ref_bus_keys = [idx_slack]
    else
        ref_bus_keys = [id_map.bus_to_idx[id] for id in id_map.bus_ids if bus_tbl[id].bus_type == 3]
        if isempty(ref_bus_keys)
            _SILENCE_WARNINGS[] || @warn "No reference bus (type 3) in the network; defaulting to bus 1 as slack. Pass `idx_slack` to choose explicitly."
            push!(ref_bus_keys, 1)
        end
        idx_slack = first(ref_bus_keys)
    end
    vm_min = [bus_tbl[id].vmin for id in id_map.bus_ids]
    vm_max = [bus_tbl[id].vmax for id in id_map.bus_ids]

    return ACNetwork(
        n_bus, n_branch, sparse(A), incidences, g, b, g_shunt, b_shunt,
        ones(n_branch), trues(n_branch), idx_slack, vm_min, vm_max,
        id_map, f_bus, t_bus, br_r, br_x, br_b, g_fr, b_fr, g_to, b_to,
        tap, shift, tm, angmin, angmax, rate_a, pd, qd, gs, bs, pg, qg,
        gen_bus, pmin, pmax, qmin, qmax, cq, cl, cc, ref_bus_keys
    )
end

"""
    ACNetwork(Y::AbstractMatrix{<:Complex}; idx_slack=1)

Construct ACNetwork from a complex admittance matrix.

Extracts edge-based representation from the full admittance matrix.
Useful for direct construction from a raw admittance matrix.
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
    return ACNetwork(
        n, m,
        A, edges,
        g, b, g_shunt, b_shunt,
        sw, is_switchable,
        idx_slack,
        vm_min, vm_max,
        IDMapping(n, m, 0),
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
admittance_matrix(net::ACNetwork) = admittance_matrix(net, net.sw)

"""
    admittance_matrix(net::ACNetwork, sw::AbstractVector) → SparseMatrixCSC{ComplexF64}

Reconstruct admittance matrix with switching states.

    Y(sw) = A' * Diag((g + j*b) .* sw) * A + Diag(g_shunt + j*b_shunt)
"""
function admittance_matrix(net::ACNetwork, sw::AbstractVector)
    length(sw) == net.m || throw(DimensionMismatch("switching vector must have length $(net.m)"))
    rows = collect(1:net.n)
    cols = collect(1:net.n)
    vals = ComplexF64.(net.g_shunt .+ im .* net.b_shunt)
    sizehint!(rows, net.n + 4net.m)
    sizehint!(cols, net.n + 4net.m)
    sizehint!(vals, net.n + 4net.m)
    for l in 1:net.m
        yff, yft, ytf, ytt = _branch_admittance_coefficients(net, l)
        fb, tb = net.f_bus[l], net.t_bus[l]
        append!(rows, (fb, fb, tb, tb))
        append!(cols, (fb, tb, fb, tb))
        append!(vals, sw[l] .* (yff, yft, ytf, ytt))
    end
    return sparse(rows, cols, vals, net.n, net.n)
end

@inline function _branch_admittance_coefficients(net::ACNetwork, l::Int)
    y = net.g[l] + im * net.b[l]
    tap = net.tap[l] * cis(net.shift[l])
    yff = (y + net.g_fr[l] + im * net.b_fr[l]) / abs2(tap)
    yft = -y / conj(tap)
    ytf = -y / tap
    ytt = y + net.g_to[l] + im * net.b_to[l]
    return yff, yft, ytf, ytt
end

@inline function _branch_current_coefficients(net::ACNetwork, l::Int)
    yff, yft, _, _ = _branch_admittance_coefficients(net, l)
    return net.sw[l] * yff, net.sw[l] * yft
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

Complex branch currents injected from the from-side bus.
"""
function branch_current(net::ACNetwork, v::AbstractVector{<:Complex})
    return [
        let (yff, yft) = _branch_current_coefficients(net, l)
            yff * v[net.f_bus[l]] + yft * v[net.t_bus[l]]
        end
        for l in 1:net.m
    ]
end

"""
    branch_power(net::ACNetwork, v::AbstractVector{<:Complex}) → Vector{ComplexF64}

Complex branch power flows injected from the from-side bus.
"""
function branch_power(net::ACNetwork, v::AbstractVector{<:Complex})
    I = branch_current(net, v)
    return v[net.f_bus] .* conj.(I)
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
- `pg`, `pd`, `qg`, `qd`: Generation and demand vectors (default to network values)
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

    pg_vec = isnothing(pg) ? copy(net.pg) : pg
    pd_vec = isnothing(pd) ? copy(net.pd) : pd
    qg_vec = isnothing(qg) ? copy(net.qg) : qg
    qd_vec = isnothing(qd) ? copy(net.qd) : qd

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

Reject the removed dictionary API with a migration hint.
"""
function ACPowerFlowState(pm_net::Dict)
    throw(ArgumentError("dictionary constructors were removed; construct ACPowerFlowState(ACNetwork(data), v)"))
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
