# =============================================================================
# ACNetwork: AC Network Data Structure
# =============================================================================
#
# Unified AC network representation with vectorized admittance for differentiation.
# Analogous to DCNetwork for DC OPF, incorporating patterns from RandomizedSwitching.

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
- `branch_data`: Branch dictionary with sequential indices (optional, for legacy)
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
    # Preprocess
    pm_data = deepcopy(net)
    PM.standardize_cost_terms!(pm_data, order=2)
    PM.calc_thermal_limits!(pm_data)

    # Build ref structure
    ref = PM.build_ref(pm_data)[:it][:pm][:nw][0]
    id_map = IDMapping(ref)

    n_bus = length(id_map.bus_ids)
    n_branch = length(id_map.branch_ids)

    # Build incidence matrix and edge list
    A = spzeros(n_branch, n_bus)
    incidences = Vector{Tuple{Int,Int}}(undef, n_branch)

    for (orig_id, br) in ref[:branch]
        ix = id_map.branch_to_idx[orig_id]
        f_idx = id_map.bus_to_idx[br["f_bus"]]
        t_idx = id_map.bus_to_idx[br["t_bus"]]
        A[ix, f_idx] = 1.0
        A[ix, t_idx] = -1.0
        incidences[ix] = (f_idx, t_idx)
    end

    # Compute individual branch admittances from impedance
    g = zeros(n_branch)
    b = zeros(n_branch)

    for (orig_id, br) in ref[:branch]
        ix = id_map.branch_to_idx[orig_id]
        r = br["br_r"]
        x = br["br_x"]

        # Branch admittance: y = 1/(r + jx) = (r - jx)/(r² + x²)
        z2 = r^2 + x^2
        if z2 > 1e-10
            g[ix] = r / z2
            b[ix] = -x / z2
        end
    end

    # Shunt admittances: use PM.calc_admittance_matrix to get the full Y matrix,
    # then extract shunts from diagonal minus branch contributions
    Y_mat = PM.calc_admittance_matrix(pm_data).matrix
    # Y_mat is indexed by PM's internal bus ordering; use its idx_to_bus mapping
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
        for (orig_br_id, br) in ref[:branch]
            br_idx = id_map.branch_to_idx[orig_br_id]
            if br["f_bus"] == orig_bus_id || br["t_bus"] == orig_bus_id
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
        orig_ref = first(keys(ref[:ref_buses]))
        idx_slack = id_map.bus_to_idx[orig_ref]
    end

    # Voltage limits (iterate in sequential order)
    vm_min = [get(ref[:bus][id_map.bus_ids[i]], "vmin", 0.9) for i in 1:n_bus]
    vm_max = [get(ref[:bus][id_map.bus_ids[i]], "vmax", 1.1) for i in 1:n_bus]

    # Current limits (from rate_a if available)
    i_max = [get(ref[:branch][id_map.branch_ids[i]], "rate_a", Inf) for i in 1:n_branch]

    return ACNetwork(
        n_bus, n_branch,
        sparse(A), incidences,
        g, b, g_shunt, b_shunt,
        sw, is_switchable,
        idx_slack,
        vm_min, vm_max, i_max,
        id_map
    )
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
        IDMapping(n, m, 0, 0)
    )
end

"""
Find slack/reference bus from network data (returns original bus ID).
"""
function _find_slack_bus(net::Dict)
    for (key, bus) in net["bus"]
        if bus["bus_type"] == 3  # Reference bus type in MATPOWER
            return haskey(bus, "bus_i") ? bus["bus_i"] : parse(Int, key)
        end
    end
    return 1  # Default to bus 1
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
    id_map = net.id_map

    # Get voltage solution using build_ref + bus voltage data
    pm_data = deepcopy(pm_net)
    PM.standardize_cost_terms!(pm_data, order=2)
    PM.calc_thermal_limits!(pm_data)
    ref = PM.build_ref(pm_data)[:it][:pm][:nw][0]

    n = net.n
    m = net.m

    # Extract bus voltages in sequential order
    v = Vector{ComplexF64}(undef, n)
    for i in 1:n
        orig_id = id_map.bus_ids[i]
        bus = ref[:bus][orig_id]
        vm_val = get(bus, "vm", 1.0)
        va_val = get(bus, "va", 0.0)
        v[i] = vm_val * cis(va_val)
    end

    Y = admittance_matrix(net)

    # Extract generation and demand in sequential bus order
    pg = zeros(n)
    qg = zeros(n)
    for (orig_id, gen_ids) in ref[:bus_gens]
        bus_idx = id_map.bus_to_idx[orig_id]
        for gen_id in gen_ids
            gen = ref[:gen][gen_id]
            pg[bus_idx] += get(gen, "pg", 0.0)
            qg[bus_idx] += get(gen, "qg", 0.0)
        end
    end

    pd = zeros(n)
    qd = zeros(n)
    for (orig_id, load_ids) in ref[:bus_loads]
        bus_idx = id_map.bus_to_idx[orig_id]
        for load_id in load_ids
            load = ref[:load][load_id]
            pd[bus_idx] += get(load, "pd", 0.0)
            qd[bus_idx] += get(load, "qd", 0.0)
        end
    end

    p_net = pg - pd
    q_net = qg - qd

    # Build branch_data with sequential indices for current sensitivity
    seq_branch = Dict{String,Any}()
    for (orig_id, br) in ref[:branch]
        seq_idx = id_map.branch_to_idx[orig_id]
        seq_br = copy(br)
        seq_br["index"] = seq_idx
        seq_br["f_bus"] = id_map.bus_to_idx[br["f_bus"]]
        seq_br["t_bus"] = id_map.bus_to_idx[br["t_bus"]]
        seq_branch[string(seq_idx)] = seq_br
    end

    return ACPowerFlowState(
        net, v, Y,
        p_net, q_net,
        pg, pd, qg, qd,
        seq_branch, net.idx_slack, n, m
    )
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
