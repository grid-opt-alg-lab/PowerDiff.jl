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
- `incidences`: Edge list [(i,j), ...] for each branch
- `g`: Branch conductances
- `b`: Branch susceptances (note: typically negative for inductive lines)
- `g_shunt`: Shunt conductances per bus (from shunts + line charging)
- `b_shunt`: Shunt susceptances per bus
- `sw`: Branch switching states ∈ [0,1]^m
- `is_switchable`: Which branches can be switched
- `idx_slack`: Slack bus index
- `vm_min`, `vm_max`: Voltage magnitude limits per bus
- `i_max`: Branch current magnitude limits
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
- `branch_data`: PowerModels-style branch dictionary (optional, for legacy)
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

Construct ACNetwork from a PowerModels basic network dictionary.

Extracts edge-based admittances from branch data and constructs the
incidence matrix. Shunt admittances are computed from the full admittance
matrix diagonal.

# Arguments
- `net`: PowerModels network dictionary (must be basic_network)
- `idx_slack`: Slack bus index (if not specified, uses reference bus from data)
"""
function ACNetwork(net::Dict{String,<:Any}; idx_slack::Union{Nothing,Int}=nothing)
    @assert haskey(net, "basic_network") && net["basic_network"] "Network must be a basic network"

    n_bus = length(net["bus"])
    n_branch = length(net["branch"])

    # Build incidence matrix and edge list
    A = Float64.(PM.calc_basic_incidence_matrix(net))
    incidences = Vector{Tuple{Int,Int}}(undef, n_branch)

    for (_, br) in net["branch"]
        ix = br["index"]
        incidences[ix] = (br["f_bus"], br["t_bus"])
    end

    # Compute individual branch admittances from impedance
    g = zeros(n_branch)
    b = zeros(n_branch)

    for (_, br) in net["branch"]
        ix = br["index"]
        r = br["br_r"]
        x = br["br_x"]

        # Branch admittance: y = 1/(r + jx) = (r - jx)/(r² + x²)
        z2 = r^2 + x^2
        if z2 > 1e-10
            g[ix] = r / z2
            b[ix] = -x / z2
        end
    end

    # Shunt admittances (diagonal of Y matrix minus off-diagonal contributions)
    # This includes shunt elements and line charging
    Y_mat = PM.calc_basic_admittance_matrix(net)
    g_shunt = zeros(n_bus)
    b_shunt = zeros(n_bus)

    for i in 1:n_bus
        # Y_ii = sum of all admittances connected to bus i
        # Shunt = Y_ii - sum of off-diagonal (branch) admittances
        y_sum = Y_mat[i, i]

        # Subtract branch contributions
        for (_, br) in net["branch"]
            ix = br["index"]
            if br["f_bus"] == i || br["t_bus"] == i
                y_sum -= g[ix] + im * b[ix]
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
        idx_slack = _find_slack_bus(net)
    end

    # Voltage limits
    vm_min = [get(net["bus"][string(i)], "vmin", 0.9) for i in 1:n_bus]
    vm_max = [get(net["bus"][string(i)], "vmax", 1.1) for i in 1:n_bus]

    # Current limits (from rate_a if available)
    i_max = [get(net["branch"][string(i)], "rate_a", Inf) for i in 1:n_branch]

    return ACNetwork(
        n_bus, n_branch,
        sparse(A), incidences,
        g, b, g_shunt, b_shunt,
        sw, is_switchable,
        idx_slack,
        vm_min, vm_max, i_max
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
        vm_min, vm_max, i_max
    )
end

"""
Find slack/reference bus from network data.
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

Extracts voltage solution and injection data from the network dictionary.
Creates an ACNetwork internally for access to edge-level data.
The network must have a solved power flow.
"""
function ACPowerFlowState(pm_net::Dict)
    @assert haskey(pm_net, "basic_network") && pm_net["basic_network"] "Network must be a basic network"

    # Create ACNetwork from the PowerModels data
    net = ACNetwork(pm_net)

    # Get voltage solution
    v = PM.calc_basic_bus_voltage(pm_net)
    Y = admittance_matrix(net)

    n = net.n
    m = net.m

    # Extract generation and demand
    pg = zeros(n)
    qg = zeros(n)
    for (_, gen) in pm_net["gen"]
        bus_idx = gen["gen_bus"]
        pg[bus_idx] += get(gen, "pg", 0.0)
        qg[bus_idx] += get(gen, "qg", 0.0)
    end

    pd = zeros(n)
    qd = zeros(n)
    for (_, load) in pm_net["load"]
        bus_idx = load["load_bus"]
        pd[bus_idx] += get(load, "pd", 0.0)
        qd[bus_idx] += get(load, "qd", 0.0)
    end

    p_net = pg - pd
    q_net = qg - qd

    return ACPowerFlowState(
        net, v, Y,
        p_net, q_net,
        pg, pd, qg, qd,
        pm_net["branch"], net.idx_slack, n, m
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
