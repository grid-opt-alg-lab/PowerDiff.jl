# =============================================================================
# AC Power Flow State Type
# =============================================================================

# Forward declaration - ACNetwork is defined in ac_network.jl
# This allows us to reference it in the struct before it's included

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

See `src/types/ac_network.jl` for constructor implementations.
"""
struct ACPowerFlowState <: AbstractPowerFlowState
    net::Any  # Union{ACNetwork, Nothing} - using Any to avoid forward ref issues
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
