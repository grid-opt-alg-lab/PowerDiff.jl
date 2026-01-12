# =============================================================================
# Measurement Types (legacy, for state estimation)
# =============================================================================

abstract type AbstractMeasurement end

struct NodalMeasurement <: AbstractMeasurement
    var::Union{String,Int,Symbol}
    val::Float64
    σ::Float64
    node::Int
end

struct NetworkMeasurement <: AbstractMeasurement
    var::Union{String,Int,Symbol}
    val::Vector{Float64}
    σ::Union{Vector{Float64},Float64}
    nodes::Vector{Int}
end

struct PowerNetworkMeasurement <: AbstractMeasurement
    p::Union{Vector{NodalMeasurement},NetworkMeasurement}
    q::Union{Vector{NodalMeasurement},NetworkMeasurement}
    v::Union{Vector{NodalMeasurement},NetworkMeasurement}
    δ::Union{Vector{NodalMeasurement},NetworkMeasurement}
end

mutable struct ComplexNetworkState
    G::AbstractVector{Float64}
    B::AbstractVector{Float64}
    Vre::Vector{Float64}
    Vim::Vector{Float64}
end

# =============================================================================
# AC Power Flow State (unified representation for AC sensitivity analysis)
# =============================================================================

"""
    ACPowerFlowState <: AbstractPowerFlowState

AC power flow solution with full injection tracking.

Provides a common interface for AC sensitivity computations, analogous to
`DCPowerFlowState` for DC power flow. Can be constructed from a PowerModels
network or from raw voltage/admittance data.

# Fields
- `v`: Complex voltage phasors at all buses
- `Y`: Bus admittance matrix
- `p`: Net real power injection (p = pg - pd)
- `q`: Net reactive power injection (q = qg - qd)
- `pg`: Real power generation per bus
- `pd`: Real power demand per bus
- `qg`: Reactive power generation per bus
- `qd`: Reactive power demand per bus
- `branch_data`: PowerModels-style branch dictionary (optional)
- `idx_slack`: Index of the slack (reference) bus
- `n`: Number of buses
- `m`: Number of branches
"""
struct ACPowerFlowState <: AbstractPowerFlowState
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

    ACPowerFlowState(Vector(v), Y_sparse, p, q, pg_vec, pd_vec, qg_vec, qd_vec,
                     branch_data, idx_slack, n, m)
end

# NOTE: ACPowerFlowState(net::ACNetwork, ...) constructor is defined in types/ac_network.jl
# NOTE: ACPowerFlowState(net::Dict) constructor is defined in sens/voltage.jl

# =============================================================================
# AC Voltage Sensitivity Structs
# =============================================================================

"""
    VoltagePowerSensitivity <: AbstractSensitivityPower

Sensitivity of bus voltages with respect to power injections.

# Fields
- `∂v_∂p`: Complex phasor sensitivity ∂v/∂p (n × n)
- `∂v_∂q`: Complex phasor sensitivity ∂v/∂q (n × n)
- `∂vm_∂p`: Magnitude sensitivity ∂|v|/∂p (n × n)
- `∂vm_∂q`: Magnitude sensitivity ∂|v|/∂q (n × n)
"""
struct VoltagePowerSensitivity <: AbstractSensitivityPower
    ∂v_∂p::Matrix{ComplexF64}
    ∂v_∂q::Matrix{ComplexF64}
    ∂vm_∂p::Matrix{Float64}
    ∂vm_∂q::Matrix{Float64}
end

"""
    VoltageTopologySensitivity <: AbstractSensitivityTopology

Sensitivity of bus voltage magnitudes with respect to topology (admittance) parameters.

# Fields
- `∂vm_∂g`: Sensitivity ∂|v|/∂G (n × num_edges)
- `∂vm_∂b`: Sensitivity ∂|v|/∂B (n × num_edges)
"""
struct VoltageTopologySensitivity <: AbstractSensitivityTopology
    ∂vm_∂g::Matrix{Float64}
    ∂vm_∂b::Matrix{Float64}
end

# =============================================================================
# AC Current Sensitivity Structs
# =============================================================================

"""
    CurrentPowerSensitivity <: AbstractSensitivityPower

Sensitivity of branch currents with respect to power injections.

# Fields
- `∂I_∂p`: Complex current phasor sensitivity ∂I/∂p (m × n)
- `∂I_∂q`: Complex current phasor sensitivity ∂I/∂q (m × n)
- `∂Im_∂p`: Current magnitude sensitivity ∂|I|/∂p (m × n)
- `∂Im_∂q`: Current magnitude sensitivity ∂|I|/∂q (m × n)
"""
struct CurrentPowerSensitivity <: AbstractSensitivityPower
    ∂I_∂p::Matrix{ComplexF64}
    ∂I_∂q::Matrix{ComplexF64}
    ∂Im_∂p::Matrix{Float64}
    ∂Im_∂q::Matrix{Float64}
end

"""
    CurrentTopologySensitivity <: AbstractSensitivityTopology

Sensitivity of branch currents with respect to topology (admittance) parameters.

# Fields
- `∂Im_∂g`: Sensitivity ∂|I|/∂G (m × num_edges)
- `∂Im_∂b`: Sensitivity ∂|I|/∂B (m × num_edges)
"""
struct CurrentTopologySensitivity <: AbstractSensitivityTopology
    ∂Im_∂g::Matrix{Float64}
    ∂Im_∂b::Matrix{Float64}
end


# =============================================================================
# DC OPF Types
# =============================================================================

"""
    DCNetwork <: AbstractPowerNetwork

DC network data for B-θ OPF formulation. Uses susceptance-weighted Laplacian
`B = A' * Diagonal(-b .* z) * A` which preserves graphical structure for
topology sensitivity analysis and integration with RandomizedSwitching tools.

# Fields
- `n`, `m`, `k`: Number of buses, branches, and generators
- `A`: Branch-bus incidence matrix (m × n)
- `G_inc`: Generator-bus incidence matrix (n × k)
- `b`: Branch susceptances (imaginary part of 1/z)
- `z`: Branch switching states (1 = closed, 0 = open)
- `fmax`, `gmax`, `gmin`: Flow and generation limits
- `Δθ_max`, `Δθ_min`: Phase angle difference limits
- `cq`, `cl`: Quadratic and linear generation cost coefficients
- `ref_bus`: Reference bus index (phase angle = 0)
- `τ`: Regularization parameter for strong convexity
"""
struct DCNetwork <: AbstractPowerNetwork
    n::Int
    m::Int
    k::Int
    A::SparseMatrixCSC{Float64,Int}
    G_inc::SparseMatrixCSC{Float64,Int}
    b::Vector{Float64}
    z::Vector{Float64}
    fmax::Vector{Float64}
    gmax::Vector{Float64}
    gmin::Vector{Float64}
    Δθ_max::Vector{Float64}
    Δθ_min::Vector{Float64}
    cq::Vector{Float64}
    cl::Vector{Float64}
    ref_bus::Int
    τ::Float64
end

"""
    DCOPFSolution <: AbstractOPFSolution

Solution container for DC OPF problem, storing both primal and dual variables.

# Fields
- `θ`: Phase angles at each bus
- `g`: Generator outputs
- `f`: Line flows
- `ν_bal`: Power balance dual variables (nodal, used for LMP computation)
- `λ_ub`, `λ_lb`: Line flow upper/lower bound duals
- `ρ_ub`, `ρ_lb`: Generator upper/lower bound duals
- `objective`: Optimal objective value
"""
struct DCOPFSolution <: AbstractOPFSolution
    θ::Vector{Float64}
    g::Vector{Float64}
    f::Vector{Float64}
    ν_bal::Vector{Float64}
    λ_ub::Vector{Float64}
    λ_lb::Vector{Float64}
    ρ_ub::Vector{Float64}
    ρ_lb::Vector{Float64}
    objective::Float64
end

"""
    DCPowerFlowState <: AbstractPowerFlowState

DC power flow solution (phase angles from Laplacian solve, no optimization).
Supports both generation and demand for flexible sensitivity analysis.

Unlike DCOPFSolution, this represents a simple power flow solution θ = L⁺p
without optimal dispatch or constraint handling.

# Fields
- `net`: DCNetwork data
- `θ`: Phase angles (rad)
- `p`: Net injection vector (p = g - d)
- `g`: Generation vector
- `d`: Demand vector
- `f`: Branch flows (computed from θ)
"""
struct DCPowerFlowState <: AbstractPowerFlowState
    net::DCNetwork
    θ::Vector{Float64}
    p::Vector{Float64}
    g::Vector{Float64}
    d::Vector{Float64}
    f::Vector{Float64}
end

"""
    DemandSensitivity <: AbstractSensitivity

Sensitivity of DC OPF solution with respect to nodal demand.

# Fields
- `dθ_dd`: Jacobian ∂θ/∂d (n × n)
- `dg_dd`: Jacobian ∂g/∂d (k × n)
- `df_dd`: Jacobian ∂f/∂d (m × n)
- `dlmp_dd`: Jacobian ∂LMP/∂d (n × n)
"""
struct DemandSensitivity <: AbstractSensitivity
    dθ_dd::Matrix{Float64}
    dg_dd::Matrix{Float64}
    df_dd::Matrix{Float64}
    dlmp_dd::Matrix{Float64}
end

"""
    CostSensitivity <: AbstractSensitivity

Sensitivity of DC OPF solution with respect to cost coefficients.

# Fields
- `dg_dcq`: Jacobian ∂g/∂cq (k × k)
- `dg_dcl`: Jacobian ∂g/∂cl (k × k)
- `dlmp_dcq`: Jacobian ∂LMP/∂cq (n × k)
- `dlmp_dcl`: Jacobian ∂LMP/∂cl (n × k)
"""
struct CostSensitivity <: AbstractSensitivity
    dg_dcq::Matrix{Float64}
    dg_dcl::Matrix{Float64}
    dlmp_dcq::Matrix{Float64}
    dlmp_dcl::Matrix{Float64}
end

"""
    FlowLimitSensitivity <: AbstractSensitivity

Sensitivity of DC OPF solution with respect to line flow limits.

# Fields
- `dθ_dfmax`: Jacobian ∂θ/∂fmax (n × m)
- `dg_dfmax`: Jacobian ∂g/∂fmax (k × m)
- `df_dfmax`: Jacobian ∂f/∂fmax (m × m)
- `dlmp_dfmax`: Jacobian ∂LMP/∂fmax (n × m)
"""
struct FlowLimitSensitivity <: AbstractSensitivity
    dθ_dfmax::Matrix{Float64}
    dg_dfmax::Matrix{Float64}
    df_dfmax::Matrix{Float64}
    dlmp_dfmax::Matrix{Float64}
end

"""
    SusceptanceSensitivity <: AbstractSensitivity

Sensitivity of DC OPF solution with respect to branch susceptances.

# Fields
- `dθ_db`: Jacobian ∂θ/∂b (n × m)
- `dg_db`: Jacobian ∂g/∂b (k × m)
- `df_db`: Jacobian ∂f/∂b (m × m)
- `dlmp_db`: Jacobian ∂LMP/∂b (n × m)
"""
struct SusceptanceSensitivity <: AbstractSensitivity
    dθ_db::Matrix{Float64}
    dg_db::Matrix{Float64}
    df_db::Matrix{Float64}
    dlmp_db::Matrix{Float64}
end

"""
    SwitchingSensitivity <: AbstractSensitivityTopology

Sensitivity of power flow or OPF solution with respect to switching states.

Works for both DCPowerFlowState (simple Laplacian derivative) and DCOPFSolution
(via KKT implicit differentiation).

# Fields
- `dθ_dz`: Jacobian ∂θ/∂z (n × m)
- `dg_dz`: Jacobian ∂g/∂z (k × m), zeros for power flow
- `df_dz`: Jacobian ∂f/∂z (m × m)
- `dlmp_dz`: Jacobian ∂LMP/∂z (n × m), zeros for power flow
"""
struct SwitchingSensitivity <: AbstractSensitivityTopology
    dθ_dz::Matrix{Float64}
    dg_dz::Matrix{Float64}
    df_dz::Matrix{Float64}
    dlmp_dz::Matrix{Float64}
end

"""
    SwitchingSensitivity(dθ_dz, df_dz)

Convenience constructor for power flow (no generation or LMP sensitivity).
"""
function SwitchingSensitivity(dθ_dz::Matrix{Float64}, df_dz::Matrix{Float64})
    n, m = size(dθ_dz)
    dg_dz = zeros(0, m)  # No generators in power flow
    dlmp_dz = zeros(n, m)  # No LMP in power flow
    SwitchingSensitivity(dθ_dz, dg_dz, df_dz, dlmp_dz)
end

