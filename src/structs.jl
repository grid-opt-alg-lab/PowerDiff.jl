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




#TODO
"""
"""
function VectorizedAdmittanceMatrix(Y::PowerModels.AdmittanceMatrix)
    nothing
end

abstract type AbstractSensitivity end
abstract type AbstractSensitivityPower <: AbstractSensitivity end
abstract type AbstractSensitivityToplogy <: AbstractSensitivity end
abstract type AbstractSensitivityTap <: AbstractSensitivity end

struct CurrentSensitivityPower <: AbstractSensitivityPower
    p::AbstractMatrix{Float64}
    q::AbstractMatrix{Float64}
end

struct VoltageSensitivityPower <: AbstractSensitivityPower
    p::AbstractMatrix{Float64}
    q::AbstractMatrix{Float64}
end

struct CurrentSensitivityTopology <: AbstractSensitivityToplogy
    p::AbstractMatrix{Float64}
    q::AbstractMatrix{Float64}
end
                                            
struct VoltageSensitivityTopology <: AbstractSensitivityToplogy
    p::AbstractMatrix{Float64}
    q::AbstractMatrix{Float64}
end


# =============================================================================
# DC OPF Types
# =============================================================================

"""
    DCNetwork

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
struct DCNetwork
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
    DCOPFSolution

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
struct DCOPFSolution
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

