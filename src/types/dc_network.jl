# =============================================================================
# DCNetwork: DC Network Data Structure
# =============================================================================
#
# DC network representation for B-theta OPF formulation with susceptance-weighted
# Laplacian L = A' * Diag(-b .* z) * A.

"""
    DCNetwork <: AbstractPowerNetwork

DC network data for B-theta OPF formulation. Uses susceptance-weighted Laplacian
`B = A' * Diagonal(-b .* z) * A` which preserves graphical structure for
topology sensitivity analysis and integration with RandomizedSwitching tools.

# Fields
- `n`, `m`, `k`: Number of buses, branches, and generators
- `A`: Branch-bus incidence matrix (m x n)
- `G_inc`: Generator-bus incidence matrix (n x k)
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

# =============================================================================
# DC Power Flow and OPF State Types
# =============================================================================

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

Unlike DCOPFSolution, this represents a simple power flow solution theta = L^+ p
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
