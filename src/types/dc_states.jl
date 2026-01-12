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
