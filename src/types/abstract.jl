# =============================================================================
# Abstract Type Hierarchy for PowerModelsDiff
# =============================================================================
#
# Provides a unified type structure for multiple dispatch across:
# - DC power flow and DC OPF
# - AC power flow (and future AC OPF)
# - Sensitivity analysis with parameter-based dispatch

# =============================================================================
# Level 1: Abstract Network (static data container)
# =============================================================================

"""
    AbstractPowerNetwork

Abstract base type for power network data containers.

Concrete subtypes:
- `DCNetwork`: DC network with susceptance-weighted Laplacian
- `ACNetwork`: AC network with complex admittance matrix
"""
abstract type AbstractPowerNetwork end

# =============================================================================
# Level 2: Abstract State (solved power flow or OPF solution)
# =============================================================================

"""
    AbstractPowerFlowState

Abstract base type for power flow solutions (DC or AC).

Concrete subtypes:
- `DCPowerFlowState`: DC power flow solution (θ = L⁺p)
- `ACPowerFlowState`: AC power flow solution (complex voltages)
- `AbstractOPFSolution` and its subtypes
"""
abstract type AbstractPowerFlowState end

"""
    AbstractOPFSolution <: AbstractPowerFlowState

Abstract type for optimal power flow solutions with dual variables.

Concrete subtypes:
- `DCOPFSolution`: DC OPF with generation dispatch and duals
- `ACOPFSolution`: (Future) AC OPF with duals
"""
abstract type AbstractOPFSolution <: AbstractPowerFlowState end

# =============================================================================
# Level 3: Abstract Problem (optimization wrapper)
# =============================================================================

"""
    AbstractOPFProblem

Abstract base type for OPF problem wrappers (JuMP models).

Concrete subtypes:
- `DCOPFProblem`: DC OPF problem (B-θ formulation)
- `ACOPFProblem`: (Future) AC OPF problem
"""
abstract type AbstractOPFProblem end

# =============================================================================
# Level 4: Abstract Sensitivity Types
# =============================================================================

"""
    AbstractSensitivity

Abstract base type for sensitivity analysis results.
"""
abstract type AbstractSensitivity end

"""
    AbstractSensitivityPower <: AbstractSensitivity

Sensitivity with respect to power injections (P, Q).
"""
abstract type AbstractSensitivityPower <: AbstractSensitivity end

"""
    AbstractSensitivityTopology <: AbstractSensitivity

Sensitivity with respect to network topology/parameters (z, b, g).
"""
abstract type AbstractSensitivityTopology <: AbstractSensitivity end
