# =============================================================================
# Abstract Type Hierarchy for PowerModelsDiff
# =============================================================================
#
# Provides a unified type structure for multiple dispatch across:
# - DC power flow and DC OPF
# - AC power flow and AC OPF
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
- `DCPowerFlowState`: DC power flow solution (θ_r = L_r \\ p_r)
- `ACPowerFlowState`: AC power flow solution (complex voltages)
- `AbstractOPFSolution` and its subtypes
"""
abstract type AbstractPowerFlowState end

"""
    AbstractOPFSolution <: AbstractPowerFlowState

Abstract type for optimal power flow solutions with dual variables.

Concrete subtypes:
- `DCOPFSolution`: DC OPF with generation dispatch and duals
- `ACOPFSolution`: AC OPF with voltages, generation, and duals
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
- `ACOPFProblem`: AC OPF problem (polar formulation)
"""
abstract type AbstractOPFProblem end
