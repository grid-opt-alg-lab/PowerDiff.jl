# =============================================================================
# Sensitivity Parameter Types for Multiple Dispatch
# =============================================================================
#
# These singleton types enable unified `calc_sensitivity(state, parameter)` API
# with type-based dispatch to the appropriate sensitivity computation.

"""
    AbstractSensitivityParameter

Abstract base type for sensitivity parameters.

Used for dispatch in `calc_sensitivity(state, parameter)`.
"""
abstract type AbstractSensitivityParameter end

# =============================================================================
# DC Power Flow / OPF Parameters
# =============================================================================

"""
    DemandParameter <: AbstractSensitivityParameter

Sensitivity with respect to demand vector d.
"""
struct DemandParameter <: AbstractSensitivityParameter end

"""
    GenerationParameter <: AbstractSensitivityParameter

Sensitivity with respect to generation vector g.
"""
struct GenerationParameter <: AbstractSensitivityParameter end

"""
    CostParameter <: AbstractSensitivityParameter

Sensitivity with respect to cost coefficients (linear and quadratic).
"""
struct CostParameter <: AbstractSensitivityParameter end

"""
    FlowLimitParameter <: AbstractSensitivityParameter

Sensitivity with respect to branch flow limits.
"""
struct FlowLimitParameter <: AbstractSensitivityParameter end

"""
    SusceptanceParameter <: AbstractSensitivityParameter

Sensitivity with respect to branch susceptances.
"""
struct SusceptanceParameter <: AbstractSensitivityParameter end

"""
    SwitchingParameter <: AbstractSensitivityParameter

Sensitivity with respect to switching states z ∈ [0,1]^m.
"""
struct SwitchingParameter <: AbstractSensitivityParameter end

# =============================================================================
# AC Power Flow Parameters
# =============================================================================

"""
    PowerInjectionParameter <: AbstractSensitivityParameter

Sensitivity with respect to complex power injections (P + jQ).
"""
struct PowerInjectionParameter <: AbstractSensitivityParameter end

"""
    TopologyParameter <: AbstractSensitivityParameter

Sensitivity with respect to admittance parameters (G, B).
"""
struct TopologyParameter <: AbstractSensitivityParameter end

# =============================================================================
# Singleton Instances for Convenience
# =============================================================================

"""Singleton instance for demand sensitivity."""
const DEMAND = DemandParameter()

"""Singleton instance for generation sensitivity."""
const GENERATION = GenerationParameter()

"""Singleton instance for cost sensitivity."""
const COST = CostParameter()

"""Singleton instance for flow limit sensitivity."""
const FLOWLIMIT = FlowLimitParameter()

"""Singleton instance for susceptance sensitivity."""
const SUSCEPTANCE = SusceptanceParameter()

"""Singleton instance for switching sensitivity."""
const SWITCHING = SwitchingParameter()

"""Singleton instance for power injection sensitivity."""
const POWER = PowerInjectionParameter()

"""Singleton instance for topology sensitivity."""
const TOPOLOGY = TopologyParameter()
