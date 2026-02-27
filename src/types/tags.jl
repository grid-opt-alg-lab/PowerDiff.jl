# =============================================================================
# Singleton Type Tags for Sensitivity Dispatch
# =============================================================================
#
# Following PowerModels.jl conventions, these singleton types enable type-based
# dispatch for sensitivity results. The Sensitivity{F,O,P} type uses these as
# type parameters for formulation, operand, and parameter.
#
# Usage:
#   sens = calc_sensitivity(prob, LMP(), Demand())  # Returns Sensitivity{DCOPF, LMP, Demand}
#   process(s::Sensitivity{DCOPF, LMP, Demand}) = ...  # Dispatch on specific type

# =============================================================================
# Formulation Tags (DC/AC, PowerFlow/OPF)
# =============================================================================

"""
    DCOPF <: AbstractFormulation

Tag for DC Optimal Power Flow (B-theta formulation).
"""
struct DCOPF <: AbstractFormulation end

"""
    ACOPF <: AbstractFormulation

Tag for AC Optimal Power Flow (polar formulation).
"""
struct ACOPF <: AbstractFormulation end

"""
    DCPF <: AbstractFormulation

Tag for DC Power Flow (linear B-theta).
"""
struct DCPF <: AbstractFormulation end

"""
    ACPF <: AbstractFormulation

Tag for AC Power Flow.
"""
struct ACPF <: AbstractFormulation end

# =============================================================================
# Operand Tags (what we differentiate)
# =============================================================================

"""
    VoltageAngle <: AbstractOperand

Voltage angle operand (radians).
"""
struct VoltageAngle <: AbstractOperand end

"""
    VoltageMagnitude <: AbstractOperand

Voltage magnitude operand (per unit).
"""
struct VoltageMagnitude <: AbstractOperand end

"""
    Generation <: AbstractOperand

Active power generation operand (MW).
"""
struct Generation <: AbstractOperand end

"""
    ReactiveGeneration <: AbstractOperand

Reactive power generation operand (MVAr).
"""
struct ReactiveGeneration <: AbstractOperand end

"""
    Flow <: AbstractOperand

Branch flow operand (MW for DC, complex for AC).
"""
struct Flow <: AbstractOperand end

"""
    LMP <: AbstractOperand

Locational marginal price operand (\$/MWh).
"""
struct LMP <: AbstractOperand end

"""
    CurrentMagnitude <: AbstractOperand

Current magnitude operand (per unit).
"""
struct CurrentMagnitude <: AbstractOperand end

"""
    Voltage <: AbstractOperand

Complex voltage phasor operand.
"""
struct Voltage <: AbstractOperand end

# =============================================================================
# Parameter Tags (what we differentiate with respect to)
# =============================================================================

"""
    Demand <: AbstractParameter

Demand parameter (MW).
"""
struct Demand <: AbstractParameter end

"""
    Switching <: AbstractParameter

Switching state parameter (0=open, 1=closed).
"""
struct Switching <: AbstractParameter end

"""
    QuadraticCost <: AbstractParameter

Quadratic cost coefficient parameter (\$/MW^2).
"""
struct QuadraticCost <: AbstractParameter end

"""
    LinearCost <: AbstractParameter

Linear cost coefficient parameter (\$/MW).
"""
struct LinearCost <: AbstractParameter end

"""
    FlowLimit <: AbstractParameter

Flow limit parameter (MW).
"""
struct FlowLimit <: AbstractParameter end

"""
    Susceptance <: AbstractParameter

Branch susceptance parameter (per unit).
"""
struct Susceptance <: AbstractParameter end

"""
    ActivePower <: AbstractParameter

Active power injection parameter (MW).
"""
struct ActivePower <: AbstractParameter end

"""
    ReactivePower <: AbstractParameter

Reactive power injection parameter (MVAr).
"""
struct ReactivePower <: AbstractParameter end
