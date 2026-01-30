# =============================================================================
# Unified Sensitivity Interface
# =============================================================================
#
# Type-based dispatch for sensitivity computation:
#   calc_sensitivity(state, Operand(), Parameter()) → Sensitivity{F, O, P}
#
# Returns typed sensitivity results enabling dispatch on formulation, operand,
# and parameter. Includes bidirectional index mappings for element lookup.
#
# Operand types: VoltageAngle, VoltageMagnitude, Generation, Flow, LMP, CurrentMagnitude, Voltage
# Parameter types: Demand, Switching, QuadraticCost, LinearCost, FlowLimit, Susceptance, ActivePower, ReactivePower

"""
    calc_sensitivity(state, operand::AbstractOperand, parameter::AbstractParameter) → Sensitivity{F, O, P}

Compute sensitivity of `operand` with respect to `parameter`.

Returns a `Sensitivity{F, O, P}` typed result that:
- Acts like a matrix (implements AbstractMatrix interface)
- Has type parameters for dispatch (F=formulation, O=operand, P=parameter)
- Includes bidirectional index mappings for element IDs

Invalid combinations throw ArgumentError.

# Operand Types
- `VoltageAngle()`: Voltage phase angles (DC)
- `Flow()`: Branch flows (DC)
- `Generation()`: Generator real power (DC OPF only)
- `LMP()`: Locational marginal prices (DC OPF only)
- `VoltageMagnitude()`: Voltage magnitude (AC)
- `CurrentMagnitude()`: Current magnitude (AC)
- `Voltage()`: Complex voltage phasor (AC)

# Parameter Types
- `Demand()`: Demand
- `Switching()`: Switching states
- `QuadraticCost()`, `LinearCost()`: Cost coefficients (DC OPF)
- `FlowLimit()`: Flow limits (DC OPF)
- `Susceptance()`: Susceptances (DC OPF)
- `ActivePower()`: Active power injection (AC)
- `ReactivePower()`: Reactive power injection (AC)

# Examples
```julia
# DC Power Flow
pf_state = DCPowerFlowState(net, demand)
sens = calc_sensitivity(pf_state, VoltageAngle(), Demand())  # Sensitivity{DCPF, VoltageAngle, Demand}
sens[2, 3]                                                    # Access like matrix
sens.row_to_id[2]                                             # Get bus ID for row 2

# DC OPF
prob = DCOPFProblem(net, demand)
solve!(prob)
sens = calc_sensitivity(prob, LMP(), Demand())     # Sensitivity{DCOPF, LMP, Demand}

# Dispatch on result type
process(s::Sensitivity{DCOPF, LMP, Demand}) = "DC OPF LMP sensitivity"

# AC Power Flow
sens = calc_sensitivity(ac_state, VoltageMagnitude(), ActivePower())  # Sensitivity{ACPF, VoltageMagnitude, ActivePower}
```

# Invalid combinations throw ArgumentError
```julia
calc_sensitivity(pf_state, LMP(), Demand())        # ERROR: no LMP for power flow
calc_sensitivity(pf_state, Generation(), Demand()) # ERROR: no generation dispatch in PF
```
"""
function calc_sensitivity end

# =============================================================================
# Formulation/Operand/Parameter Type Mappings
# =============================================================================

# Map state type to formulation singleton type
_formulation_type(::Type{<:DCPowerFlowState}) = DCPF
_formulation_type(::Type{<:DCOPFProblem}) = DCOPF
_formulation_type(::Type{<:ACPowerFlowState}) = ACPF
_formulation_type(::Type{<:ACOPFProblem}) = ACOPF

# Map operand type to element type for rows (index mapping)
_element_type(::Type{VoltageAngle}) = :bus
_element_type(::Type{VoltageMagnitude}) = :bus
_element_type(::Type{Voltage}) = :bus
_element_type(::Type{LMP}) = :bus
_element_type(::Type{Flow}) = :branch
_element_type(::Type{Generation}) = :gen
_element_type(::Type{ReactiveGeneration}) = :gen
_element_type(::Type{CurrentMagnitude}) = :branch

# Map parameter type to element type for cols (index mapping)
_element_type(::Type{Demand}) = :bus
_element_type(::Type{Switching}) = :branch
_element_type(::Type{QuadraticCost}) = :gen
_element_type(::Type{LinearCost}) = :gen
_element_type(::Type{FlowLimit}) = :branch
_element_type(::Type{Susceptance}) = :branch
_element_type(::Type{ActivePower}) = :bus
_element_type(::Type{ReactivePower}) = :bus

# =============================================================================
# Main Entry Point
# =============================================================================

function calc_sensitivity(state, op::O, param::P) where {O <: AbstractOperand, P <: AbstractParameter}
    # Get type information
    F_type = _formulation_type(typeof(state))
    row_element = _element_type(O)
    col_element = _element_type(P)

    # Compute raw matrix via internal dispatch
    matrix = _calc_sensitivity_impl(state, op, param)

    # Build bidirectional index mappings
    row_mapping = _element_mapping(state, row_element)
    col_mapping = _element_mapping(state, col_element)

    return Sensitivity{F_type, O, P}(Matrix(matrix), row_mapping, col_mapping)
end

# =============================================================================
# Internal Implementation Dispatch
# =============================================================================

# Fallback: invalid combination
function _calc_sensitivity_impl(state, op::O, param::P) where {O <: AbstractOperand, P <: AbstractParameter}
    throw(ArgumentError(
        "calc_sensitivity($(typeof(state)), $(typeof(op))(), $(typeof(param))()) is not defined. " *
        "See ?calc_sensitivity for valid operand/parameter combinations."
    ))
end

# =============================================================================
# DC Power Flow: Implementation
# =============================================================================

# dva/dd = -L+ (phase angles w.r.t. demand)
function _calc_sensitivity_impl(state::DCPowerFlowState, ::VoltageAngle, ::Demand)
    return -state.L_pinv
end

# df/dd: f = W*A*va, so df/dd = W*A*dva/dd
function _calc_sensitivity_impl(state::DCPowerFlowState, ::Flow, ::Demand)
    net = state.net
    W = Diagonal(-net.b .* net.z)
    return W * net.A * (-state.L_pinv)
end

# dva/dz (switching)
function _calc_sensitivity_impl(state::DCPowerFlowState, ::VoltageAngle, ::Switching)
    return calc_sensitivity_switching(state).dva_dz
end

# df/dz (switching)
function _calc_sensitivity_impl(state::DCPowerFlowState, ::Flow, ::Switching)
    return calc_sensitivity_switching(state).df_dz
end

# =============================================================================
# DC OPF: Implementation (uses cached KKT derivatives)
# =============================================================================

# Demand sensitivities (via cached KKT system)
_calc_sensitivity_impl(prob::DCOPFProblem, ::VoltageAngle, ::Demand) =
    _extract_sensitivity(prob, _get_dz_dd!(prob), :va)

_calc_sensitivity_impl(prob::DCOPFProblem, ::Generation, ::Demand) =
    _extract_sensitivity(prob, _get_dz_dd!(prob), :pg)

_calc_sensitivity_impl(prob::DCOPFProblem, ::Flow, ::Demand) =
    _extract_sensitivity(prob, _get_dz_dd!(prob), :f)

_calc_sensitivity_impl(prob::DCOPFProblem, ::LMP, ::Demand) =
    _extract_sensitivity(prob, _get_dz_dd!(prob), :lmp)

# Cost sensitivities (via cached KKT system)
_calc_sensitivity_impl(prob::DCOPFProblem, ::Generation, ::QuadraticCost) =
    _extract_sensitivity(prob, _get_dz_dcq!(prob), :pg)

_calc_sensitivity_impl(prob::DCOPFProblem, ::Generation, ::LinearCost) =
    _extract_sensitivity(prob, _get_dz_dcl!(prob), :pg)

_calc_sensitivity_impl(prob::DCOPFProblem, ::LMP, ::QuadraticCost) =
    _extract_sensitivity(prob, _get_dz_dcq!(prob), :lmp)

_calc_sensitivity_impl(prob::DCOPFProblem, ::LMP, ::LinearCost) =
    _extract_sensitivity(prob, _get_dz_dcl!(prob), :lmp)

# Switching sensitivities (via cached KKT system)
_calc_sensitivity_impl(prob::DCOPFProblem, ::VoltageAngle, ::Switching) =
    _extract_sensitivity(prob, _get_dz_dz!(prob), :va)

_calc_sensitivity_impl(prob::DCOPFProblem, ::Generation, ::Switching) =
    _extract_sensitivity(prob, _get_dz_dz!(prob), :pg)

_calc_sensitivity_impl(prob::DCOPFProblem, ::Flow, ::Switching) =
    _extract_sensitivity(prob, _get_dz_dz!(prob), :f)

_calc_sensitivity_impl(prob::DCOPFProblem, ::LMP, ::Switching) =
    _extract_sensitivity(prob, _get_dz_dz!(prob), :lmp)

# Flow limit sensitivities (via cached KKT system)
_calc_sensitivity_impl(prob::DCOPFProblem, ::VoltageAngle, ::FlowLimit) =
    _extract_sensitivity(prob, _get_dz_dfmax!(prob), :va)

_calc_sensitivity_impl(prob::DCOPFProblem, ::Generation, ::FlowLimit) =
    _extract_sensitivity(prob, _get_dz_dfmax!(prob), :pg)

_calc_sensitivity_impl(prob::DCOPFProblem, ::Flow, ::FlowLimit) =
    _extract_sensitivity(prob, _get_dz_dfmax!(prob), :f)

_calc_sensitivity_impl(prob::DCOPFProblem, ::LMP, ::FlowLimit) =
    _extract_sensitivity(prob, _get_dz_dfmax!(prob), :lmp)

# Susceptance sensitivities (via cached KKT system)
_calc_sensitivity_impl(prob::DCOPFProblem, ::VoltageAngle, ::Susceptance) =
    _extract_sensitivity(prob, _get_dz_db!(prob), :va)

_calc_sensitivity_impl(prob::DCOPFProblem, ::Generation, ::Susceptance) =
    _extract_sensitivity(prob, _get_dz_db!(prob), :pg)

_calc_sensitivity_impl(prob::DCOPFProblem, ::Flow, ::Susceptance) =
    _extract_sensitivity(prob, _get_dz_db!(prob), :f)

_calc_sensitivity_impl(prob::DCOPFProblem, ::LMP, ::Susceptance) =
    _extract_sensitivity(prob, _get_dz_db!(prob), :lmp)

# =============================================================================
# AC Power Flow: Implementation
# =============================================================================

# Voltage magnitude sensitivities w.r.t. active power
_calc_sensitivity_impl(state::ACPowerFlowState, ::VoltageMagnitude, ::ActivePower) =
    calc_voltage_power_sensitivities(state).dvm_dp

# Voltage magnitude sensitivities w.r.t. reactive power
_calc_sensitivity_impl(state::ACPowerFlowState, ::VoltageMagnitude, ::ReactivePower) =
    calc_voltage_power_sensitivities(state).dvm_dq

# Complex voltage sensitivities
_calc_sensitivity_impl(state::ACPowerFlowState, ::Voltage, ::ActivePower) =
    calc_voltage_power_sensitivities(state).dv_dp

_calc_sensitivity_impl(state::ACPowerFlowState, ::Voltage, ::ReactivePower) =
    calc_voltage_power_sensitivities(state).dv_dq

# Current magnitude sensitivities
function _calc_sensitivity_impl(state::ACPowerFlowState, ::CurrentMagnitude, ::ActivePower)
    sens = calc_current_power_sensitivities(state)
    return sens.dIm_dp
end

function _calc_sensitivity_impl(state::ACPowerFlowState, ::CurrentMagnitude, ::ReactivePower)
    sens = calc_current_power_sensitivities(state)
    return sens.dIm_dq
end

# =============================================================================
# AC OPF: Implementation
# =============================================================================

# Voltage magnitude sensitivities w.r.t. switching
_calc_sensitivity_impl(prob::ACOPFProblem, ::VoltageMagnitude, ::Switching) =
    calc_sensitivity_switching(prob).dvm_dz

# Voltage angle sensitivities w.r.t. switching
_calc_sensitivity_impl(prob::ACOPFProblem, ::VoltageAngle, ::Switching) =
    calc_sensitivity_switching(prob).dva_dz

# Active generation sensitivities w.r.t. switching
_calc_sensitivity_impl(prob::ACOPFProblem, ::Generation, ::Switching) =
    calc_sensitivity_switching(prob).dpg_dz

# Reactive generation sensitivities w.r.t. switching
_calc_sensitivity_impl(prob::ACOPFProblem, ::ReactiveGeneration, ::Switching) =
    calc_sensitivity_switching(prob).dqg_dz
