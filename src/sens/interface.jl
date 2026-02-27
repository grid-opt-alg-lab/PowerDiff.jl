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
- `VoltageAngle()`: Voltage phase angles (DC PF, DC OPF, AC OPF)
- `Flow()`: Branch flows (DC PF, DC OPF)
- `Generation()`: Generator active power (DC OPF, AC OPF)
- `ReactiveGeneration()`: Generator reactive power (AC OPF)
- `LMP()`: Locational marginal prices (DC OPF only)
- `VoltageMagnitude()`: Voltage magnitude (AC PF, AC OPF)
- `CurrentMagnitude()`: Current magnitude (AC PF)
- `Voltage()`: Complex voltage phasor (AC PF) — returns ComplexF64 elements

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
# Symbol-Based Entry Point
# =============================================================================

"""
    calc_sensitivity(state, operand::Symbol, parameter::Symbol)

Symbol-based entry point for sensitivity computation.

Maps symbol names to singleton types and delegates to the typed dispatch:
    calc_sensitivity(state, OperandType(), ParameterType()) → Sensitivity{F, O, P}

# Examples
```julia
calc_sensitivity(prob, :lmp, :d)    # equivalent to calc_sensitivity(prob, LMP(), Demand())
calc_sensitivity(prob, :pg, :cq)    # equivalent to calc_sensitivity(prob, Generation(), QuadraticCost())
calc_sensitivity(state, :vm, :p)    # equivalent to calc_sensitivity(state, VoltageMagnitude(), ActivePower())
```
"""
function calc_sensitivity(state, operand::Symbol, parameter::Symbol)
    op = _resolve_operand(operand)
    param = _resolve_parameter(parameter)
    return calc_sensitivity(state, op, param)
end

_resolve_operand(s::Symbol) = get(_OPERAND_MAP, s) do
    throw(ArgumentError("Unknown operand symbol :$s. Valid: :va, :f, :pg, :g, :lmp, :vm, :im, :v, :qg"))
end
_resolve_parameter(s::Symbol) = get(_PARAMETER_MAP, s) do
    throw(ArgumentError("Unknown parameter symbol :$s. Valid: :d, :pd, :z, :cq, :cl, :fmax, :b, :p, :q"))
end

const _OPERAND_MAP = Dict{Symbol, AbstractOperand}(
    :va => VoltageAngle(), :f => Flow(), :pg => Generation(), :g => Generation(),
    :lmp => LMP(), :vm => VoltageMagnitude(), :im => CurrentMagnitude(),
    :v => Voltage(), :qg => ReactiveGeneration(),
)

const _PARAMETER_MAP = Dict{Symbol, AbstractParameter}(
    :d => Demand(), :pd => Demand(), :z => Switching(),
    :cq => QuadraticCost(), :cl => LinearCost(),
    :fmax => FlowLimit(), :b => Susceptance(),
    :p => ActivePower(), :q => ReactivePower(),
)

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

    mat = Matrix(matrix)
    return Sensitivity{F_type, O, P}(mat, row_mapping, col_mapping)
end

# =============================================================================
# Internal Implementation Dispatch
# =============================================================================

# Fallback: invalid combination with actionable error message
function _calc_sensitivity_impl(state, op::O, param::P) where {O <: AbstractOperand, P <: AbstractParameter}
    state_name = _state_display_name(typeof(state))
    valid = _valid_combinations(typeof(state))
    msg = "calc_sensitivity($state_name, $(nameof(O))(), $(nameof(P))()) is not defined."
    if !isempty(valid)
        msg *= "\nValid combinations for $state_name:\n"
        msg *= join(["  :$(o) w.r.t. :$(p)" for (o, p) in valid], "\n")
    end
    throw(ArgumentError(msg))
end

_state_display_name(::Type{<:DCPowerFlowState}) = "DCPowerFlowState"
_state_display_name(::Type{<:DCOPFProblem}) = "DCOPFProblem"
_state_display_name(::Type{<:ACPowerFlowState}) = "ACPowerFlowState"
_state_display_name(::Type{<:ACOPFProblem}) = "ACOPFProblem"
_state_display_name(T::Type) = string(T)

# Auto-discover valid (operand, parameter) combinations from the method table.
# This avoids manually maintaining lists that can drift from implementations.
function _valid_combinations(::Type{T}) where T
    combos = Tuple{Symbol,Symbol}[]
    for (op_sym, op) in _OPERAND_MAP
        op_sym === :g && continue  # skip alias (:g is alias for :pg)
        op_sym === :pd && continue # skip alias (:pd is alias for :d)
        for (p_sym, param) in _PARAMETER_MAP
            p_sym === :pd && continue # skip alias
            if hasmethod(_calc_sensitivity_impl, Tuple{T, typeof(op), typeof(param)})
                push!(combos, (op_sym, p_sym))
            end
        end
    end
    return unique(combos)
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
_calc_sensitivity_impl(prob::DCOPFProblem, ::VoltageAngle, ::QuadraticCost) =
    _extract_sensitivity(prob, _get_dz_dcq!(prob), :va)

_calc_sensitivity_impl(prob::DCOPFProblem, ::VoltageAngle, ::LinearCost) =
    _extract_sensitivity(prob, _get_dz_dcl!(prob), :va)

_calc_sensitivity_impl(prob::DCOPFProblem, ::Generation, ::QuadraticCost) =
    _extract_sensitivity(prob, _get_dz_dcq!(prob), :pg)

_calc_sensitivity_impl(prob::DCOPFProblem, ::Generation, ::LinearCost) =
    _extract_sensitivity(prob, _get_dz_dcl!(prob), :pg)

_calc_sensitivity_impl(prob::DCOPFProblem, ::Flow, ::QuadraticCost) =
    _extract_sensitivity(prob, _get_dz_dcq!(prob), :f)

_calc_sensitivity_impl(prob::DCOPFProblem, ::Flow, ::LinearCost) =
    _extract_sensitivity(prob, _get_dz_dcl!(prob), :f)

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
# AC OPF: Implementation (uses cached KKT derivatives)
# =============================================================================

# Voltage magnitude sensitivities w.r.t. switching
function _calc_sensitivity_impl(prob::ACOPFProblem, ::VoltageMagnitude, ::Switching)
    dx_ds = _get_ac_dx_ds!(prob)
    idx = ac_kkt_indices(prob)
    return dx_ds[idx.vm, :]
end

# Voltage angle sensitivities w.r.t. switching
function _calc_sensitivity_impl(prob::ACOPFProblem, ::VoltageAngle, ::Switching)
    dx_ds = _get_ac_dx_ds!(prob)
    idx = ac_kkt_indices(prob)
    return dx_ds[idx.va, :]
end

# Active generation sensitivities w.r.t. switching
function _calc_sensitivity_impl(prob::ACOPFProblem, ::Generation, ::Switching)
    dx_ds = _get_ac_dx_ds!(prob)
    idx = ac_kkt_indices(prob)
    return dx_ds[idx.pg, :]
end

# Reactive generation sensitivities w.r.t. switching
function _calc_sensitivity_impl(prob::ACOPFProblem, ::ReactiveGeneration, ::Switching)
    dx_ds = _get_ac_dx_ds!(prob)
    idx = ac_kkt_indices(prob)
    return dx_ds[idx.qg, :]
end
