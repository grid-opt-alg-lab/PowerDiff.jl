# =============================================================================
# Unified Sensitivity Interface
# =============================================================================
#
# Symbol-based dispatch for sensitivity computation:
#   calc_sensitivity(state, :operand, :parameter) → Sensitivity{T}
#
# Returns sensitivity results with symbol metadata and bidirectional index mappings.

"""
    calc_sensitivity(state, operand::Symbol, parameter::Symbol) → Sensitivity{T}

Compute sensitivity of `operand` with respect to `parameter`.

Returns a `Sensitivity{T}` result that:
- Acts like a matrix (implements AbstractMatrix interface)
- Has symbol fields for formulation, operand, and parameter
- Includes bidirectional index mappings for element IDs

Invalid combinations throw ArgumentError.

# Operand Symbols
- `:va`: Voltage phase angles (DC PF, DC OPF, AC OPF)
- `:f`: Branch flows (DC PF, DC OPF)
- `:pg` / `:g`: Generator active power (DC OPF, AC OPF)
- `:qg`: Generator reactive power (AC OPF)
- `:lmp`: Locational marginal prices (DC OPF only)
- `:vm`: Voltage magnitude (AC PF, AC OPF)
- `:im`: Current magnitude (AC PF)
- `:v`: Complex voltage phasor (AC PF) — returns ComplexF64 elements

# Parameter Symbols
- `:d` / `:pd`: Demand
- `:sw`: Switching states
- `:cq`, `:cl`: Cost coefficients (DC OPF)
- `:fmax`: Flow limits (DC OPF)
- `:b`: Susceptances (DC OPF)
- `:p`: Active power injection (AC)
- `:q`: Reactive power injection (AC)

# Examples
```julia
# DC Power Flow
pf_state = DCPowerFlowState(net, demand)
sens = calc_sensitivity(pf_state, :va, :d)
sens.formulation  # :dcpf
sens.operand      # :va
sens.parameter    # :d

# DC OPF
prob = DCOPFProblem(net, demand)
solve!(prob)
sens = calc_sensitivity(prob, :lmp, :d)

# AC Power Flow
sens = calc_sensitivity(ac_state, :vm, :p)
```
"""
function calc_sensitivity end

# =============================================================================
# Alias Resolution
# =============================================================================

const _OPERAND_ALIASES = Dict{Symbol, Symbol}(
    :g => :pg,
)

const _PARAMETER_ALIASES = Dict{Symbol, Symbol}(
    :pd => :d,
)

const _VALID_OPERANDS = Set([:va, :f, :pg, :lmp, :vm, :im, :v, :qg])
const _VALID_PARAMETERS = Set([:d, :sw, :cq, :cl, :fmax, :b, :p, :q])

function _resolve_operand(s::Symbol)
    s = get(_OPERAND_ALIASES, s, s)
    s in _VALID_OPERANDS || throw(ArgumentError(
        "Unknown operand symbol :$s. Valid: :va, :f, :pg, :lmp, :vm, :im, :v, :qg (alias: :g → :pg)"))
    return s
end

function _resolve_parameter(s::Symbol)
    s = get(_PARAMETER_ALIASES, s, s)
    s in _VALID_PARAMETERS || throw(ArgumentError(
        "Unknown parameter symbol :$s. Valid: :d, :sw, :cq, :cl, :fmax, :b, :p, :q (alias: :pd → :d)"))
    return s
end

# =============================================================================
# Element Type Mappings (for index mappings)
# =============================================================================

# Map operand symbol to element type for rows
const _OPERAND_ELEMENT = Dict{Symbol, Symbol}(
    :va => :bus, :vm => :bus, :v => :bus, :lmp => :bus,
    :f => :branch, :im => :branch,
    :pg => :gen, :qg => :gen,
)

# Map parameter symbol to element type for cols
const _PARAM_ELEMENT = Dict{Symbol, Symbol}(
    :d => :bus, :p => :bus, :q => :bus,
    :sw => :branch, :fmax => :branch, :b => :branch,
    :cq => :gen, :cl => :gen,
)

# =============================================================================
# Formulation Symbol Mapping
# =============================================================================

_formulation_symbol(::DCPowerFlowState) = :dcpf
_formulation_symbol(::DCOPFProblem) = :dcopf
_formulation_symbol(::ACPowerFlowState) = :acpf
_formulation_symbol(::ACOPFProblem) = :acopf

# =============================================================================
# Valid Combinations per Formulation
# =============================================================================

_valid_combinations(::Type{<:DCPowerFlowState}) = [
    (:va, :d), (:f, :d), (:va, :sw), (:f, :sw),
]

_valid_combinations(::Type{<:DCOPFProblem}) = [
    (:va, :d), (:pg, :d), (:f, :d), (:lmp, :d),
    (:va, :sw), (:pg, :sw), (:f, :sw), (:lmp, :sw),
    (:va, :cq), (:pg, :cq), (:f, :cq), (:lmp, :cq),
    (:va, :cl), (:pg, :cl), (:f, :cl), (:lmp, :cl),
    (:va, :fmax), (:pg, :fmax), (:f, :fmax), (:lmp, :fmax),
    (:va, :b), (:pg, :b), (:f, :b), (:lmp, :b),
]

_valid_combinations(::Type{<:ACPowerFlowState}) = [
    (:vm, :p), (:vm, :q), (:v, :p), (:v, :q), (:im, :p), (:im, :q),
]

_valid_combinations(::Type{<:ACOPFProblem}) = [
    (:vm, :sw), (:va, :sw), (:pg, :sw), (:qg, :sw),
]

# =============================================================================
# Main Entry Point
# =============================================================================

_state_display_name(::Type{<:DCPowerFlowState}) = "DCPowerFlowState"
_state_display_name(::Type{<:DCOPFProblem}) = "DCOPFProblem"
_state_display_name(::Type{<:ACPowerFlowState}) = "ACPowerFlowState"
_state_display_name(::Type{<:ACOPFProblem}) = "ACOPFProblem"
_state_display_name(T::Type) = string(T)

function calc_sensitivity(state, operand::Symbol, parameter::Symbol)
    op = _resolve_operand(operand)
    param = _resolve_parameter(parameter)

    # Check validity
    valid = _valid_combinations(typeof(state))
    if (op, param) ∉ valid
        state_name = _state_display_name(typeof(state))
        msg = "calc_sensitivity($state_name, :$op, :$param) is not defined."
        if !isempty(valid)
            msg *= "\nValid combinations for $state_name:\n"
            msg *= join(["  :$(o) w.r.t. :$(p)" for (o, p) in valid], "\n")
        end
        throw(ArgumentError(msg))
    end

    # Compute raw matrix via internal dispatch
    matrix = _calc_sensitivity_matrix(state, op, param)

    # Build bidirectional index mappings
    row_element = _OPERAND_ELEMENT[op]
    col_element = _PARAM_ELEMENT[param]
    row_mapping = _element_mapping(state, row_element)
    col_mapping = _element_mapping(state, col_element)

    mat = Matrix(matrix)
    form = _formulation_symbol(state)
    return Sensitivity(mat, form, op, param, row_mapping, col_mapping)
end

# =============================================================================
# DC Power Flow: Implementation
# =============================================================================

function _calc_sensitivity_matrix(state::DCPowerFlowState, op::Symbol, param::Symbol)
    if param === :d
        sens = calc_sensitivity_demand(state)
        return op === :va ? sens.dva_dd : sens.df_dd
    else  # :sw
        sens = calc_sensitivity_switching(state)
        if op === :va
            return sens.dva_dsw
        else  # :f
            return sens.df_dsw
        end
    end
end

# =============================================================================
# DC OPF: Implementation (uses cached KKT derivatives)
# =============================================================================

# Map parameter symbols to cached derivative functions
const _DC_OPF_CACHE_FN = Dict{Symbol, Function}(
    :d    => _get_dz_dd!,
    :sw   => _get_dz_dsw!,
    :cq   => _get_dz_dcq!,
    :cl   => _get_dz_dcl!,
    :fmax => _get_dz_dfmax!,
    :b    => _get_dz_db!,
)

function _calc_sensitivity_matrix(prob::DCOPFProblem, op::Symbol, param::Symbol)
    cache_fn = _DC_OPF_CACHE_FN[param]
    dz_dp = cache_fn(prob)
    return _extract_sensitivity(prob, dz_dp, op)
end

# =============================================================================
# AC Power Flow: Implementation
# =============================================================================

function _calc_sensitivity_matrix(state::ACPowerFlowState, op::Symbol, param::Symbol)
    if op === :vm || op === :v
        sens = calc_voltage_power_sensitivities(state)
        if op === :vm
            return param === :p ? sens.dvm_dp : sens.dvm_dq
        else  # :v
            return param === :p ? sens.dv_dp : sens.dv_dq
        end
    else  # :im
        sens = calc_current_power_sensitivities(state)
        return param === :p ? sens.dIm_dp : sens.dIm_dq
    end
end

# =============================================================================
# AC OPF: Implementation (uses cached KKT derivatives)
# =============================================================================

function _calc_sensitivity_matrix(prob::ACOPFProblem, op::Symbol, param::Symbol)
    param === :sw || throw(ArgumentError(
        "ACOPFProblem currently only supports :sw parameter, got :$param"))
    dx_ds = _get_ac_dx_ds!(prob)
    idx = ac_kkt_indices(prob)
    if op === :vm
        return dx_ds[idx.vm, :]
    elseif op === :va
        return dx_ds[idx.va, :]
    elseif op === :pg
        return dx_ds[idx.pg, :]
    else  # :qg
        return dx_ds[idx.qg, :]
    end
end
