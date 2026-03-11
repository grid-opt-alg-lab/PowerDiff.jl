# Copyright 2026 Samuel Talkington and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
- `:va`: Voltage phase angles (DC PF, DC OPF, AC PF, AC OPF)
- `:f`: Branch active power flows (DC PF, DC OPF, AC PF)
- `:pg` / `:g`: Generator active power (DC OPF, AC OPF)
- `:psh`: Load shedding (DC OPF only)
- `:qg`: Generator reactive power (AC OPF)
- `:lmp`: Locational marginal prices (DC OPF, AC OPF)
- `:qlmp`: Reactive power locational marginal prices (AC OPF)
- `:vm`: Voltage magnitude (AC PF, AC OPF)
- `:im`: Current magnitude (AC PF)
- `:v`: Complex voltage phasor (AC PF) — returns ComplexF64 elements
- `:p`: Active power injection (AC PF, as operand for Jacobian blocks)
- `:q`: Reactive power injection (AC PF, as operand for Jacobian blocks)

# Parameter Symbols
- `:d` / `:pd`: Active demand (DC PF, DC OPF, AC OPF; AC PF via transform)
- `:qd`: Reactive demand (AC OPF; AC PF via transform)
- `:sw`: Switching states
- `:cq`, `:cl`: Cost coefficients (DC OPF, AC OPF)
- `:fmax`: Flow limits (DC OPF, AC OPF)
- `:b`: Branch susceptances (DC PF, DC OPF, AC PF)
- `:g`: Branch conductances (AC PF)
- `:p`: Active power injection (AC PF)
- `:q`: Reactive power injection (AC PF)
- `:va`: Voltage phase angle (AC PF, as parameter for Jacobian blocks)
- `:vm`: Voltage magnitude (AC PF, as parameter for Jacobian blocks)

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

# AC Power Flow — voltage/current sensitivities
sens = calc_sensitivity(ac_state, :vm, :p)
topo = calc_sensitivity(ac_state, :vm, :g)

# AC Power Flow — Jacobian blocks
J1 = calc_sensitivity(ac_state, :p, :va)   # ∂P/∂θ
J2 = calc_sensitivity(ac_state, :p, :vm)   # ∂P/∂|V|

# AC Power Flow — demand transform (∂/∂d = -∂/∂p since p_net = pg - pd)
dvm_dd = calc_sensitivity(ac_state, :vm, :d)

# AC OPF — all parameter types supported
dlmp_dd = calc_sensitivity(ac_prob, :lmp, :d)      # dLMP/dd
dpg_dcq = calc_sensitivity(ac_prob, :pg, :cq)      # dpg/dcq
dvm_dfmax = calc_sensitivity(ac_prob, :vm, :fmax)   # d|V|/dfmax
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

const _VALID_OPERANDS = Set([:va, :f, :pg, :lmp, :qlmp, :psh, :vm, :im, :v, :qg, :p, :q])
const _VALID_PARAMETERS = Set([:d, :sw, :cq, :cl, :fmax, :b, :g, :p, :q, :va, :vm, :qd])

function _resolve_operand(s::Symbol)
    s = get(_OPERAND_ALIASES, s, s)
    s in _VALID_OPERANDS || throw(ArgumentError(
        "Unknown operand symbol :$s. Valid: :va, :f, :pg, :psh, :lmp, :qlmp, :vm, :im, :v, :qg, :p, :q (alias: :g → :pg)"))
    return s
end

function _resolve_parameter(s::Symbol)
    s = get(_PARAMETER_ALIASES, s, s)
    s in _VALID_PARAMETERS || throw(ArgumentError(
        "Unknown parameter symbol :$s. Valid: :d, :sw, :cq, :cl, :fmax, :b, :g, :p, :q, :va, :vm, :qd (alias: :pd → :d)"))
    return s
end

# =============================================================================
# Element Type Mappings (for index mappings)
# =============================================================================

# Map operand symbol to element type for rows
const _OPERAND_ELEMENT = Dict{Symbol, Symbol}(
    :va => :bus, :vm => :bus, :v => :bus, :lmp => :bus, :qlmp => :bus, :psh => :bus,
    :p => :bus, :q => :bus,
    :f => :branch, :im => :branch,
    :pg => :gen, :qg => :gen,
)

# Map parameter symbol to element type for cols
const _PARAM_ELEMENT = Dict{Symbol, Symbol}(
    :d => :bus, :p => :bus, :q => :bus, :va => :bus, :vm => :bus, :qd => :bus,
    :sw => :branch, :fmax => :branch, :b => :branch, :g => :branch,
    :cq => :gen, :cl => :gen,
)

# =============================================================================
# Formulation Symbol Mapping
# =============================================================================

_formulation_symbol(::DCPowerFlowState) = :dcpf
_formulation_symbol(::DCOPFProblem) = :dcopf
_formulation_symbol(::ACPowerFlowState) = :acpf
_formulation_symbol(::ACOPFProblem) = :acopf
_formulation_symbol(state) = throw(ArgumentError(
    "No formulation symbol defined for $(typeof(state)). " *
    "Define _formulation_symbol(::$(typeof(state))) to add support."))

# =============================================================================
# Valid Combinations per Formulation
# =============================================================================

_valid_combinations(::Type{<:DCPowerFlowState}) = [
    (:va, :d), (:f, :d), (:va, :sw), (:f, :sw), (:va, :b), (:f, :b),
]

_valid_combinations(::Type{<:DCOPFProblem}) = [
    (:va, :d), (:pg, :d), (:f, :d), (:psh, :d), (:lmp, :d),
    (:va, :sw), (:pg, :sw), (:f, :sw), (:psh, :sw), (:lmp, :sw),
    (:va, :cq), (:pg, :cq), (:f, :cq), (:psh, :cq), (:lmp, :cq),
    (:va, :cl), (:pg, :cl), (:f, :cl), (:psh, :cl), (:lmp, :cl),
    (:va, :fmax), (:pg, :fmax), (:f, :fmax), (:psh, :fmax), (:lmp, :fmax),
    (:va, :b), (:pg, :b), (:f, :b), (:psh, :b), (:lmp, :b),
]

_valid_combinations(::Type{<:ACPowerFlowState}) = [
    (:vm, :p), (:vm, :q), (:v, :p), (:v, :q), (:im, :p), (:im, :q),
    (:va, :p), (:va, :q),
    (:f, :p), (:f, :q),
    (:p, :va), (:p, :vm), (:q, :va), (:q, :vm),
    # Topology sensitivities (branch conductance/susceptance)
    (:vm, :g), (:va, :g), (:v, :g), (:f, :g), (:im, :g),
    (:vm, :b), (:va, :b), (:v, :b), (:f, :b), (:im, :b),
]

_valid_combinations(::Type{<:ACOPFProblem}) = [
    (:vm, :sw), (:va, :sw), (:pg, :sw), (:qg, :sw), (:lmp, :sw), (:qlmp, :sw),
    (:vm, :d), (:va, :d), (:pg, :d), (:qg, :d), (:lmp, :d), (:qlmp, :d),
    (:vm, :qd), (:va, :qd), (:pg, :qd), (:qg, :qd), (:lmp, :qd), (:qlmp, :qd),
    (:vm, :cq), (:va, :cq), (:pg, :cq), (:qg, :cq), (:lmp, :cq), (:qlmp, :cq),
    (:vm, :cl), (:va, :cl), (:pg, :cl), (:qg, :cl), (:lmp, :cl), (:qlmp, :cl),
    (:vm, :fmax), (:va, :fmax), (:pg, :fmax), (:qg, :fmax), (:lmp, :fmax), (:qlmp, :fmax),
]

_valid_combinations(T::Type) = throw(ArgumentError(
    "No valid sensitivity combinations defined for $T. " *
    "Define _valid_combinations(::Type{<:$T}) to add support."))

# =============================================================================
# Type-Specific Parameter Transforms
# =============================================================================
#
# Transforms allow derived parameter symbols (e.g., :d) to be computed from
# a native parameter (e.g., :p) via a simple scaling. This is type-specific:
#
# For power flow states: p_net = pg - pd with pg fixed, so ∂/∂pd = -∂/∂p.
# For OPF: demand sensitivity goes through KKT — the transform does NOT apply.

"""
    _parameter_transform(::Type{T}, ::Val{param}) → (base_param, transform_fn) or nothing

Return a tuple (base_param, transform_fn) if `param` can be derived from a
native parameter for type `T`. The result satisfies:
    ∂(operand)/∂param = transform_fn(∂(operand)/∂base_param)

Returns `nothing` if no transform exists (default).
"""
_parameter_transform(::Type, ::Val) = nothing

# For ACPowerFlowState: ∂/∂d = -∂/∂p (since p_net = pg - pd, pg fixed)
_parameter_transform(::Type{<:ACPowerFlowState}, ::Val{:d}) = (:p, -)
# For ACPowerFlowState: ∂/∂qd = -∂/∂q (since q_net = qg - qd, qg fixed)
_parameter_transform(::Type{<:ACPowerFlowState}, ::Val{:qd}) = (:q, -)

"""
    _all_valid_combinations(T) → Vector{Tuple{Symbol,Symbol}}

All valid (operand, parameter) combinations for type `T`, including both
native combinations and those derived via parameter transforms.
"""
function _all_valid_combinations(T::Type)
    native = _valid_combinations(T)
    all_combos = copy(native)

    # Collect all transform-derived combinations
    native_params = unique(last.(native))
    native_operands = unique(first.(native))

    # Check all known parameters for transforms
    for param in _VALID_PARAMETERS
        param in native_params && continue
        transform = _parameter_transform(T, Val(param))
        isnothing(transform) && continue
        base_param, _ = transform
        # Add (op, param) for every operand that has (op, base_param)
        for (op, p) in native
            if p === base_param
                push!(all_combos, (op, param))
            end
        end
    end

    return all_combos
end

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

    T = typeof(state)
    valid = _valid_combinations(T)

    if (op, param) in valid
        # Native combination — compute directly
        matrix = _calc_sensitivity_matrix(state, op, param)
    else
        # Check for a parameter transform
        transform = _parameter_transform(T, Val(param))
        if !isnothing(transform)
            base_param, transform_fn = transform
            if (op, base_param) in valid
                base_matrix = _calc_sensitivity_matrix(state, op, base_param)
                matrix = transform_fn(base_matrix)
            else
                _throw_invalid_combo(state, op, param)
            end
        else
            _throw_invalid_combo(state, op, param)
        end
    end

    # Build bidirectional index mappings
    row_element = _OPERAND_ELEMENT[op]
    # For transform-derived params, use the original param's element type
    col_element = _PARAM_ELEMENT[param]
    row_mapping = _element_mapping(state, row_element)
    col_mapping = _element_mapping(state, col_element)

    mat = Matrix(matrix)
    if any(!isfinite, mat)
        error("Sensitivity (:$op, :$param) produced non-finite values. " *
              "The KKT system may be ill-conditioned.")
    end
    form = _formulation_symbol(state)
    return Sensitivity(mat, form, op, param, row_mapping, col_mapping)
end

function _throw_invalid_combo(state, op, param)
    T = typeof(state)
    state_name = _state_display_name(T)
    all_valid = _all_valid_combinations(T)
    msg = "calc_sensitivity($state_name, :$op, :$param) is not defined."
    if !isempty(all_valid)
        msg *= "\nValid combinations for $state_name:\n"
        msg *= join(["  :$(o) w.r.t. :$(p)" for (o, p) in all_valid], "\n")
    end
    throw(ArgumentError(msg))
end

# =============================================================================
# DC Power Flow: Implementation
# =============================================================================

function _calc_sensitivity_matrix(state::DCPowerFlowState, op::Symbol, param::Symbol)
    if param === :d
        sens = calc_sensitivity_demand(state)
        return op === :va ? sens.dva_dd : sens.df_dd
    elseif param === :sw
        sens = calc_sensitivity_switching(state)
        return op === :va ? sens.dva_dsw : sens.df_dsw
    else  # :b
        sens = calc_sensitivity_susceptance(state)
        return op === :va ? sens.dva_db : sens.df_db
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
    cache_fn = get(_DC_OPF_CACHE_FN, param) do
        error("No cached derivative function for DC OPF parameter :$param")
    end
    dz_dp = cache_fn(prob)
    return _extract_sensitivity(prob, dz_dp, op)
end

# =============================================================================
# AC Power Flow: Implementation
# =============================================================================

function _calc_sensitivity_matrix(state::ACPowerFlowState, op::Symbol, param::Symbol)
    # Voltage/phasor/angle sensitivities w.r.t. power injections
    # Compute only the needed direction (P or Q) instead of both
    if op in (:vm, :v, :va) && param in (:p, :q)
        if param === :p
            ∂v, ∂vm, ∂va = calc_voltage_active_power_sensitivities(
                state.v, state.Y; idx_slack=state.idx_slack)
        else
            ∂v, ∂vm, ∂va = calc_voltage_reactive_power_sensitivities(
                state.v, state.Y; idx_slack=state.idx_slack)
        end
        if op === :vm
            return ∂vm
        elseif op === :v
            return ∂v
        else
            return ∂va
        end
    end

    # Current magnitude sensitivity — compute only needed direction
    if op === :im && param in (:p, :q)
        ∂v = _voltage_phasor_single_dir(state, param)
        return _current_magnitude_from_dv(∂v, state)
    end

    # Branch active power flow sensitivity — compute only needed direction
    if op === :f && param in (:p, :q)
        ∂v = _voltage_phasor_single_dir(state, param)
        return _branch_flow_from_dv(∂v, state)
    end

    # Topology sensitivities (conductance/susceptance)
    if param in (:g, :b)
        dv, dvm, dva = _solve_topology_sensitivities(state, param)
        op === :vm && return dvm
        op === :va && return dva
        op === :v  && return dv
        op === :im && return _current_magnitude_from_dv(dv, state, param)
        op === :f  && return _branch_flow_from_dv(dv, state, param)
    end

    # Power flow Jacobian blocks (operand is :p or :q, param is :va or :vm)
    if op in (:p, :q) && param in (:va, :vm)
        jac = calc_power_flow_jacobian(state)
        if op === :p
            return param === :va ? jac.dp_dva : jac.dp_dvm
        else  # :q
            return param === :va ? jac.dq_dva : jac.dq_dvm
        end
    end

    error("Unhandled AC PF combination (:$op, :$param)")
end

# =============================================================================
# AC PF Single-Direction Helpers
# =============================================================================

"""Compute complex voltage phasor sensitivity for a single direction (:p or :q)."""
function _voltage_phasor_single_dir(state::ACPowerFlowState, param::Symbol)
    if param === :p
        ∂v, _, _ = calc_voltage_active_power_sensitivities(
            state.v, state.Y; idx_slack=state.idx_slack)
    else
        ∂v, _, _ = calc_voltage_reactive_power_sensitivities(
            state.v, state.Y; idx_slack=state.idx_slack)
    end
    return ∂v
end

"""Compute branch current phasor sensitivity from pre-computed voltage phasor sensitivity."""
function _branch_current_from_dv(∂v, state::ACPowerFlowState)
    !isnothing(state.branch_data) || throw(ArgumentError(
        "ACPowerFlowState must have branch_data for current sensitivities"))
    m = state.m
    ncols = size(∂v, 2)
    ∂I = zeros(ComplexF64, m, ncols)
    for (_, br) in state.branch_data
        ℓ = br["index"]
        f_bus = br["f_bus"]
        t_bus = br["t_bus"]
        Y_ft = state.Y[f_bus, t_bus]
        ∂I[ℓ, :] = Y_ft .* (∂v[f_bus, :] .- ∂v[t_bus, :])
    end
    return ∂I
end

"""Compute topology direct current term from explicit dependence on branch admittance."""
function _topology_branch_current_direct(state::ACPowerFlowState, param::Symbol)
    net = state.net
    isnothing(net) && throw(ArgumentError(
        "ACPowerFlowState must have an ACNetwork (net field) for topology sensitivities"))

    m = state.m
    ∂I_direct = zeros(ComplexF64, m, m)
    ΔV = net.A * state.v
    pair_to_cols = Dict{Tuple{Int,Int}, Vector{Int}}()

    for e in 1:m
        i, j = net.incidences[e]
        key = i < j ? (i, j) : (j, i)
        push!(get!(pair_to_cols, key, Int[]), e)
    end

    scale = param === :g ? ComplexF64(-1.0, 0.0) : ComplexF64(0.0, -1.0)
    for ℓ in 1:m
        i, j = net.incidences[ℓ]
        key = i < j ? (i, j) : (j, i)
        for e in pair_to_cols[key]
            ∂I_direct[ℓ, e] = scale * ΔV[ℓ]
        end
    end

    return ∂I_direct
end

"""Project complex branch current sensitivities to current magnitude sensitivities."""
function _current_magnitude_from_dI(∂I, state::ACPowerFlowState)
    !isnothing(state.branch_data) || throw(ArgumentError(
        "ACPowerFlowState must have branch_data for current sensitivities"))
    v = state.v
    m = state.m
    ncols = size(∂I, 2)
    ∂Im = zeros(Float64, m, ncols)
    n_suppressed = 0
    for (_, br) in state.branch_data
        ℓ = br["index"]
        f_bus = br["f_bus"]
        t_bus = br["t_bus"]
        Y_ft = state.Y[f_bus, t_bus]
        I_ℓ = Y_ft * (v[f_bus] - v[t_bus])
        if abs(I_ℓ) <= VOLTAGE_ZERO_TOL
            n_suppressed += 1
            continue
        end
        ∂Im[ℓ, :] = real.(∂I[ℓ, :] .* conj(I_ℓ)) ./ abs(I_ℓ)
    end
    if n_suppressed > 0
        @debug "Current magnitude sensitivity: $n_suppressed branches had |I| < $VOLTAGE_ZERO_TOL; their ∂|I| rows are zero."
    end
    return ∂Im
end

"""Compute current magnitude sensitivity from pre-computed voltage phasor sensitivity."""
function _current_magnitude_from_dv(∂v, state::ACPowerFlowState)
    return _current_magnitude_from_dI(_branch_current_from_dv(∂v, state), state)
end

"""Compute current magnitude sensitivity from pre-computed voltage phasor sensitivity."""
function _current_magnitude_from_dv(∂v, state::ACPowerFlowState, param::Symbol)
    ∂I = _branch_current_from_dv(∂v, state) + _topology_branch_current_direct(state, param)
    return _current_magnitude_from_dI(∂I, state)
end

"""Project voltage/current sensitivities to branch active power flow sensitivities."""
function _branch_flow_from_dv_dI(∂v, ∂I, state::ACPowerFlowState)
    !isnothing(state.branch_data) || throw(ArgumentError(
        "ACPowerFlowState must have branch_data for flow sensitivities"))
    v = state.v
    m = state.m
    ncols = size(∂I, 2)
    df = zeros(Float64, m, ncols)
    for (_, br) in state.branch_data
        ℓ = br["index"]
        f_bus = br["f_bus"]
        t_bus = br["t_bus"]
        Y_ft = state.Y[f_bus, t_bus]
        I_ℓ = Y_ft * (v[f_bus] - v[t_bus])
        v_f = v[f_bus]
        df[ℓ, :] = real.(∂v[f_bus, :] .* conj(I_ℓ) .+ v_f .* conj.(∂I[ℓ, :]))
    end
    return df
end

"""Compute branch active power flow sensitivity from pre-computed voltage phasor sensitivity."""
function _branch_flow_from_dv(∂v, state::ACPowerFlowState)
    ∂I = _branch_current_from_dv(∂v, state)
    return _branch_flow_from_dv_dI(∂v, ∂I, state)
end

"""Compute branch active power flow sensitivity from pre-computed voltage phasor sensitivity."""
function _branch_flow_from_dv(∂v, state::ACPowerFlowState, param::Symbol)
    ∂I = _branch_current_from_dv(∂v, state) + _topology_branch_current_direct(state, param)
    return _branch_flow_from_dv_dI(∂v, ∂I, state)
end

# =============================================================================
# AC OPF: Implementation (uses cached KKT derivatives)
# =============================================================================

function _calc_sensitivity_matrix(prob::ACOPFProblem, op::Symbol, param::Symbol)
    dz_dp = _get_ac_dz_dparam!(prob, param)
    idx = kkt_indices(prob)
    if op === :va
        return dz_dp[idx.va, :]
    elseif op === :vm
        return dz_dp[idx.vm, :]
    elseif op === :pg
        return dz_dp[idx.pg, :]
    elseif op === :qg
        return dz_dp[idx.qg, :]
    elseif op === :lmp
        return -dz_dp[idx.nu_p_bal, :]
    elseif op === :qlmp
        return -dz_dp[idx.nu_q_bal, :]
    else
        throw(ArgumentError("Unknown AC OPF operand: $op"))
    end
end

# =============================================================================
# Symbol Introspection API
# =============================================================================

"""
    operand_symbols(state) → Vector{Symbol}

Return the valid operand symbols for `calc_sensitivity` on the given state or problem.
Includes symbols available via parameter transforms.

# Examples
```julia
operand_symbols(pf_state)  # [:va, :f]
operand_symbols(prob)       # [:va, :pg, :f, :psh, :lmp]  (DCOPFProblem)
```
"""
operand_symbols(state) = unique(first.(_all_valid_combinations(typeof(state))))

"""
    parameter_symbols(state) → Vector{Symbol}

Return the valid parameter symbols for `calc_sensitivity` on the given state or problem.
Includes symbols available via parameter transforms.

# Examples
```julia
parameter_symbols(pf_state)  # [:d, :sw, :b]
parameter_symbols(prob)       # [:d, :sw, :cq, :cl, :fmax, :b]  (DCOPFProblem)
```
"""
parameter_symbols(state) = unique(last.(_all_valid_combinations(typeof(state))))
