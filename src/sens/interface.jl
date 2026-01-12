# =============================================================================
# Unified Sensitivity Interface
# =============================================================================
#
# Symbol-based dispatch for sensitivity computation:
#   calc_sensitivity(state, :operand, :parameter) → Matrix
#
# Operand symbols: :va (angles), :f (flows), :pg (generation), :lmp, :vm, :im
# Parameter symbols: :d/:pd (demand), :z (switching), :cq/:cl (cost), :p/:q (power)

"""
    calc_sensitivity(state, operand::Symbol, parameter::Symbol) → Matrix

Compute sensitivity of `operand` with respect to `parameter`.

Returns a single matrix - only computes what you ask for.
Invalid combinations throw ArgumentError.

# Operand Symbols
- `:va`: Voltage phase angles (DC)
- `:f`: Branch flows (DC)
- `:pg` or `:g`: Generator real power (DC OPF only)
- `:lmp`: Locational marginal prices (DC OPF only)
- `:vm`: Voltage magnitude (AC)
- `:im`: Current magnitude (AC)

# Parameter Symbols
- `:d` or `:pd`: Demand
- `:z`: Switching states
- `:cq`, `:cl`: Cost coefficients (DC OPF)
- `:fmax`: Flow limits (DC OPF)
- `:b`: Susceptances (DC OPF)
- `:p`: Active power injection (AC)
- `:q`: Reactive power injection (AC)

# Examples
```julia
# DC Power Flow
pf_state = DCPowerFlowState(net, d)
dva_dd = calc_sensitivity(pf_state, :va, :d)    # n×n
df_dd = calc_sensitivity(pf_state, :f, :d)      # m×n
dva_dz = calc_sensitivity(pf_state, :va, :z)    # n×m

# DC OPF
prob = DCOPFProblem(net, d)
solve!(prob)
dlmp_dd = calc_sensitivity(prob, :lmp, :d)      # n×n
dpg_dcq = calc_sensitivity(prob, :pg, :cq)      # k×k

# AC Power Flow
dvm_dp = calc_sensitivity(ac_state, :vm, :p)    # n×n
```

# Invalid combinations throw ArgumentError
```julia
calc_sensitivity(pf_state, :lmp, :d)  # ERROR: no LMP for power flow
calc_sensitivity(pf_state, :pg, :d)   # ERROR: no generation dispatch in PF
```
"""
function calc_sensitivity end

# =============================================================================
# Symbol Dispatch Infrastructure
# =============================================================================

# Main entry point: convert symbols to Val types for dispatch
function calc_sensitivity(state, operand::Symbol, parameter::Symbol)
    _calc_sensitivity(state, Val(operand), Val(parameter))
end

# Fallback: invalid combination
function _calc_sensitivity(state, ::Val{O}, ::Val{P}) where {O, P}
    throw(ArgumentError(
        "calc_sensitivity($(typeof(state)), :$O, :$P) is not defined. " *
        "See ?calc_sensitivity for valid operand/parameter combinations."
    ))
end

# =============================================================================
# DC Power Flow: Symbol Dispatch
# =============================================================================

# dva/dd = -L+ (phase angles w.r.t. demand)
# Since va = L+ * p and p = g - d, we have dva/dd = -L+
# Uses cached L_pinv from state for O(1) lookup instead of O(n³) pseudoinverse
function _calc_sensitivity(state::DCPowerFlowState, ::Val{:va}, ::Val{:d})
    return -state.L_pinv
end

# df/dd: f = W*A*va, so df/dd = W*A*dva/dd where W = Diag(-b.*z)
# Uses cached L_pinv from state
function _calc_sensitivity(state::DCPowerFlowState, ::Val{:f}, ::Val{:d})
    net = state.net
    W = Diagonal(-net.b .* net.z)
    return W * net.A * (-state.L_pinv)
end

# dva/dz (switching) - delegates to calc_sensitivity_switching for consistency
_calc_sensitivity(state::DCPowerFlowState, ::Val{:va}, ::Val{:z}) =
    calc_sensitivity_switching(state).dva_dz

# df/dz (switching) - delegates to calc_sensitivity_switching for consistency
_calc_sensitivity(state::DCPowerFlowState, ::Val{:f}, ::Val{:z}) =
    calc_sensitivity_switching(state).df_dz

# Aliases for :pd -> :d
_calc_sensitivity(s::DCPowerFlowState, ::Val{:va}, ::Val{:pd}) = _calc_sensitivity(s, Val(:va), Val(:d))
_calc_sensitivity(s::DCPowerFlowState, ::Val{:f}, ::Val{:pd}) = _calc_sensitivity(s, Val(:f), Val(:d))

# =============================================================================
# DC Power Flow: Legacy Two-Argument API (Deprecated)
# =============================================================================

function calc_sensitivity(state::DCPowerFlowState, ::DemandParameter)
    Base.depwarn(
        "calc_sensitivity(state, DEMAND) is deprecated. " *
        "Use calc_sensitivity(state, :va, :d) or calc_sensitivity(state, :f, :d) instead.",
        :calc_sensitivity
    )
    calc_sensitivity_demand(state)
end

function calc_sensitivity(state::DCPowerFlowState, ::SwitchingParameter)
    Base.depwarn(
        "calc_sensitivity(state, SWITCHING) is deprecated. " *
        "Use calc_sensitivity(state, :va, :z) or calc_sensitivity(state, :f, :z) instead.",
        :calc_sensitivity
    )
    calc_sensitivity_switching(state)
end

# =============================================================================
# DC OPF: Symbol Dispatch
# =============================================================================

# Demand sensitivities (via KKT system)
_calc_sensitivity(prob::DCOPFProblem, ::Val{:va}, ::Val{:d}) =
    calc_sensitivity_demand(prob).dva_dd

_calc_sensitivity(prob::DCOPFProblem, ::Val{:pg}, ::Val{:d}) =
    calc_sensitivity_demand(prob).dg_dd

_calc_sensitivity(prob::DCOPFProblem, ::Val{:f}, ::Val{:d}) =
    calc_sensitivity_demand(prob).df_dd

_calc_sensitivity(prob::DCOPFProblem, ::Val{:lmp}, ::Val{:d}) =
    calc_sensitivity_demand(prob).dlmp_dd

# Cost sensitivities
_calc_sensitivity(prob::DCOPFProblem, ::Val{:pg}, ::Val{:cq}) =
    calc_sensitivity_cost(prob).dg_dcq

_calc_sensitivity(prob::DCOPFProblem, ::Val{:pg}, ::Val{:cl}) =
    calc_sensitivity_cost(prob).dg_dcl

_calc_sensitivity(prob::DCOPFProblem, ::Val{:lmp}, ::Val{:cq}) =
    calc_sensitivity_cost(prob).dlmp_dcq

_calc_sensitivity(prob::DCOPFProblem, ::Val{:lmp}, ::Val{:cl}) =
    calc_sensitivity_cost(prob).dlmp_dcl

# Switching sensitivities
_calc_sensitivity(prob::DCOPFProblem, ::Val{:va}, ::Val{:z}) =
    calc_sensitivity_switching(prob).dva_dz

_calc_sensitivity(prob::DCOPFProblem, ::Val{:pg}, ::Val{:z}) =
    calc_sensitivity_switching(prob).dg_dz

_calc_sensitivity(prob::DCOPFProblem, ::Val{:f}, ::Val{:z}) =
    calc_sensitivity_switching(prob).df_dz

_calc_sensitivity(prob::DCOPFProblem, ::Val{:lmp}, ::Val{:z}) =
    calc_sensitivity_switching(prob).dlmp_dz

# Flow limit sensitivities
_calc_sensitivity(prob::DCOPFProblem, ::Val{:va}, ::Val{:fmax}) =
    calc_sensitivity_flowlimit(prob).dva_dfmax

_calc_sensitivity(prob::DCOPFProblem, ::Val{:pg}, ::Val{:fmax}) =
    calc_sensitivity_flowlimit(prob).dg_dfmax

_calc_sensitivity(prob::DCOPFProblem, ::Val{:f}, ::Val{:fmax}) =
    calc_sensitivity_flowlimit(prob).df_dfmax

_calc_sensitivity(prob::DCOPFProblem, ::Val{:lmp}, ::Val{:fmax}) =
    calc_sensitivity_flowlimit(prob).dlmp_dfmax

# Susceptance sensitivities
_calc_sensitivity(prob::DCOPFProblem, ::Val{:va}, ::Val{:b}) =
    calc_sensitivity_susceptance(prob).dva_db

_calc_sensitivity(prob::DCOPFProblem, ::Val{:pg}, ::Val{:b}) =
    calc_sensitivity_susceptance(prob).dg_db

_calc_sensitivity(prob::DCOPFProblem, ::Val{:f}, ::Val{:b}) =
    calc_sensitivity_susceptance(prob).df_db

_calc_sensitivity(prob::DCOPFProblem, ::Val{:lmp}, ::Val{:b}) =
    calc_sensitivity_susceptance(prob).dlmp_db

# Aliases for DCOPFProblem: :g -> :pg, :pd -> :d
_calc_sensitivity(p::DCOPFProblem, ::Val{:g}, param::Val) = _calc_sensitivity(p, Val(:pg), param)
_calc_sensitivity(p::DCOPFProblem, op::Val, ::Val{:pd}) = _calc_sensitivity(p, op, Val(:d))

# =============================================================================
# DC OPF: Legacy Two-Argument API (Deprecated)
# =============================================================================

function calc_sensitivity(prob::DCOPFProblem, ::DemandParameter)
    Base.depwarn(
        "calc_sensitivity(prob, DEMAND) is deprecated. " *
        "Use calc_sensitivity(prob, :va, :d), calc_sensitivity(prob, :lmp, :d), etc.",
        :calc_sensitivity
    )
    return calc_sensitivity_demand(prob)
end

function calc_sensitivity(prob::DCOPFProblem, ::CostParameter)
    Base.depwarn(
        "calc_sensitivity(prob, COST) is deprecated. " *
        "Use calc_sensitivity(prob, :pg, :cq) or calc_sensitivity(prob, :lmp, :cl), etc.",
        :calc_sensitivity
    )
    return calc_sensitivity_cost(prob)
end

function calc_sensitivity(prob::DCOPFProblem, ::FlowLimitParameter)
    Base.depwarn(
        "calc_sensitivity(prob, FLOWLIMIT) is deprecated. " *
        "Use calc_sensitivity(prob, :va, :fmax), calc_sensitivity(prob, :lmp, :fmax), etc.",
        :calc_sensitivity
    )
    return calc_sensitivity_flowlimit(prob)
end

function calc_sensitivity(prob::DCOPFProblem, ::SusceptanceParameter)
    Base.depwarn(
        "calc_sensitivity(prob, SUSCEPTANCE) is deprecated. " *
        "Use calc_sensitivity(prob, :va, :b), calc_sensitivity(prob, :lmp, :b), etc.",
        :calc_sensitivity
    )
    return calc_sensitivity_susceptance(prob)
end

function calc_sensitivity(prob::DCOPFProblem, ::SwitchingParameter)
    Base.depwarn(
        "calc_sensitivity(prob, SWITCHING) is deprecated. " *
        "Use calc_sensitivity(prob, :va, :z), calc_sensitivity(prob, :lmp, :z), etc.",
        :calc_sensitivity
    )
    return calc_sensitivity_switching(prob)
end

# =============================================================================
# AC Power Flow: Symbol Dispatch
# =============================================================================

# Voltage magnitude sensitivities w.r.t. active power
_calc_sensitivity(state::ACPowerFlowState, ::Val{:vm}, ::Val{:p}) =
    calc_voltage_power_sensitivities(state).∂vm_∂p

# Voltage magnitude sensitivities w.r.t. reactive power
_calc_sensitivity(state::ACPowerFlowState, ::Val{:vm}, ::Val{:q}) =
    calc_voltage_power_sensitivities(state).∂vm_∂q

# Complex voltage sensitivities
_calc_sensitivity(state::ACPowerFlowState, ::Val{:v}, ::Val{:p}) =
    calc_voltage_power_sensitivities(state).∂v_∂p

_calc_sensitivity(state::ACPowerFlowState, ::Val{:v}, ::Val{:q}) =
    calc_voltage_power_sensitivities(state).∂v_∂q

# Current magnitude sensitivities
function _calc_sensitivity(state::ACPowerFlowState, ::Val{:im}, ::Val{:p})
    sens = calc_current_power_sensitivities(state)
    return sens.∂Im_∂p
end

function _calc_sensitivity(state::ACPowerFlowState, ::Val{:im}, ::Val{:q})
    sens = calc_current_power_sensitivities(state)
    return sens.∂Im_∂q
end

# =============================================================================
# AC Power Flow: Legacy Two-Argument API (Deprecated)
# =============================================================================

function calc_sensitivity(state::ACPowerFlowState, ::PowerInjectionParameter)
    Base.depwarn(
        "calc_sensitivity(state, POWER) is deprecated. " *
        "Use calc_sensitivity(state, :vm, :p) or calc_sensitivity(state, :vm, :q), etc.",
        :calc_sensitivity
    )
    calc_voltage_power_sensitivities(state)
end

function calc_sensitivity(state::ACPowerFlowState, ::TopologyParameter)
    error("TopologyParameter sensitivity for ACPowerFlowState requires network Dict. " *
          "Use voltage_topology_sensitivities(net::Dict) directly or provide branch_data.")
end

# =============================================================================
# Unified Voltage Sensitivity Interface (Deprecated)
# =============================================================================

"""
    calc_voltage_sensitivity(state, parameter) → voltage sensitivity matrix

DEPRECATED: Use the symbol-based API instead:
- `calc_sensitivity(state, :va, :d)` for DC phase angle sensitivity
- `calc_sensitivity(state, :vm, :p)` for AC voltage magnitude sensitivity

Unified interface for "voltage" sensitivities across DC and AC formulations.
"""
function calc_voltage_sensitivity end

# DC Power Flow (deprecated)
function calc_voltage_sensitivity(state::DCPowerFlowState, ::DemandParameter)
    Base.depwarn("calc_voltage_sensitivity is deprecated. Use calc_sensitivity(state, :va, :d)", :calc_voltage_sensitivity)
    calc_sensitivity(state, :va, :d)
end

function calc_voltage_sensitivity(state::DCPowerFlowState, ::SwitchingParameter)
    Base.depwarn("calc_voltage_sensitivity is deprecated. Use calc_sensitivity(state, :va, :z)", :calc_voltage_sensitivity)
    calc_sensitivity(state, :va, :z)
end

# DC OPF (deprecated)
function calc_voltage_sensitivity(prob::DCOPFProblem, ::DemandParameter)
    Base.depwarn("calc_voltage_sensitivity is deprecated. Use calc_sensitivity(prob, :va, :d)", :calc_voltage_sensitivity)
    calc_sensitivity(prob, :va, :d)
end

function calc_voltage_sensitivity(prob::DCOPFProblem, ::SwitchingParameter)
    Base.depwarn("calc_voltage_sensitivity is deprecated. Use calc_sensitivity(prob, :va, :z)", :calc_voltage_sensitivity)
    calc_sensitivity(prob, :va, :z)
end

# AC Power Flow (deprecated)
function calc_voltage_sensitivity(state::ACPowerFlowState, ::PowerInjectionParameter)
    Base.depwarn("calc_voltage_sensitivity is deprecated. Use calc_sensitivity(state, :vm, :p)", :calc_voltage_sensitivity)
    calc_voltage_power_sensitivities(state)
end
