# =============================================================================
# Unified Sensitivity Interface
# =============================================================================
#
# Provides a single `calc_sensitivity(state, parameter)` function that dispatches
# to the appropriate sensitivity computation based on the state and parameter types.

"""
    calc_sensitivity(state, parameter) → AbstractSensitivity

Unified sensitivity computation with type dispatch.

Computes the sensitivity of a power flow or OPF solution with respect to
the specified parameter type.

# Arguments
- `state`: A power flow state or OPF solution (DCPowerFlowState, DCOPFSolution, ACPowerFlowState)
- `parameter`: A sensitivity parameter type (DEMAND, SWITCHING, POWER, etc.)

# Examples
```julia
# DC Power Flow sensitivities
pf_state = DCPowerFlowState(net, d)
sens = calc_sensitivity(pf_state, DEMAND)      # → DemandSensitivity
sens = calc_sensitivity(pf_state, SWITCHING)   # → SwitchingSensitivity

# DC OPF sensitivities
prob = DCOPFProblem(net, d)
sol = solve!(prob)
sens = calc_sensitivity(sol, DEMAND)           # → DemandSensitivity
sens = calc_sensitivity(sol, COST)             # → CostSensitivity
sens = calc_sensitivity(sol, FLOWLIMIT)        # → FlowLimitSensitivity
sens = calc_sensitivity(sol, SWITCHING)        # → SwitchingSensitivity

# AC power flow sensitivities
state = ACPowerFlowState(net)
sens = calc_sensitivity(state, POWER)          # → VoltagePowerSensitivity
sens = calc_sensitivity(state, TOPOLOGY)       # → VoltageTopologySensitivity
```
"""
function calc_sensitivity end

# =============================================================================
# DC Power Flow Dispatch (non-OPF)
# =============================================================================

calc_sensitivity(state::DCPowerFlowState, ::DemandParameter) =
    calc_sensitivity_demand(state)

calc_sensitivity(state::DCPowerFlowState, ::SwitchingParameter) =
    calc_sensitivity_switching(state)

# =============================================================================
# DC OPF Dispatch
# =============================================================================

# Note: DCOPFProblem stores the solution after solve!, so we dispatch on the problem
# For demand sensitivity, we use the problem-based function
function calc_sensitivity(prob::DCOPFProblem, ::DemandParameter)
    return calc_sensitivity_demand(prob)
end

function calc_sensitivity(prob::DCOPFProblem, ::CostParameter)
    return calc_sensitivity_cost(prob)
end

function calc_sensitivity(prob::DCOPFProblem, ::FlowLimitParameter)
    return calc_sensitivity_flowlimit(prob)
end

function calc_sensitivity(prob::DCOPFProblem, ::SusceptanceParameter)
    return calc_sensitivity_susceptance(prob)
end

function calc_sensitivity(prob::DCOPFProblem, ::SwitchingParameter)
    return calc_sensitivity_switching(prob)
end

# =============================================================================
# AC Power Flow Dispatch
# =============================================================================

calc_sensitivity(state::ACPowerFlowState, ::PowerInjectionParameter) =
    calc_voltage_power_sensitivities(state)

# For topology sensitivity, we need to convert ACPowerFlowState to Dict format
# or provide a direct implementation
function calc_sensitivity(state::ACPowerFlowState, ::TopologyParameter)
    # Use the existing voltage_topology_sensitivities with appropriate conversion
    error("TopologyParameter sensitivity for ACPowerFlowState requires network Dict. " *
          "Use voltage_topology_sensitivities(net::Dict) directly or provide branch_data.")
end

# =============================================================================
# Unified Voltage Sensitivity Interface
# =============================================================================

"""
    calc_voltage_sensitivity(state, parameter) → voltage sensitivity matrix

Unified interface for "voltage" sensitivities across DC and AC formulations.

Returns the sensitivity of voltage-like quantities:
- DC formulations: ∂θ/∂parameter (phase angles)
- AC formulations: ∂v/∂parameter (complex voltages or magnitudes)

# Examples
```julia
# DC Power Flow: ∂θ/∂d
dθ_dd = calc_voltage_sensitivity(pf_state, DEMAND)

# DC OPF: ∂θ/∂z
dθ_dz = calc_voltage_sensitivity(prob, SWITCHING)

# AC Power Flow: ∂v/∂p, ∂|v|/∂p, etc.
sens = calc_voltage_sensitivity(ac_state, POWER)
```
"""
function calc_voltage_sensitivity end

# DC Power Flow
calc_voltage_sensitivity(state::DCPowerFlowState, ::DemandParameter) =
    calc_sensitivity_demand(state).dθ_dd

calc_voltage_sensitivity(state::DCPowerFlowState, ::SwitchingParameter) =
    calc_sensitivity_switching(state).dθ_dz

# DC OPF
calc_voltage_sensitivity(prob::DCOPFProblem, ::DemandParameter) =
    calc_sensitivity_demand(prob).dθ_dd

calc_voltage_sensitivity(prob::DCOPFProblem, ::SwitchingParameter) =
    calc_sensitivity_switching(prob).dθ_dz

# AC Power Flow
calc_voltage_sensitivity(state::ACPowerFlowState, ::PowerInjectionParameter) =
    calc_voltage_power_sensitivities(state)
