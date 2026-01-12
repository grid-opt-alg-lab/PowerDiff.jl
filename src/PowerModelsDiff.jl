module PowerModelsDiff

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using JuMP
using Clarabel
using Ipopt
using ForwardDiff
using BlockArrays
import PowerModels
const PM = PowerModels

const MOI = JuMP.MOI

# =============================================================================
# Abstract type hierarchy and parameter types
# =============================================================================
include("types/abstract.jl")
include("types/parameters.jl")

# =============================================================================
# Core type definitions (modular structure)
# =============================================================================
include("types/dc_network.jl")      # DCNetwork, DCPowerFlowState, DCOPFSolution
include("types/ac_network.jl")      # ACNetwork, ACPowerFlowState
include("types/sensitivities.jl")   # All sensitivity result types

# =============================================================================
# Power flow and graph utilities
# =============================================================================
include("pf/admittance_matrix.jl")
include("graphs/laplacian.jl")
include("pf/bus_injection.jl")
include("pf/pf_eqns.jl")

# =============================================================================
# Deprecated types (for backwards compatibility)
# =============================================================================
include("deprecated/pf_structs.jl")
include("deprecated/measurements.jl")

# =============================================================================
# DC OPF (B-θ formulation)
# =============================================================================
include("opf/problem.jl")
include("opf/kkt.jl")

# =============================================================================
# Sensitivity analysis
# =============================================================================
include("sens/topology.jl")
include("sens/lmp.jl")
include("sens/demand.jl")
include("sens/cost.jl")
include("sens/flowlimit.jl")
include("sens/susceptance.jl")
include("sens/voltage.jl")
include("sens/current.jl")
include("sens/interface.jl")

# =============================================================================
# Exports
# =============================================================================

# -----------------------------------------------------------------------------
# Abstract Type Hierarchy
# -----------------------------------------------------------------------------
export AbstractPowerNetwork, AbstractPowerFlowState, AbstractOPFSolution
export AbstractOPFProblem
export AbstractSensitivity, AbstractSensitivityPower, AbstractSensitivityTopology

# -----------------------------------------------------------------------------
# Sensitivity Parameter Types (for unified dispatch)
# -----------------------------------------------------------------------------
export AbstractSensitivityParameter
export DemandParameter, GenerationParameter, CostParameter
export FlowLimitParameter, SusceptanceParameter, SwitchingParameter
export PowerInjectionParameter, TopologyParameter
export DEMAND, GENERATION, COST, FLOWLIMIT, SUSCEPTANCE, SWITCHING, POWER, TOPOLOGY

# -----------------------------------------------------------------------------
# DC Power Flow Types
# -----------------------------------------------------------------------------
export DCNetwork, DCPowerFlowState

# -----------------------------------------------------------------------------
# DC OPF Types and Functions
# -----------------------------------------------------------------------------
export DCOPFProblem, DCOPFSolution
export solve!, update_demand!
export calc_demand_vector, calc_susceptance_matrix

# DC Sensitivity Types
export DemandSensitivity, CostSensitivity, SwitchingSensitivity
export FlowLimitSensitivity, SusceptanceSensitivity

# DC Sensitivity Functions
export calc_sensitivity_demand, calc_sensitivity_demand_primal
export calc_generation_participation_factors, calc_ptdf_from_sensitivity
export calc_sensitivity_switching, update_switching!
export calc_sensitivity_cost
export calc_sensitivity_flowlimit, calc_sensitivity_susceptance

# LMP Functions
export calc_lmp, calc_congestion_component, calc_energy_component

# KKT Functions (advanced)
export kkt, kkt_dims, calc_kkt_jacobian
export flatten_variables, unflatten_variables

# -----------------------------------------------------------------------------
# AC Power Flow Types and Functions
# -----------------------------------------------------------------------------
export ACNetwork, ACPowerFlowState
export admittance_matrix, branch_current, branch_power

# AC Sensitivity Types
export VoltagePowerSensitivity, VoltageTopologySensitivity
export CurrentPowerSensitivity, CurrentTopologySensitivity

# AC Voltage Sensitivity Functions
export calc_voltage_power_sensitivities
export calc_voltage_active_power_sensitivities, calc_voltage_reactive_power_sensitivities
export voltage_topology_sensitivities

# AC Current Sensitivity Functions
export calc_current_power_sensitivities
export calc_current_magnitude_active_power_sensitivity, calc_current_magnitude_reactive_power_sensitivity

# -----------------------------------------------------------------------------
# Graph Utilities
# -----------------------------------------------------------------------------
export VectorizedAdmittanceMatrix, vectorize_laplacian_weights
export laplacian, full_incidence_matrix, calc_incidence_matrix

# Power Flow Equations
export NetworkTopology, PowerFlowEquations
export p, q, vm, vm2, pf_eqns
export p_polar, q_polar
export branch_flow, p_flow, q_flow

# -----------------------------------------------------------------------------
# Unified Sensitivity Interface
# -----------------------------------------------------------------------------
export calc_sensitivity, calc_voltage_sensitivity

end # module PowerModelsDiff
