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
# Abstract type hierarchy and singleton tags
# =============================================================================
include("types/abstract.jl")
include("types/tags.jl")

# =============================================================================
# Core type definitions (modular structure)
# =============================================================================
include("types/dc_network.jl")      # DCNetwork, DCPowerFlowState, DCOPFSolution + constructors
include("types/dc_opf_problem.jl")  # DCOPFProblem + constructors
include("types/ac_network.jl")      # ACNetwork, ACPowerFlowState
include("types/ac_opf_problem.jl")  # ACOPFProblem, ACOPFSolution + constructors
include("types/sensitivities.jl")   # Sensitivity{F,O,P} and DC power flow bundled types

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
# DC OPF (B-theta formulation) - solving and KKT conditions
# =============================================================================
include("prob/dc_opf.jl")
include("prob/kkt_dc_opf.jl")

# =============================================================================
# AC OPF (Polar formulation) - solving and KKT conditions
# =============================================================================
include("prob/ac_opf_solve.jl")
include("prob/kkt_ac_opf.jl")

# =============================================================================
# Sensitivity analysis
# =============================================================================
include("sens/index_mapping.jl")
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
# Singleton Type Tags (for Sensitivity{F,O,P} dispatch)
# -----------------------------------------------------------------------------
export AbstractFormulation, AbstractOperand, AbstractParameter
# Formulation tags
export DCOPF, ACOPF, DCPF, ACPF
# Operand tags
export VoltageAngle, VoltageMagnitude, Generation, ReactiveGeneration
export Flow, LMP, CurrentMagnitude, Voltage
# Parameter tags
export Demand, Switching, QuadraticCost, LinearCost
export FlowLimit, Susceptance, ActivePower, ReactivePower

# -----------------------------------------------------------------------------
# Main Sensitivity Type
# -----------------------------------------------------------------------------
export Sensitivity
export formulation, operand, parameter

# DC Power Flow Bundled types (used by DCPowerFlowState)
export DCPFDemandSens, DCPFSwitchingSens

# -----------------------------------------------------------------------------
# DC Power Flow Types
# -----------------------------------------------------------------------------
export DCNetwork, DCPowerFlowState

# -----------------------------------------------------------------------------
# DC OPF Types and Functions
# -----------------------------------------------------------------------------
export DCOPFProblem, DCOPFSolution
export SensitivityCache, invalidate!
export solve!, update_demand!
export calc_demand_vector, calc_susceptance_matrix

# DC Sensitivity Functions (convenience wrappers)
export calc_generation_participation_factors, calc_ptdf_from_sensitivity
export update_switching!
# Note: calc_sensitivity_switching(prob::DCOPFProblem) removed - use calc_sensitivity(prob, :va, :z) instead

# LMP Functions
export calc_lmp, calc_congestion_component, calc_energy_component

# KKT Functions (advanced)
export kkt, kkt_dims, kkt_indices, calc_kkt_jacobian
export flatten_variables, unflatten_variables

# -----------------------------------------------------------------------------
# AC OPF Types and Functions
# -----------------------------------------------------------------------------
export ACOPFProblem, ACOPFSolution
export ac_kkt_dims, ac_kkt_indices, ac_flatten_variables, ac_unflatten_variables
export calc_ac_kkt_jacobian, ac_kkt

# -----------------------------------------------------------------------------
# AC Power Flow Types and Functions
# -----------------------------------------------------------------------------
export ACNetwork, ACPowerFlowState
export admittance_matrix, branch_current, branch_power

# AC Sensitivity Types (kept for backwards compatibility)
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
export calc_sensitivity

end # module PowerModelsDiff
