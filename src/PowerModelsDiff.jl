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
# Core type definitions
# =============================================================================
include("structs.jl")

# =============================================================================
# Power flow and graph utilities
# =============================================================================
include("pf/admittance_matrix.jl")
include("graphs/laplacian.jl")
include("pf/bus_injection.jl")
include("pf/pf_eqns.jl")

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

# =============================================================================
# Exports
# =============================================================================

# DC OPF Types
export DCNetwork, DCOPFProblem, DCOPFSolution

# OPF Functions
export solve!, update_demand!
export calc_demand_vector, calc_susceptance_matrix

# LMP Functions
export calc_lmp, calc_lmp_from_duals
export calc_congestion_component

# Sensitivity Types
export DemandSensitivity, CostSensitivity, TopologySensitivity

# Sensitivity Functions
export calc_sensitivity_demand, calc_sensitivity_demand_primal
export calc_generation_participation_factors, calc_ptdf_from_sensitivity
export calc_sensitivity_switching, update_switching!

# KKT Functions (advanced)
export kkt, kkt_dims, calc_kkt_jacobian
export flatten_variables, unflatten_variables

# Existing exports (AC power flow sensitivities)
export VoltageSensitivityTopology, voltage_topology_sensitivities

# Graph utilities
export VectorizedAdmittanceMatrix, vectorize_laplacian_weights
export laplacian, full_incidence_matrix, calc_incidence_matrix

end # module PowerModelsDiff
