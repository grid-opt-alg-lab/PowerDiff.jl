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

module PowerDiff

using LinearAlgebra
using SparseArrays
using JuMP
using Ipopt
using ForwardDiff
import PowerModels
const PM = PowerModels

const MOI = JuMP.MOI

# =============================================================================
# Warning suppression
# =============================================================================
const _SILENCE_WARNINGS = Ref(false)

"""
    silence()

Suppress all warning messages from PowerDiff for the rest of the session.

Warnings from other packages (PowerModels, JuMP, Ipopt, etc.) are not affected.
To suppress PowerModels output, also call `PowerModels.silence()`.
"""
function silence()
    _SILENCE_WARNINGS[] = true
    return nothing
end

# =============================================================================
# Abstract type hierarchy
# =============================================================================
include("types/abstract.jl")

# =============================================================================
# Core type definitions (modular structure)
# =============================================================================
include("types/id_mapping.jl")      # IDMapping (must come before network types)
include("types/dc_network.jl")      # DCNetwork, DCPowerFlowState, DCOPFSolution + constructors
include("types/dc_opf_problem.jl")  # DCOPFProblem + constructors
include("types/ac_network.jl")      # ACNetwork, ACPowerFlowState
include("types/ac_opf_problem.jl")  # ACOPFProblem, ACOPFSolution + constructors
include("types/sensitivities.jl")   # Sensitivity{T} (public API wrapper)
include("types/show.jl")           # Pretty-printing (Base.show methods)

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
include("sens/topology_ac.jl")
include("sens/current.jl")
include("sens/jacobian.jl")
include("sens/interface.jl")
include("sens/vjp_jvp.jl")

# =============================================================================
# Exports — clean public API
# =============================================================================

# Abstract Type Hierarchy
export AbstractPowerNetwork, AbstractPowerFlowState, AbstractOPFSolution
export AbstractOPFProblem
export IDMapping

# Sensitivity Interface
export calc_sensitivity, calc_sensitivity_column
export Sensitivity, silence
export operand_symbols, parameter_symbols
export jvp, vjp, jvp!, vjp!, dict_to_vec, vec_to_dict, kkt_dims

# DC Power Flow Types
export DCNetwork, DCPowerFlowState

# DC OPF Types and Functions
export DCOPFProblem, DCOPFSolution
export DCSensitivityCache, invalidate!
export solve!, update_demand!
export calc_demand_vector, calc_susceptance_matrix

# DC Sensitivity Functions (convenience wrappers)
export calc_generation_participation_factors, calc_ptdf_from_sensitivity
export update_switching!

# LMP Functions
export calc_lmp, calc_qlmp, calc_congestion_component, calc_energy_component

# AC OPF Types and Functions
export ACOPFProblem, ACOPFSolution, ACSensitivityCache

# AC Power Flow Types and Functions
export ACNetwork, ACPowerFlowState
export admittance_matrix, branch_current, branch_power
export calc_power_flow_jacobian

# =============================================================================
# PTDF convenience (core, no APF dependency)
# =============================================================================

"""
    ptdf_matrix(state::DCPowerFlowState) → Matrix{Float64}

Return the standard PTDF matrix (`∂f/∂p`) from a DC power flow state.

PD's `calc_sensitivity(state, :f, :d)` computes `∂f/∂d = -PTDF` because
`p = g - d ⟹ ∂p/∂d = -I`. This function negates to recover the standard
PTDF sign convention: `PTDF = ∂f/∂p`.
"""
ptdf_matrix(state::DCPowerFlowState) = -Matrix(calc_sensitivity(state, :f, :d))

export ptdf_matrix

# =============================================================================
# APF extension stubs — implemented in ext/PowerDiffAPFExt.jl
# =============================================================================

const _APF_HINT = "Load AcceleratedDCPowerFlows first: `using AcceleratedDCPowerFlows`"

function to_apf_network end
function apf_ptdf end
function apf_lodf end
function compare_ptdf end
function materialize_apf_ptdf end

# Fallback methods with informative error when APF is not loaded
for fn in (:to_apf_network, :apf_ptdf, :apf_lodf, :compare_ptdf, :materialize_apf_ptdf)
    @eval $fn(args...; kwargs...) = error("$($fn) requires AcceleratedDCPowerFlows. " * _APF_HINT)
end

export to_apf_network, apf_ptdf, apf_lodf, compare_ptdf, materialize_apf_ptdf

end # module PowerDiff
