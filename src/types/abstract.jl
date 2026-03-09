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
# Abstract Type Hierarchy for PowerDiff
# =============================================================================
#
# Provides a unified type structure for multiple dispatch across:
# - DC power flow and DC OPF
# - AC power flow and AC OPF
# - Sensitivity analysis with parameter-based dispatch

# =============================================================================
# Level 1: Abstract Network (static data container)
# =============================================================================

"""
    AbstractPowerNetwork

Abstract base type for power network data containers.

Concrete subtypes:
- `DCNetwork`: DC network with susceptance-weighted Laplacian
- `ACNetwork`: AC network with complex admittance matrix
"""
abstract type AbstractPowerNetwork end

# =============================================================================
# Level 2: Abstract State (solved power flow or OPF solution)
# =============================================================================

"""
    AbstractPowerFlowState

Abstract base type for power flow solutions (DC or AC).

Concrete subtypes:
- `DCPowerFlowState`: DC power flow solution (θ_r = L_r \\ p_r)
- `ACPowerFlowState`: AC power flow solution (complex voltages)
- `AbstractOPFSolution` and its subtypes
"""
abstract type AbstractPowerFlowState end

"""
    AbstractOPFSolution <: AbstractPowerFlowState

Abstract type for optimal power flow solutions with dual variables.

Concrete subtypes:
- `DCOPFSolution`: DC OPF with generation dispatch and duals
- `ACOPFSolution`: AC OPF with voltages, generation, and duals
"""
abstract type AbstractOPFSolution <: AbstractPowerFlowState end

# =============================================================================
# Level 3: Abstract Problem (optimization wrapper)
# =============================================================================

"""
    AbstractOPFProblem

Abstract base type for OPF problem wrappers (JuMP models).

Concrete subtypes:
- `DCOPFProblem`: DC OPF problem (B-θ formulation)
- `ACOPFProblem`: AC OPF problem (polar formulation)
"""
abstract type AbstractOPFProblem end
