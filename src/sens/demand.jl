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

# Demand Sensitivity Analysis for DC OPF
# Uses implicit differentiation via KKT conditions
#
# Note: For DCOPFProblem, demand sensitivities are computed via the cached
# KKT system in kkt_dc_opf.jl. The functions here provide convenience wrappers.

"""
    calc_generation_participation_factors(prob::DCOPFProblem) → Sensitivity{Float64}

Compute generation participation factors from demand sensitivity.

The participation factor for generator i at bus j is dg_i/dd_j,
representing how much generator i output changes when demand at bus j increases by 1 MW.

# Returns
`Sensitivity{Float64}` (k × n) with formulation=:dcopf, operand=:pg, parameter=:d.
"""
function calc_generation_participation_factors(prob::DCOPFProblem)
    return calc_sensitivity(prob, :pg, :d)
end

"""
    calc_ptdf_from_sensitivity(prob::DCOPFProblem) → Sensitivity{Float64}

Compute Power Transfer Distribution Factors from flow sensitivity.

PTDF[e, j] = df_e/dd_j represents how much flow on line e changes
when power is injected at bus j (and withdrawn at the slack bus).

For the B-theta formulation, this is directly available from the sensitivity analysis.

# Returns
`Sensitivity{Float64}` (m × n) with formulation=:dcopf, operand=:f, parameter=:d.
"""
function calc_ptdf_from_sensitivity(prob::DCOPFProblem)
    return calc_sensitivity(prob, :f, :d)
end
