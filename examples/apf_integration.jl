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
# Joint APF + PD Workflow: N-1 Screening + Sensitivity Analysis
# =============================================================================
#
# This example demonstrates using AcceleratedDCPowerFlows (APF) for fast
# contingency screening and PowerDiff (PD) for gradient-based
# economic sensitivity analysis. APF handles the speed-critical forward
# computation; PD provides the differentiable optimization layer.

using PowerModels
using AcceleratedDCPowerFlows
using PowerDiff

const PM = PowerModels
const APF = AcceleratedDCPowerFlows

PM.silence()

# --- Load network ---
case_path = joinpath(dirname(pathof(PM)), "..", "test", "data", "matpower", "case14.m")
pm_data = PM.parse_file(case_path)
PM.make_basic_network!(pm_data)

# Tighten flow limits to create congestion — case14's default limits are very
# loose, so N-1 screening finds zero critical contingencies.  Scaling by 0.2
# makes several post-contingency flows exceed 80% loading.
for (_, br) in pm_data["branch"]
    br["rate_a"] *= 0.2
end

println("=== Joint APF + PD Workflow on case14 ===\n")

# --- Step 1: APF for fast N-1 contingency screening ---
# Use APF.from_power_models (not PD's to_apf_network) because APF's converter
# preserves load/gen data in bus.pd, which is needed for compute_flow!.
# PD's to_apf_network zeros demand since PD separates demand from topology.
apf_net = APF.from_power_models(pm_data)
N = APF.num_buses(apf_net)
E = APF.num_branches(apf_net)

# Build LODF for O(1) per-contingency flow computation
L = APF.full_lodf(apf_net)

# Pre-contingency flows via PTDF
Φ = APF.full_ptdf(apf_net)
p = [bus.pd for bus in apf_net.buses]  # net injections (load - gen)
pf0 = zeros(E)
APF.compute_flow!(pf0, p, Φ)

# Screen all N-1 contingencies
println("Step 1: N-1 contingency screening via APF LODF")
pfc = zeros(E)
critical = Int[]
pmax = [br.pmax for br in apf_net.branches]
for e in 1:E
    br = apf_net.branches[e]
    APF.compute_flow!(pfc, pf0, L, br)
    max_loading = maximum(abs.(pfc) ./ pmax)
    if max_loading > 0.8
        push!(critical, e)
    end
end
println("  Found $(length(critical)) critical contingencies (>80% loading)")
for e in critical
    println("    Branch $e: $(apf_net.branches[e].bus_fr) → $(apf_net.branches[e].bus_to)")
end

# --- Step 2: PD for economic sensitivity ---
println("\nStep 2: DC OPF + sensitivity analysis via PD")
dc_net = DCNetwork(pm_data)
d = calc_demand_vector(pm_data)
prob = DCOPFProblem(dc_net, d)
PowerDiff.solve!(prob)

# LMP sensitivity w.r.t. switching
dlmp_dsw = calc_sensitivity(prob, :lmp, :sw)

println("\n  LMP impact of critical contingencies:")
for e in critical
    lmp_grad = dlmp_dsw[:, e]
    max_impact_bus = argmax(abs.(lmp_grad))
    println("    Branch $e outage → max LMP impact = $(round(lmp_grad[max_impact_bus], sigdigits=3)) at bus $max_impact_bus")
end

# --- Step 3: Cross-validate PTDF ---
println("\nStep 3: PTDF cross-validation")
pf_state = DCPowerFlowState(dc_net, d)
result = compare_ptdf(pf_state)
println("  PD ↔ APF PTDF match: $(result.match) (max error: $(round(result.maxerr, sigdigits=3)))")

# Note: LMP and generation sensitivities on case14 are small (~1e-5 to 1e-9)
# because 4/5 generators sit at their upper bounds, creating KKT degeneracy.
# A network with more interior generators would show larger, more meaningful values.

# --- Step 4: Generation sensitivity for topology optimization ---
println("\nStep 4: Generation sensitivity for topology optimization")
dpg_dsw = calc_sensitivity(prob, :pg, :sw)
println("  ∂pg/∂sw matrix size: $(size(dpg_dsw))")
println("  Most sensitive generator-branch pair:")
dpg_mat = Matrix(dpg_dsw)
idx = argmax(abs.(dpg_mat))
println("    Generator $(idx[1]), Branch $(idx[2]): $(round(dpg_mat[idx], sigdigits=3))")

println("\n=== Done ===")
