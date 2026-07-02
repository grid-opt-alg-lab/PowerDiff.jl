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

# This experiment depends on the unregistered AcceleratedDCPowerFlows.jl.
# Install it in a local environment before running:
#   import Pkg; Pkg.add(url="https://github.com/mtanneau/AcceleratedDCPowerFlows.jl.git")

using PowerModels
using AcceleratedDCPowerFlows
using PowerDiff

const PM = PowerModels
const APF = AcceleratedDCPowerFlows

PM.silence()

function materialize_apf_ptdf(phi)
    ptdf = zeros(phi.E, phi.N)
    injection = zeros(phi.N)
    for i in 1:phi.N
        injection[i] = 1.0
        APF.compute_flow!(@view(ptdf[:, i]), injection, phi)
        injection[i] = 0.0
    end
    return ptdf
end

# --- Load network ---
case_path = joinpath(dirname(pathof(PM)), "..", "test", "data", "matpower", "case14.m")
pm_data = PM.parse_file(case_path)
PM.make_basic_network!(pm_data)

# Tighten flow limits so N-1 screening finds congested contingencies on case14.
for (_, br) in pm_data["branch"]
    br["rate_a"] *= 0.2
end

println("=== Joint APF + PowerDiff Workflow on case14 ===\n")

# --- Step 1: APF for N-1 contingency screening ---
apf_net = APF.from_power_models(pm_data)
N = APF.num_buses(apf_net)
E = APF.num_branches(apf_net)

L = APF.full_lodf(apf_net)

phi = APF.full_ptdf(apf_net)
p = [bus.pd for bus in apf_net.buses]
pf0 = zeros(E)
APF.compute_flow!(pf0, p, phi)

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
    println("    Branch $e: $(apf_net.branches[e].bus_fr) -> $(apf_net.branches[e].bus_to)")
end

# --- Step 2: PowerDiff for economic sensitivity ---
println("\nStep 2: DC OPF and sensitivity analysis via PowerDiff")
typed_data = parse_file(case_path)
dc_net = DCNetwork(typed_data)
dc_net.fmax .*= 0.2
d = calc_demand_vector(typed_data)
prob = DCOPFProblem(dc_net, d)
PowerDiff.solve!(prob)

dlmp_dsw = calc_sensitivity(prob, :lmp, :sw)

println("\n  LMP impact of critical contingencies:")
for e in critical
    lmp_grad = dlmp_dsw[:, e]
    max_impact_bus = argmax(abs.(lmp_grad))
    println("    Branch $e outage -> max LMP impact = $(round(lmp_grad[max_impact_bus], sigdigits=3)) at bus $max_impact_bus")
end

# --- Step 3: Cross-check PTDF matrices ---
println("\nStep 3: PTDF cross-check")
pf_state = DCPowerFlowState(dc_net, d)
pd_ptdf = ptdf_matrix(pf_state)
apf_ptdf = materialize_apf_ptdf(phi)
maxerr = maximum(abs, pd_ptdf - apf_ptdf)
println("  PowerDiff <-> APF PTDF max error: $(round(maxerr, sigdigits=3))")

# LMP and generation sensitivities on case14 are small because most generators
# sit at their upper bounds, which creates KKT degeneracy.

# --- Step 4: Generation sensitivity for topology optimization ---
println("\nStep 4: Generation sensitivity for topology optimization")
dpg_dsw = calc_sensitivity(prob, :pg, :sw)
println("  dpg/dsw matrix size: $(size(dpg_dsw))")
println("  Most sensitive generator branch pair:")
dpg_mat = Matrix(dpg_dsw)
idx = argmax(abs.(dpg_mat))
println("    Generator $(idx[1]), Branch $(idx[2]): $(round(dpg_mat[idx], sigdigits=3))")

println("\n=== Done ===")
