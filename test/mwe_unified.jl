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
# Minimum Working Example: Symbol-Based Sensitivity API
# =============================================================================
#
# Demonstrates the new symbol-based sensitivity API:
#   calc_sensitivity(state, :operand, :parameter) → Matrix

using PowerModelsDiff
using PowerModels

# Load a test network
case_path = joinpath(dirname(pathof(PowerModels)), "..", "test", "data", "matpower", "case14.m")
data = PowerModels.parse_file(case_path)
net_data = PowerModels.make_basic_network(data)

# =============================================================================
# DC Power Flow Example (non-OPF)
# =============================================================================
println("=" ^ 60)
println("=== DC Power Flow: Symbol-Based Sensitivities ===")
println("=" ^ 60)

# Create DC network and demand
net = DCNetwork(net_data)
d = calc_demand_vector(net_data)

println("Network: n=$(net.n) buses, m=$(net.m) branches, k=$(net.k) generators")

# Solve DC power flow (NOT OPF - just theta = L^+ * p)
pf_state = DCPowerFlowState(net, d)
println("Phase angles: θ = ", round.(pf_state.va, digits=4))
println("Flows: f = ", round.(pf_state.f, digits=4))

# New symbol-based API: request exactly what you need
println("\n--- Symbol-Based Sensitivity API ---")

dva_dd = calc_sensitivity(pf_state, :va, :d)
println("dva/dd shape: ", size(dva_dd), "  [1,1] = ", round(dva_dd[1,1], digits=6))

df_dd = calc_sensitivity(pf_state, :f, :d)
println("df/dd shape: ", size(df_dd), "  [1,1] = ", round(df_dd[1,1], digits=6))

dva_dsw = calc_sensitivity(pf_state, :va, :sw)
println("dva/dsw shape: ", size(dva_dsw), "  [1,1] = ", round(dva_dsw[1,1], digits=6))

df_dsw = calc_sensitivity(pf_state, :f, :sw)
println("df/dsw shape: ", size(df_dsw), "  [1,1] = ", round(df_dsw[1,1], digits=6))

# Aliases work too
dva_dd_alias = calc_sensitivity(pf_state, :va, :pd)  # :pd → :d
println("\nAlias test: calc_sensitivity(pf_state, :va, :pd) == calc_sensitivity(pf_state, :va, :d): ",
        dva_dd_alias ≈ dva_dd)

# Invalid combinations throw ArgumentError
print("\nInvalid combination test: calc_sensitivity(pf_state, :lmp, :d) → ")
try
    calc_sensitivity(pf_state, :lmp, :d)
    println("ERROR: should have thrown!")
catch e
    println("ArgumentError (as expected)")
end

# =============================================================================
# DC OPF Example
# =============================================================================
println("\n" * "=" ^ 60)
println("=== DC OPF: Symbol-Based Sensitivities ===")
println("=" ^ 60)

# Solve OPF
prob = DCOPFProblem(net, d)
sol = solve!(prob)
println("OPF solved: objective = ", round(sol.objective, digits=2))
println("Generation: g = ", round.(sol.pg, digits=4))

# OPF has more operands available (including :lmp and :pg)
println("\n--- OPF Demand Sensitivities ---")
dva_dd_opf = calc_sensitivity(prob, :va, :d)
println("dva/dd shape: ", size(dva_dd_opf))

dpg_dd = calc_sensitivity(prob, :pg, :d)
println("dpg/dd shape: ", size(dpg_dd))

df_dd_opf = calc_sensitivity(prob, :f, :d)
println("df/dd shape: ", size(df_dd_opf))

dlmp_dd = calc_sensitivity(prob, :lmp, :d)
println("dlmp/dd shape: ", size(dlmp_dd))

println("\n--- OPF Cost Sensitivities ---")
dpg_dcq = calc_sensitivity(prob, :pg, :cq)
println("dpg/dcq shape: ", size(dpg_dcq), "  [1,1] = ", round(dpg_dcq[1,1], digits=6))

dlmp_dcl = calc_sensitivity(prob, :lmp, :cl)
println("dlmp/dcl shape: ", size(dlmp_dcl))

println("\n--- OPF Switching Sensitivities ---")
dva_dsw_opf = calc_sensitivity(prob, :va, :sw)
println("dva/dsw shape: ", size(dva_dsw_opf))

dlmp_dsw = calc_sensitivity(prob, :lmp, :sw)
println("dlmp/dsw shape: ", size(dlmp_dsw))

# =============================================================================
# AC Power Flow Example
# =============================================================================
println("\n" * "=" ^ 60)
println("=== AC Power Flow: Symbol-Based Sensitivities ===")
println("=" ^ 60)

# Solve AC power flow
PowerModels.compute_ac_pf!(net_data)

# Create ACNetwork and ACPowerFlowState
ac_net = ACNetwork(net_data)
state = ACPowerFlowState(net_data)

println("AC Network: n=$(ac_net.n) buses, m=$(ac_net.m) branches")
println("Voltage magnitudes: |v| = ", round.(abs.(state.v), digits=4))

# AC sensitivities
println("\n--- AC Voltage-Power Sensitivities ---")
dvm_dp = calc_sensitivity(state, :vm, :p)
println("d|v|/dp shape: ", size(dvm_dp))
if state.n >= 3
    println("  [2,3] = ", round(dvm_dp[2,3], digits=6))
end

dvm_dq = calc_sensitivity(state, :vm, :q)
println("d|v|/dq shape: ", size(dvm_dq))

# Current sensitivities
dim_dp = calc_sensitivity(state, :im, :p)
println("d|I|/dp shape: ", size(dim_dp))

# =============================================================================
# Type Hierarchy Verification
# =============================================================================
println("\n" * "=" ^ 60)
println("=== Type Hierarchy ===")
println("=" ^ 60)

println("DCNetwork <: AbstractPowerNetwork: ", net isa AbstractPowerNetwork)
println("ACNetwork <: AbstractPowerNetwork: ", ac_net isa AbstractPowerNetwork)
println("DCPowerFlowState <: AbstractPowerFlowState: ", pf_state isa AbstractPowerFlowState)
println("DCOPFSolution <: AbstractOPFSolution: ", sol isa AbstractOPFSolution)
println("ACPowerFlowState <: AbstractPowerFlowState: ", state isa AbstractPowerFlowState)

println("\n" * "=" ^ 60)
println("MWE completed successfully!")
println("=" ^ 60)
