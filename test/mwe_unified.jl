# =============================================================================
# Minimum Working Example: Unified PowerModelsDiff Architecture
# =============================================================================
#
# This script demonstrates the unified type hierarchy and sensitivity API.

using PowerModelsDiff
using PowerModels

# Load a test network
case_path = joinpath(dirname(pathof(PowerModels)), "..", "test", "data", "matpower", "case5.m")
data = PowerModels.parse_file(case_path)
net_data = PowerModels.make_basic_network(data)

# =============================================================================
# DC Power Flow Example (non-OPF)
# =============================================================================
println("=" ^ 60)
println("=== DC Power Flow Sensitivities ===")
println("=" ^ 60)

# Create DC network and demand
net = DCNetwork(net_data)
d = calc_demand_vector(net_data)

println("Network: n=$(net.n) buses, m=$(net.m) branches, k=$(net.k) generators")

# Solve DC power flow (NOT OPF - just theta = L^+ * p)
pf_state = DCPowerFlowState(net, d)
println("Phase angles: theta = ", round.(pf_state.θ, digits=4))
println("Flows: f = ", round.(pf_state.f, digits=4))

# Switching sensitivity for power flow
sens_switching_pf = calc_sensitivity(pf_state, SWITCHING)
println("\nPF Switching sensitivity dtheta/dz (n x m):")
println("  Shape: ", size(sens_switching_pf.dθ_dz))
println("  [1,1] = ", round(sens_switching_pf.dθ_dz[1,1], digits=6))

# Demand sensitivity for power flow
sens_demand_pf = calc_sensitivity(pf_state, DEMAND)
println("\nPF Demand sensitivity dtheta/dd (n x n):")
println("  Shape: ", size(sens_demand_pf.dθ_dd))
println("  [1,1] = ", round(sens_demand_pf.dθ_dd[1,1], digits=6))

# =============================================================================
# DC OPF Example
# =============================================================================
println("\n" * "=" ^ 60)
println("=== DC OPF Sensitivities ===")
println("=" ^ 60)

# Solve OPF
prob = DCOPFProblem(net, d)
sol = solve!(prob)
println("OPF solved: objective = ", round(sol.objective, digits=2))
println("Generation: g = ", round.(sol.g, digits=4))

# Unified sensitivity API
sens_demand = calc_sensitivity(prob, DEMAND)
sens_cost = calc_sensitivity(prob, COST)
sens_switching_opf = calc_sensitivity(prob, SWITCHING)

println("\nOPF Demand sensitivity dtheta/dd:")
println("  Shape: ", size(sens_demand.dθ_dd))
println("  [1,1] = ", round(sens_demand.dθ_dd[1,1], digits=6))

println("\nOPF Cost sensitivity dg/dcq:")
println("  Shape: ", size(sens_cost.dg_dcq))
println("  [1,1] = ", round(sens_cost.dg_dcq[1,1], digits=6))

println("\nOPF Switching sensitivity dtheta/dz:")
println("  Shape: ", size(sens_switching_opf.dθ_dz))
println("  [1,1] = ", round(sens_switching_opf.dθ_dz[1,1], digits=6))

# =============================================================================
# AC Power Flow Example
# =============================================================================
println("\n" * "=" ^ 60)
println("=== AC Power Flow Sensitivities ===")
println("=" ^ 60)

# Solve AC power flow
PowerModels.compute_ac_pf!(net_data)

# Create ACNetwork and ACPowerFlowState
ac_net = ACNetwork(net_data)
state = ACPowerFlowState(net_data)

println("AC Network: n=$(ac_net.n) buses, m=$(ac_net.m) branches")
println("Voltage magnitudes: |v| = ", round.(abs.(state.v), digits=4))

# Voltage-power sensitivities
sens_power = calc_sensitivity(state, POWER)

println("\nAC voltage-power sensitivity d|v|/dp:")
println("  Shape: ", size(sens_power.∂vm_∂p))
if state.n >= 3
    println("  [2,3] = ", round(sens_power.∂vm_∂p[2,3], digits=6))
end

# =============================================================================
# Unified Voltage Sensitivity Interface
# =============================================================================
println("\n" * "=" ^ 60)
println("=== Unified Voltage Sensitivity Interface ===")
println("=" ^ 60)

# Both return "voltage" sensitivity, but different formulations
dθ_dd = calc_voltage_sensitivity(pf_state, DEMAND)
dθ_dz_pf = calc_voltage_sensitivity(pf_state, SWITCHING)
dθ_dd_opf = calc_voltage_sensitivity(prob, DEMAND)
dθ_dz_opf = calc_voltage_sensitivity(prob, SWITCHING)

println("\nDC Power Flow:")
println("  dtheta/dd shape = ", size(dθ_dd))
println("  dtheta/dz shape = ", size(dθ_dz_pf))

println("\nDC OPF:")
println("  dtheta/dd shape = ", size(dθ_dd_opf))
println("  dtheta/dz shape = ", size(dθ_dz_opf))

println("\nAC Power Flow:")
sens_ac = calc_voltage_sensitivity(state, POWER)
println("  d|v|/dp shape = ", size(sens_ac.∂vm_∂p))
println("  d|v|/dq shape = ", size(sens_ac.∂vm_∂q))

# =============================================================================
# Type Hierarchy Verification
# =============================================================================
println("\n" * "=" ^ 60)
println("=== Type Hierarchy Verification ===")
println("=" ^ 60)

println("DCNetwork <: AbstractPowerNetwork: ", net isa AbstractPowerNetwork)
println("ACNetwork <: AbstractPowerNetwork: ", ac_net isa AbstractPowerNetwork)
println("DCPowerFlowState <: AbstractPowerFlowState: ", pf_state isa AbstractPowerFlowState)
println("DCOPFSolution <: AbstractOPFSolution: ", sol isa AbstractOPFSolution)
println("ACPowerFlowState <: AbstractPowerFlowState: ", state isa AbstractPowerFlowState)

println("\nParameter singletons:")
println("  DEMAND isa DemandParameter: ", DEMAND isa DemandParameter)
println("  SWITCHING isa SwitchingParameter: ", SWITCHING isa SwitchingParameter)
println("  POWER isa PowerInjectionParameter: ", POWER isa PowerInjectionParameter)

println("\n" * "=" ^ 60)
println("MWE completed successfully!")
println("=" ^ 60)
