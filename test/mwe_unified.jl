# =============================================================================
# Minimum Working Example: Symbol-Based Sensitivity API
# =============================================================================
#
# Demonstrates the new symbol-based sensitivity API:
#   calc_sensitivity(state, :operand, :parameter) → Matrix

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
println("=== DC Power Flow: Symbol-Based Sensitivities ===")
println("=" ^ 60)

# Create DC network and demand
net = DCNetwork(net_data)
d = calc_demand_vector(net_data)

println("Network: n=$(net.n) buses, m=$(net.m) branches, k=$(net.k) generators")

# Solve DC power flow (NOT OPF - just theta = L^+ * p)
pf_state = DCPowerFlowState(net, d)
println("Phase angles: θ = ", round.(pf_state.θ, digits=4))
println("Flows: f = ", round.(pf_state.f, digits=4))

# New symbol-based API: request exactly what you need
println("\n--- Symbol-Based Sensitivity API ---")

dθ_dd = calc_sensitivity(pf_state, :θ, :d)
println("dθ/dd shape: ", size(dθ_dd), "  [1,1] = ", round(dθ_dd[1,1], digits=6))

df_dd = calc_sensitivity(pf_state, :f, :d)
println("df/dd shape: ", size(df_dd), "  [1,1] = ", round(df_dd[1,1], digits=6))

dθ_dz = calc_sensitivity(pf_state, :θ, :z)
println("dθ/dz shape: ", size(dθ_dz), "  [1,1] = ", round(dθ_dz[1,1], digits=6))

df_dz = calc_sensitivity(pf_state, :f, :z)
println("df/dz shape: ", size(df_dz), "  [1,1] = ", round(df_dz[1,1], digits=6))

# Aliases work too
dθ_dd_alias = calc_sensitivity(pf_state, :va, :pd)  # :va → :θ, :pd → :d
println("\nAlias test: calc_sensitivity(pf_state, :va, :pd) == calc_sensitivity(pf_state, :θ, :d): ",
        dθ_dd_alias ≈ dθ_dd)

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
println("Generation: g = ", round.(sol.g, digits=4))

# OPF has more operands available (including :lmp and :pg)
println("\n--- OPF Demand Sensitivities ---")
dθ_dd_opf = calc_sensitivity(prob, :θ, :d)
println("dθ/dd shape: ", size(dθ_dd_opf))

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
dθ_dz_opf = calc_sensitivity(prob, :θ, :z)
println("dθ/dz shape: ", size(dθ_dz_opf))

dlmp_dz = calc_sensitivity(prob, :lmp, :z)
println("dlmp/dz shape: ", size(dlmp_dz))

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
