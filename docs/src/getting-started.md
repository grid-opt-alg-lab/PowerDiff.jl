# Getting Started

This guide walks through the main workflows: DC power flow, DC OPF with LMP analysis, AC power flow, and AC OPF.

## Setup

```julia
using PowerDiff
using PowerModels

# Load a MATPOWER case (make_basic_network is optional)
net = PowerModels.parse_file("case14.m")
# OR: net = PowerModels.make_basic_network(raw)  # sequential IDs
```

## Interactive Exploration

For a pre-loaded REPL session with case14, run:

```bash
julia --project=. -iL examples/interactive_repl.jl
```

This loads a `DCNetwork`, solves both DC power flow and DC OPF, and prints
suggested `calc_sensitivity` commands to try.

## DC Power Flow

DC power flow computes voltage angles from the reduced system ``\theta_r = B_r^{-1} p_r``, where ``B_r`` is the susceptance-weighted Laplacian with the reference bus row and column deleted.

```julia
dc_net = DCNetwork(net)
d = calc_demand_vector(net)
pf_state = DCPowerFlowState(dc_net, d)
```

Compute sensitivities with the symbol API:

```julia
dva_dd = calc_sensitivity(pf_state, :va, :d)   # dtheta/dd (n x n)
df_dsw  = calc_sensitivity(pf_state, :f, :sw)   # df/dsw (m x m)
dva_dsw = calc_sensitivity(pf_state, :va, :sw)  # dtheta/dsw (n x m)
df_dd  = calc_sensitivity(pf_state, :f, :d)    # df/dd (m x n)
dva_db = calc_sensitivity(pf_state, :va, :b)   # dtheta/db (n x m)
df_db  = calc_sensitivity(pf_state, :f, :b)    # df/db (m x m)
```

Each result is a [`Sensitivity{T}`](@ref) that acts like a matrix but carries metadata:

```julia
dva_dd.formulation  # :dcpf
dva_dd.operand      # :va
dva_dd.parameter    # :d
size(dva_dd)        # (n, n)
Matrix(dva_dd)      # extract raw matrix
```

## DC OPF

DC OPF solves the B-theta optimal power flow and provides access to LMPs through dual variables.

```julia
prob = DCOPFProblem(dc_net, d)
sol = solve!(prob)
```

### LMP Analysis

```julia
lmps = calc_lmp(sol, dc_net)
energy = calc_energy_component(sol, dc_net)
congestion = calc_congestion_component(sol, dc_net)
# lmps == energy .+ congestion
```

### OPF Sensitivities

DC OPF supports sensitivities for all five operands (`:va`, `:pg`, `:f`, `:psh`, `:lmp`) with respect to six parameters (`:d`, `:sw`, `:cq`, `:cl`, `:fmax`, `:b`):

```julia
dlmp_dd  = calc_sensitivity(prob, :lmp, :d)    # dLMP/dd (n x n)
dpg_dd   = calc_sensitivity(prob, :pg, :d)     # dpg/dd (k x n)
dpg_dcq  = calc_sensitivity(prob, :pg, :cq)    # dpg/dcq (k x k)
dlmp_dsw = calc_sensitivity(prob, :lmp, :sw)   # dLMP/dsw (n x m)
df_dfmax = calc_sensitivity(prob, :f, :fmax)   # df/dfmax (m x m)
dpsh_dd  = calc_sensitivity(prob, :psh, :d)    # dpsh/dd (n x n)
dpsh_dsw = calc_sensitivity(prob, :psh, :sw)   # dpsh/dsw (n x m)
```

### Using the `Sensitivity{T}` Result

```julia
sens = calc_sensitivity(prob, :lmp, :d)
sens.formulation          # :dcopf
sens.operand              # :lmp
sens.parameter            # :d
sens[2, 3]                # dLMP_2 / dd_3
sens.row_to_id[2]         # external bus ID for row 2
sens.id_to_row[14]        # internal row for bus 14
Matrix(sens)              # raw matrix
sens * ones(size(sens,2)) # matrix-vector product
```

## AC Power Flow

AC power flow sensitivities require a solved AC power flow solution.

```julia
PowerModels.compute_ac_pf!(net)
ac_state = ACPowerFlowState(net)

# Voltage and current sensitivities
dvm_dp = calc_sensitivity(ac_state, :vm, :p)   # d|V|/dp (n x n)
dvm_dq = calc_sensitivity(ac_state, :vm, :q)   # d|V|/dq (n x n)
dv_dp  = calc_sensitivity(ac_state, :v, :p)    # dV/dp (ComplexF64, n x n)
dim_dp = calc_sensitivity(ac_state, :im, :p)   # d|I|/dp (m x n)

# Voltage angle and branch flow sensitivities
dva_dp = calc_sensitivity(ac_state, :va, :p)   # dθ/dp (n x n)
df_dp  = calc_sensitivity(ac_state, :f, :p)    # dP_flow/dp (m x n)

# Demand sensitivities (∂/∂d = -∂/∂p since p_net = pg - pd)
dvm_dd = calc_sensitivity(ac_state, :vm, :d)   # d|V|/dd (n x n)
```

### Power Flow Jacobian

The 4 standard Jacobian blocks are available as sensitivity combinations:

```julia
J1 = calc_sensitivity(ac_state, :p, :va)   # ∂P/∂θ  (n x n)
J2 = calc_sensitivity(ac_state, :p, :vm)   # ∂P/∂|V| (n x n)
J3 = calc_sensitivity(ac_state, :q, :va)   # ∂Q/∂θ  (n x n)
J4 = calc_sensitivity(ac_state, :q, :vm)   # ∂Q/∂|V| (n x n)
```

Or compute all 4 at once:

```julia
jac = calc_power_flow_jacobian(ac_state)
jac.dp_dva   # J1
jac.dp_dvm   # J2
jac.dq_dva   # J3
jac.dq_dvm   # J4
```

## AC OPF

AC OPF computes sensitivities via implicit differentiation of the full nonlinear KKT system,
supporting 6 parameter types: switching, demand, reactive demand, costs, and flow limits.

```julia
ac_prob = ACOPFProblem(net)
solve!(ac_prob)

# Switching sensitivities
dvm_dsw = calc_sensitivity(ac_prob, :vm, :sw)   # d|V|/dsw (n x m)
dva_dsw = calc_sensitivity(ac_prob, :va, :sw)   # dva/dsw (n x m)
dpg_dsw = calc_sensitivity(ac_prob, :pg, :sw)   # dpg/dsw (k x m)
dqg_dsw = calc_sensitivity(ac_prob, :qg, :sw)   # dqg/dsw (k x m)

# Demand, cost, and flow limit sensitivities
dlmp_dd = calc_sensitivity(ac_prob, :lmp, :d)     # dLMP/dd (n x n)
dpg_dcq = calc_sensitivity(ac_prob, :pg, :cq)     # dpg/dcq (k x k)
dvm_dfmax = calc_sensitivity(ac_prob, :vm, :fmax)  # d|V|/dfmax (n x m)
```
