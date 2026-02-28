# Getting Started

This guide walks through the main workflows: DC power flow, DC OPF with LMP analysis, AC power flow, and AC OPF.

## Setup

```julia
using PowerModelsDiff
using PowerModels

# Load a MATPOWER case (must use make_basic_network)
raw = PowerModels.parse_file("case14.m")
net = PowerModels.make_basic_network(raw)
```

## DC Power Flow

DC power flow computes voltage angles from the linear approximation ``\theta = L^+ p``, where ``L`` is the susceptance-weighted Laplacian.

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
dpg_dd   = calc_sensitivity(prob, :pg, :d)     # dg/dd (k x n)
dpg_dcq  = calc_sensitivity(prob, :pg, :cq)    # dg/dcq (k x k)
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

dvm_dp = calc_sensitivity(ac_state, :vm, :p)   # d|V|/dp (n x n)
dvm_dq = calc_sensitivity(ac_state, :vm, :q)   # d|V|/dq (n x n)
dv_dp  = calc_sensitivity(ac_state, :v, :p)    # dV/dp (ComplexF64, n x n)
dim_dp = calc_sensitivity(ac_state, :im, :p)   # d|I|/dp (m x n)
```

## AC OPF

AC OPF computes switching sensitivities via implicit differentiation of the full nonlinear KKT system.

```julia
ac_prob = ACOPFProblem(net)
solve!(ac_prob)

dvm_dsw = calc_sensitivity(ac_prob, :vm, :sw)   # d|V|/dsw (n x m)
dva_dsw = calc_sensitivity(ac_prob, :va, :sw)   # dva/dsw (n x m)
dpg_dsw = calc_sensitivity(ac_prob, :pg, :sw)   # dpg/dsw (k x m)
dqg_dsw = calc_sensitivity(ac_prob, :qg, :sw)   # dqg/dsw (k x m)
```
