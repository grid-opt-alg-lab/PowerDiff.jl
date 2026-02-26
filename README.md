# PowerModelsDiff.jl

A Julia package for differentiable power system analysis. Provides automatic differentiation capabilities for the power flow equations, optimal power flow (OPF) problems, and sensitivity analysis of power networks.

## Features

- **Unified type hierarchy**: Consistent API for DC OPF, DC power flow, and AC power flow
- **DC OPF (B-theta formulation)**: Susceptance-weighted Laplacian that preserves graphical structure
- **DC power flow sensitivities**: Switching and demand sensitivity for non-OPF power flow
- **Implicit differentiation**: Compute OPF sensitivities via KKT conditions
- **LMP computation**: Locational Marginal Prices from dual variables with decomposition
- **AC power flow sensitivities**: Voltage and current sensitivity w.r.t. power and topology
- **ForwardDiff verification**: All sensitivities verified against automatic differentiation

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/samtalki/PowerModelsDiff.jl")
```

## Quick Start

```julia
using PowerModelsDiff
using PowerModels

# Load a MATPOWER case
raw = PowerModels.parse_file("case14.m")
net = PowerModels.make_basic_network(raw)

# =============================================================================
# DC Power Flow (non-OPF)
# =============================================================================
dc_net = DCNetwork(net)
d = calc_demand_vector(net)
pf_state = DCPowerFlowState(dc_net, d)

# Symbol-based sensitivity API: calc_sensitivity(state, :operand, :parameter)
dva_dd = calc_sensitivity(pf_state, :va, :d)   # dθ/dd (n×n)
df_dz = calc_sensitivity(pf_state, :f, :z)     # df/dz (m×m)

# =============================================================================
# DC OPF
# =============================================================================
prob = DCOPFProblem(dc_net, d)
sol = solve!(prob)

# LMPs and decomposition
lmps = calc_lmp(sol, dc_net)
energy = calc_energy_component(sol, dc_net)
congestion = calc_congestion_component(sol, dc_net)

# OPF sensitivities (includes LMP because it has duals)
dlmp_dd = calc_sensitivity(prob, :lmp, :d)     # dLMP/dd (n×n)
dpg_dd = calc_sensitivity(prob, :pg, :d)       # dg/dd (k×n)
dpg_dcq = calc_sensitivity(prob, :pg, :cq)     # dg/dcq (k×k)
dlmp_dz = calc_sensitivity(prob, :lmp, :z)     # dLMP/dz (n×m)

# =============================================================================
# AC Power Flow
# =============================================================================
PowerModels.compute_ac_pf!(net)
ac_state = ACPowerFlowState(net)

# Voltage-power sensitivities
dvm_dp = calc_sensitivity(ac_state, :vm, :p)   # d|V|/dp (n×n)
dvm_dq = calc_sensitivity(ac_state, :vm, :q)   # d|V|/dq (n×n)

# =============================================================================
# AC OPF
# =============================================================================
ac_prob = ACOPFProblem(net)
solve!(ac_prob)

# Switching sensitivities via implicit differentiation
dvm_dz = calc_sensitivity(ac_prob, :vm, :z)    # d|V|/dz (n×m)
dpg_dz = calc_sensitivity(ac_prob, :pg, :z)    # dpg/dz (k×m)
```

## Type Hierarchy

```
AbstractPowerNetwork
├── DCNetwork           # DC B-theta formulation
└── ACNetwork           # AC with vectorized admittance

AbstractPowerFlowState
├── DCPowerFlowState    # DC power flow (theta = L+ * p)
├── ACPowerFlowState    # AC power flow (complex voltages)
└── AbstractOPFSolution
    ├── DCOPFSolution   # DC OPF with duals
    └── ACOPFSolution   # AC OPF with duals

AbstractOPFProblem
├── DCOPFProblem        # JuMP-based DC OPF
└── ACOPFProblem        # JuMP-based AC OPF (Ipopt)
```

## Sensitivity API

The symbol-based API returns exactly the Jacobian you need:

```julia
calc_sensitivity(state, :operand, :parameter) → Matrix
```

**Operand symbols** (what we differentiate):
| Symbol | Description | Works with |
|--------|-------------|------------|
| `:va` | Voltage phase angles | DCPowerFlowState, DCOPFProblem, ACOPFProblem |
| `:f` | Branch flows | DCPowerFlowState, DCOPFProblem |
| `:pg` / `:g` | Generator active power | DCOPFProblem, ACOPFProblem |
| `:qg` | Generator reactive power | ACOPFProblem |
| `:lmp` | Locational marginal prices | DCOPFProblem |
| `:vm` | Voltage magnitude | ACPowerFlowState, ACOPFProblem |
| `:im` | Current magnitude | ACPowerFlowState |

**Parameter symbols** (what we differentiate w.r.t.):
| Symbol | Description | Works with |
|--------|-------------|------------|
| `:d` / `:pd` | Demand | DCPowerFlowState, DCOPFProblem |
| `:z` | Switching states | DCPowerFlowState, DCOPFProblem, ACOPFProblem |
| `:cq`, `:cl` | Cost coefficients | DCOPFProblem |
| `:fmax` | Flow limits | DCOPFProblem |
| `:b` | Susceptances | DCOPFProblem |
| `:p`, `:q` | Power injections | ACPowerFlowState |

Invalid operand/parameter combinations throw `ArgumentError`.

## Matrix Indexing

All sensitivity matrices use **sequential 1-based indexing** matching PowerModels string keys:

```
S[i,j] = ∂(operand element i) / ∂(parameter element j)
```

- **Buses** `1:n` → `net["bus"]["1"]` through `net["bus"]["$n"]`
- **Branches** `1:m` → `net["branch"]["1"]` through `net["branch"]["$m"]`
- **Generators** `1:k` → `net["gen"]["1"]` through `net["gen"]["$k"]`

To find which bus a generator is at: `net["gen"]["$i"]["gen_bus"]`
To find branch endpoints: `net["branch"]["$j"]["f_bus"]` and `net["branch"]["$j"]["t_bus"]`

The `Sensitivity{F,O,P}` return type also carries bidirectional index mappings:

```julia
sens = calc_sensitivity(prob, :lmp, :d)
sens[2, 3]                # ∂(LMP at bus 2) / ∂(demand at bus 3)
sens.row_to_id[2]         # External bus ID for row 2
sens.id_to_row[14]        # Internal row index for bus 14
Matrix(sens)              # Extract raw matrix
```

## Core Types

### DCNetwork
```julia
struct DCNetwork <: AbstractPowerNetwork
    n, m, k       # Buses, branches, generators
    A             # Incidence matrix (m x n)
    G_inc         # Generator-bus incidence (n x k)
    b             # Susceptances
    z             # Switching states [0,1]
    fmax, gmax, gmin  # Limits
    cq, cl        # Cost coefficients
    ref_bus       # Reference bus
    tau           # Regularization
end
```

### ACNetwork
```julia
struct ACNetwork <: AbstractPowerNetwork
    n, m          # Buses, branches
    A             # Incidence matrix
    incidences    # Edge list [(i,j), ...]
    g, b          # Conductances, susceptances
    g_shunt, b_shunt  # Shunt admittances
    z             # Switching states
    # ... limits
end
```

### Sensitivity Results

The API returns `Sensitivity{F,O,P}` which acts as a matrix with type tags for dispatch:

```julia
sens = calc_sensitivity(prob, :lmp, :d)
typeof(sens)  # Sensitivity{DCOPF, LMP, Demand}
size(sens)    # (n, n)
sens * v      # Matrix operations work

# Type-based dispatch
process(s::Sensitivity{DCOPF, LMP, Demand}) = "DC OPF LMP-demand"
process(s::Sensitivity{F, O, Switching}) where {F, O} = "Any switching"
```

Bundled sensitivity types for internal use:

```julia
DCPFDemandSens         # dva_dd, df_dd
DCPFSwitchingSens      # dva_dz, df_dz
ACOPFSwitchingSens     # dvm_dz, dva_dz, dpg_dz, dqg_dz

# AC power flow (legacy)
VoltagePowerSensitivity    # dvm_dp, dvm_dq
CurrentPowerSensitivity    # dim_dp, dim_dq
```

## KKT System (Advanced)

```julia
# Flatten/unflatten primal-dual variables
z = flatten_variables(sol, prob)
vars = unflatten_variables(z, prob)

# Evaluate KKT residuals
K = kkt(z, prob, d)

# Compute analytical KKT Jacobian
J = calc_kkt_jacobian(prob)

# Sensitivities: dz/dp = -(dK/dz)^{-1} * (dK/dp)
```

## Mathematical Background

### B-theta Formulation
```
min  (1/2) g' Cq g + cl' g + (tau/2) ||f||^2
s.t. G_inc * g - d = L * theta     (power balance)
     f = W * A * theta              (flow definition)
     |f| <= fmax                    (flow limits)
     gmin <= g <= gmax              (gen limits)
     theta[ref] = 0                 (reference)
```

where L = A' * Diag(-b .* z) * A is the susceptance-weighted Laplacian.

### DC Power Flow
For non-OPF power flow: theta = L+ * p where p = g - d.

### AC Admittance
Y = A' * Diag(g + j*b) * A + Diag(g_shunt + j*b_shunt)

## Solver Support

Default: Clarabel (interior-point conic). Also supported: Ipopt, HiGHS, Gurobi.

```julia
using Ipopt
prob = DCOPFProblem(dc_net, d; optimizer=Ipopt.Optimizer)
```

## Dependencies

- [PowerModels.jl](https://github.com/lanl-ansi/PowerModels.jl): Power system modeling
- [JuMP.jl](https://github.com/jump-dev/JuMP.jl): Optimization modeling
- [Clarabel.jl](https://github.com/oxfordcontrol/Clarabel.jl): Default interior-point solver
- [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl): Automatic differentiation

## License

MIT License
