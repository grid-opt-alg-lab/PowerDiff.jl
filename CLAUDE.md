# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PowerModelsDiff is a Julia package for differentiable power system analysis. It provides automatic differentiation capabilities for power flow equations, optimal power flow (OPF) problems, and sensitivity analysis of power networks. The package builds on PowerModels.jl.

## Development Commands

```bash
# Activate the project environment
julia --project=.

# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Load the module in REPL
julia --project=.
julia> using PowerModelsDiff
```

## Architecture

### Unified Type Hierarchy

The package uses a unified type hierarchy enabling multiple dispatch for sensitivity computations:

```
AbstractPowerNetwork
├── DCNetwork           # DC B-theta formulation
└── ACNetwork           # AC with vectorized admittance

AbstractPowerFlowState
├── DCPowerFlowState    # DC power flow solution (theta = L+ * p)
├── ACPowerFlowState    # AC power flow solution (complex voltages)
└── AbstractOPFSolution
    └── DCOPFSolution   # DC OPF with generation, flows, duals

AbstractOPFProblem
└── DCOPFProblem        # JuMP-based DC OPF wrapper

AbstractSensitivityParameter
├── DemandParameter, CostParameter, FlowLimitParameter
├── SusceptanceParameter, SwitchingParameter
├── PowerInjectionParameter, TopologyParameter
```

### Unified Sensitivity API

```julia
# Parameter singletons for dispatch
DEMAND, COST, FLOWLIMIT, SUSCEPTANCE, SWITCHING, POWER, TOPOLOGY

# Unified interface
calc_sensitivity(state, parameter) -> AbstractSensitivity

# Examples:
sens = calc_sensitivity(pf_state, SWITCHING)   # DC PF switching
sens = calc_sensitivity(prob, DEMAND)          # DC OPF demand
sens = calc_sensitivity(ac_state, POWER)       # AC voltage-power
```

### DC OPF - B-theta Formulation

Uses susceptance-weighted Laplacian `L = A' * Diagonal(-b .* z) * A`:

- `DCNetwork`: Network data (topology `A`, susceptances `b`, switching `z`, limits, costs)
- `DCOPFProblem`: JuMP optimization wrapper
- `DCOPFSolution`: Primal (theta, g, f) and dual variables
- `DCPowerFlowState`: Non-OPF power flow (theta = L+ * p)

### AC Power Flow

- `ACNetwork`: Vectorized admittance (g, b vectors, incidence A, shunts)
- `ACPowerFlowState`: Complex voltages, injections, with optional ACNetwork reference
- Power flow equations as functions: `p(net, v)`, `q(net, v)`, `admittance_matrix(net)`

### Sensitivity Analysis (src/sens/)

**DC Sensitivities** (via KKT implicit differentiation):
- `calc_sensitivity_demand(prob)` -> `DemandSensitivity`
- `calc_sensitivity_switching(prob)` -> `SwitchingSensitivity`
- `calc_sensitivity_cost(prob)` -> `CostSensitivity`
- `calc_sensitivity_flowlimit(prob)` -> `FlowLimitSensitivity`
- `calc_sensitivity_susceptance(prob)` -> `SusceptanceSensitivity`

**DC Power Flow Sensitivities** (via Laplacian derivative):
- `calc_sensitivity_switching(state::DCPowerFlowState)` -> `SwitchingSensitivity`
- `calc_sensitivity_demand(state::DCPowerFlowState)` -> `DemandSensitivity`

**AC Sensitivities**:
- `calc_voltage_power_sensitivities(state)` -> `VoltagePowerSensitivity`
- `voltage_topology_sensitivities(net)` -> `VoltageTopologySensitivity`

### KKT System (src/opf/kkt.jl)

For implicit differentiation via `dz/dp = -(dK/dz)^{-1} * (dK/dp)`:

```julia
kkt_dims(net)                  # Total KKT dimension
flatten_variables(sol, prob)   # Solution -> vector
unflatten_variables(z, prob)   # Vector -> named tuple
kkt(z, prob, d)                # KKT residual vector
calc_kkt_jacobian(prob)        # Sparse Jacobian d(KKT)/dz
```

## File Organization

```
src/
├── PowerModelsDiff.jl          # Main module with exports
├── types/
│   ├── abstract.jl             # Abstract type hierarchy
│   ├── parameters.jl           # Sensitivity parameter types (singletons)
│   ├── dc_network.jl           # DCNetwork
│   ├── dc_states.jl            # DCPowerFlowState, DCOPFSolution
│   ├── ac_states.jl            # ACPowerFlowState
│   ├── ac_network.jl           # ACNetwork + constructors
│   └── sensitivities.jl        # All sensitivity result types
├── opf/
│   ├── problem.jl              # DCOPFProblem, solve!, DCNetwork constructors
│   └── kkt.jl                  # KKT system, OPF switching sensitivity
├── sens/
│   ├── interface.jl            # Unified calc_sensitivity() dispatch
│   ├── topology.jl             # DC PF switching/demand sensitivity
│   ├── demand.jl               # DC OPF demand sensitivity
│   ├── cost.jl                 # DC OPF cost sensitivity
│   ├── flowlimit.jl            # DC OPF flow limit sensitivity
│   ├── susceptance.jl          # DC OPF susceptance sensitivity
│   ├── voltage.jl              # AC voltage-power sensitivity
│   ├── current.jl              # AC current sensitivity
│   └── lmp.jl                  # LMP computation
├── pf/
│   ├── pf_eqns.jl              # Power flow equations (p, q, vm functions)
│   ├── admittance_matrix.jl    # VectorizedAdmittanceMatrix
│   └── bus_injection.jl        # Power injection calculations
├── graphs/
│   └── laplacian.jl            # Incidence matrices, Laplacian utilities
└── deprecated/
    ├── pf_structs.jl           # NetworkTopology, PowerFlowEquations (deprecated)
    └── measurements.jl         # Legacy state estimation types

test/
├── runtests.jl                 # Main test runner
├── common.jl                   # Shared test utilities
├── unified/
│   ├── test_interface.jl       # Unified API tests
│   └── test_sensitivity_verification.jl  # ForwardDiff validation
└── mwe_unified.jl              # Minimum working example
```

## Important Conventions

**PowerModels Integration**
- Networks must be processed with `make_basic_network()` before use
- Access via string keys: `net["branch"]["1"]`, `net["gen"]["1"]`
- Module alias: `const PM = PowerModels`

**Multiple Dispatch Style**
- Use dispatch on types, not function name suffixes
- Use `calc_` prefix for computation functions (PowerModels.jl convention)
- Sensitivity functions dispatch on state type AND parameter type

**Matrix Orientations**
- Incidence matrix `A` is (m x n): rows are branches, columns are buses
- B-theta Laplacian: `L = A' * Diagonal(-b .* z) * A`
- AC admittance: `Y = A' * Diag(g + j*b) * A + Diag(g_shunt + j*b_shunt)`

**Switching Variables**
- `z` in DCNetwork/ACNetwork stores switching states in [0,1]
- z=1 means branch closed (active), z=0 means open

**Default Solver**
- Clarabel (interior-point conic solver) is the default for DC OPF
- Other supported: Ipopt, HiGHS, Gurobi

## Key Exports

```julia
# Abstract Types
export AbstractPowerNetwork, AbstractPowerFlowState, AbstractOPFSolution, AbstractOPFProblem

# Parameter Types (for unified dispatch)
export DEMAND, GENERATION, COST, FLOWLIMIT, SUSCEPTANCE, SWITCHING, POWER, TOPOLOGY

# DC Types
export DCNetwork, DCPowerFlowState, DCOPFProblem, DCOPFSolution

# AC Types
export ACNetwork, ACPowerFlowState

# Sensitivity Result Types
export DemandSensitivity, CostSensitivity, SwitchingSensitivity
export FlowLimitSensitivity, SusceptanceSensitivity
export VoltagePowerSensitivity, VoltageTopologySensitivity
export CurrentPowerSensitivity, CurrentTopologySensitivity

# Unified Sensitivity Interface
export calc_sensitivity, calc_voltage_sensitivity

# DC Functions
export solve!, update_demand!, update_switching!
export calc_demand_vector, calc_susceptance_matrix
export calc_lmp, calc_congestion_component, calc_energy_component

# AC Functions
export admittance_matrix, branch_current, branch_power
export calc_voltage_power_sensitivities, voltage_topology_sensitivities
export p, q, vm, vm2, p_polar, q_polar

# KKT Functions (advanced)
export kkt, kkt_dims, calc_kkt_jacobian, flatten_variables, unflatten_variables

# Graph Utilities
export laplacian, full_incidence_matrix, calc_incidence_matrix
```
