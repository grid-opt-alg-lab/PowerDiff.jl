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

### Core Components

**DC OPF - B-theta Formulation (src/opf/)**

The primary feature is a differentiable DC OPF using the susceptance-weighted Laplacian (B-theta) formulation:

- `DCNetwork`: Network data struct with topology (`A`), generator mapping (`G_inc`), susceptances (`b`), switching states (`z`), limits, and costs
- `DCOPFProblem`: JuMP-based optimization wrapper with variables theta (angles), g (generation), f (flows)
- `DCOPFSolution`: Solution container with primal values and dual variables
- `solve!(prob)`: Solve and return solution with duals

Key files:
- `src/opf/problem.jl`: Network construction, problem formulation, solving
- `src/opf/kkt.jl`: KKT system for implicit differentiation

**Sensitivity Analysis (src/sens/)**

Sensitivities computed via implicit function theorem on KKT conditions:

- `calc_sensitivity_demand(prob)`: Returns `DemandSensitivity` with dtheta_dd, dg_dd, df_dd, dlmp_dd
- `calc_sensitivity_switching(prob)`: Returns `TopologySensitivity` with derivatives w.r.t. switching variables
- `calc_lmp(sol, net)`: Locational Marginal Prices from duals

Key files:
- `src/sens/demand.jl`: Demand sensitivity via KKT system
- `src/sens/lmp.jl`: LMP computation and decomposition
- `src/sens/topology.jl`: AC voltage topology sensitivity (existing)

**Vectorized Admittance Representation (src/graphs/, src/pf/)**

The package uses vectorized admittance for efficient differentiation:
- Off-diagonal elements stored as `G_edges`, `B_edges`
- Self-edges stored separately as `G_self`, `B_self`
- Reconstruction: `A' * Diagonal(G + B*im) * A`

Key utilities:
- `vectorize_laplacian_weights(net)`: Convert admittance matrix to vector form
- `calc_incidence_matrix(net; full_nodes, full_edges)`: Flexible incidence construction
- `laplacian(G, B, n)`: Reconstruct Laplacian from vectors

### Key Structs (src/structs.jl)

```julia
# DC OPF Types
struct DCNetwork           # Network topology and parameters
struct DCOPFSolution       # Primal and dual solution values
mutable struct DCOPFProblem # JuMP model wrapper

# Sensitivity Types
struct DemandSensitivity      # dtheta/dd, dg/dd, df/dd, dlmp/dd
struct TopologySensitivity    # dtheta/ds, dg/ds, df/ds, dlmp/ds

# AC Sensitivity Types
struct VectorizedAdmittanceMatrix
struct VoltageSensitivityTopology
```

### KKT System

For implicit differentiation, the KKT conditions are structured as:

```julia
# Variable ordering in flattened vector z:
# [theta(n), g(k), f(m), lambda_lb(m), lambda_ub(m), rho_lb(k), rho_ub(k), nu_bal(n), nu_flow(m), eta_ref(1)]

kkt_dims(net)              # Total dimension: 2n + 4m + 3k + 1
flatten_variables(sol, prob)   # Solution -> vector
unflatten_variables(z, prob)   # Vector -> named tuple
kkt(z, prob, d)            # Evaluate KKT residuals
calc_kkt_jacobian(prob)    # Analytical sparse Jacobian d(KKT)/dz
```

Sensitivities computed as: `dz/dp = -(dK/dz)^{-1} * (dK/dp)`

## Important Conventions

**PowerModels Integration**
- Networks must be processed with `make_basic_network()` before use
- Access via string keys: `net["branch"]["1"]`, `net["gen"]["1"]`
- Module alias: `const PM = PowerModels`

**Multiple Dispatch Style**
- Use dispatch instead of function name suffixes (e.g., `kkt(prob::DCOPFProblem)` not `kkt_Btheta`)
- Use `calc_` prefix for computation functions (following PowerModels.jl style)

**Matrix Orientations**
- Incidence matrix `A` is (m x n): rows are branches, columns are buses
- B-theta formulation: `B = A' * Diagonal(-b .* z) * A` (susceptance-weighted Laplacian)

**Switching Variables**
- `z` in DCNetwork stores switching states in [0,1]
- z=1 means branch is closed (active), z=0 means open

**Default Solver**
- Clarabel (interior-point conic solver) is the default
- Other supported: Ipopt, HiGHS, Gurobi

## File Organization

```
src/
├── PowerModelsDiff.jl     # Main module with exports
├── structs.jl             # Type definitions
├── opf/
│   ├── problem.jl         # DCNetwork, DCOPFProblem, solve!
│   └── kkt.jl            # KKT system, Jacobians, topology sensitivity
├── sens/
│   ├── demand.jl         # Demand sensitivity analysis
│   ├── lmp.jl            # LMP computation
│   └── topology.jl       # AC voltage topology sensitivity
├── pf/
│   ├── admittance_matrix.jl  # VectorizedAdmittanceMatrix
│   ├── bus_injection.jl      # Power injection calculations
│   └── pf_eqns.jl           # Power flow equation structs
├── graphs/
│   └── laplacian.jl      # Incidence matrices, Laplacian utilities
└── deprecated/           # Legacy implementations (dcopf.jl, dcopf_B_theta.jl)

test/
└── runtests.jl           # Comprehensive test suite
```

## Exports

```julia
# DC OPF Types
export DCNetwork, DCOPFProblem, DCOPFSolution

# OPF Functions
export solve!, update_demand!
export calc_demand_vector, calc_susceptance_matrix

# LMP Functions
export calc_lmp, calc_lmp_from_duals, calc_congestion_component

# Sensitivity Types
export DemandSensitivity, TopologySensitivity

# Sensitivity Functions
export calc_sensitivity_demand, calc_sensitivity_demand_primal
export calc_sensitivity_switching, update_switching!
export calc_generation_participation_factors, calc_ptdf_from_sensitivity

# KKT Functions
export kkt, kkt_dims, calc_kkt_jacobian
export flatten_variables, unflatten_variables

# AC Topology Sensitivity (existing)
export VoltageSensitivityTopology, voltage_topology_sensitivities

# Graph Utilities
export VectorizedAdmittanceMatrix, vectorize_laplacian_weights
export laplacian, full_incidence_matrix, calc_incidence_matrix
```
