# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PowerModelsDiff is a Julia package for differentiable power system analysis. It provides sensitivity analysis for power flow equations, optimal power flow (OPF) problems, and power networks. Built on PowerModels.jl.

## Development Commands

```bash
# Activate and run in REPL
julia --project=.

# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run all tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Run specific test file
julia --project=. test/unified/test_interface.jl

# Run MWE (minimum working example)
julia --project=. test/mwe_unified.jl
```

**Test Data**: Tests use PowerModels' built-in MATPOWER cases (case5.m, case14.m, etc.) located at:
```julia
joinpath(dirname(pathof(PowerModels)), "..", "test", "data", "matpower")
```

## Architecture

### Unified Type Hierarchy

```
AbstractPowerNetwork
├── DCNetwork           # DC B-theta formulation
└── ACNetwork           # AC with vectorized admittance

AbstractPowerFlowState
├── DCPowerFlowState    # DC power flow (θ = L⁺ * p)
├── ACPowerFlowState    # AC power flow (complex voltages)
└── AbstractOPFSolution
    └── DCOPFSolution   # DC OPF with generation, flows, duals

AbstractOPFProblem
└── DCOPFProblem        # JuMP-based DC OPF wrapper
```

### Symbol-Based Sensitivity API

The primary sensitivity interface uses symbol dispatch:

```julia
calc_sensitivity(state, :operand, :parameter) → Matrix
```

**Operand symbols** (what we differentiate):
- `:va` - Voltage phase angles (DC)
- `:f` - Branch flows (DC)
- `:pg` or `:g` - Generator power (DC OPF only)
- `:lmp` - Locational marginal prices (DC OPF only)
- `:vm` - Voltage magnitude (AC)
- `:im` - Current magnitude (AC)

**Parameter symbols** (what we differentiate w.r.t.):
- `:d` or `:pd` - Demand
- `:z` - Switching states
- `:cq`, `:cl` - Cost coefficients (DC OPF)
- `:fmax` - Flow limits (DC OPF)
- `:b` - Susceptances (DC OPF)
- `:p`, `:q` - Power injections (AC)

**Examples**:
```julia
# DC Power Flow
pf_state = DCPowerFlowState(net, d)
dva_dd = calc_sensitivity(pf_state, :va, :d)    # n×n
df_dz = calc_sensitivity(pf_state, :f, :z)      # m×m

# DC OPF (has LMP because it has duals)
prob = DCOPFProblem(net, d)
solve!(prob)
dlmp_dd = calc_sensitivity(prob, :lmp, :d)      # n×n
dpg_dcq = calc_sensitivity(prob, :pg, :cq)      # k×k

# AC Power Flow
dvm_dp = calc_sensitivity(ac_state, :vm, :p)    # n×n

# Invalid combinations throw ArgumentError
calc_sensitivity(pf_state, :lmp, :d)  # ERROR: no LMP for power flow
```

### Legacy Two-Argument API (Deprecated)

The old API using parameter singletons (`DEMAND`, `SWITCHING`, etc.) still works but shows deprecation warnings:

```julia
# Old (deprecated, shows warning):
sens = calc_sensitivity(prob, DEMAND)
sens.dlmp_dd  # Access bundled struct field

# New (preferred):
dlmp_dd = calc_sensitivity(prob, :lmp, :d)
```

### DC OPF - B-theta Formulation

Uses susceptance-weighted Laplacian `L = A' * Diagonal(-b .* z) * A`:

- `DCNetwork`: Network data (topology `A`, susceptances `b`, switching `z`, limits, costs)
- `DCOPFProblem`: JuMP optimization wrapper
- `DCOPFSolution`: Primal (θ, g, f) and dual variables (ν_bal for LMPs)
- `DCPowerFlowState`: Non-OPF power flow (θ = L⁺ * p, no optimization)

### KKT System (src/prob/kkt_dc_opf.jl)

For implicit differentiation via `dz/dp = -(dK/dz)⁻¹ * (dK/dp)`:

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
│   ├── parameters.jl           # Sensitivity parameter singletons (legacy)
│   ├── dc_network.jl           # DCNetwork, DCPowerFlowState, DCOPFSolution + constructors
│   ├── dc_opf_problem.jl       # DCOPFProblem struct + constructors
│   ├── ac_network.jl           # ACNetwork, ACPowerFlowState
│   └── sensitivities.jl        # Sensitivity result types (state-specific + legacy)
├── prob/
│   ├── dc_opf.jl               # solve!, update_demand!
│   └── kkt_dc_opf.jl           # KKT system, Jacobians, switching sensitivity
├── sens/
│   ├── interface.jl            # Symbol-based calc_sensitivity() dispatch
│   ├── topology.jl             # DC PF switching/demand sensitivity
│   ├── demand.jl               # DC OPF demand sensitivity
│   ├── cost.jl                 # DC OPF cost sensitivity
│   ├── flowlimit.jl            # DC OPF flow limit sensitivity
│   ├── susceptance.jl          # DC OPF susceptance sensitivity
│   ├── voltage.jl              # AC voltage-power sensitivity
│   ├── current.jl              # AC current sensitivity
│   └── lmp.jl                  # LMP computation
├── pf/                         # Power flow equations
├── graphs/                     # Incidence matrices, Laplacian utilities
└── deprecated/                 # Legacy types

test/
├── runtests.jl                 # Main test runner
├── unified/
│   ├── test_interface.jl       # Unified API tests
│   └── test_sensitivity_verification.jl  # ForwardDiff verification
└── mwe_unified.jl              # Minimum working example
```

## Important Conventions

**PowerModels Integration**
- Networks must be processed with `make_basic_network()` before use
- Access via string keys: `net["branch"]["1"]`, `net["gen"]["1"]`
- Module alias: `const PM = PowerModels`

**Matrix Orientations**
- Incidence matrix `A` is (m × n): rows are branches, columns are buses
- B-theta Laplacian: `L = A' * Diagonal(-b .* z) * A`

**Switching Variables**
- `z` stores switching states in [0,1]; z=1 means branch closed

**Default Solver**
- Clarabel (interior-point conic solver) is the default for DC OPF
- Use alternate solvers: `DCOPFProblem(net, d; optimizer=Ipopt.Optimizer)`

**Sensitivity Verification**
- All sensitivities are verified against ForwardDiff in tests
- Run `test/unified/test_sensitivity_verification.jl` to check numerical correctness
