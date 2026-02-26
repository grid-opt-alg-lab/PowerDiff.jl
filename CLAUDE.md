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
    ├── DCOPFSolution   # DC OPF with generation, flows, duals
    └── ACOPFSolution   # AC OPF with voltages, generation, duals

AbstractOPFProblem
├── DCOPFProblem        # JuMP-based DC OPF wrapper
└── ACOPFProblem        # JuMP-based AC OPF wrapper
```

### Sensitivity API

The **primary interface** uses symbol dispatch:

```julia
calc_sensitivity(state, :operand, :parameter) → Sensitivity{F, O, P}
```

Returns a `Sensitivity{F,O,P}` typed result that acts like a matrix but carries formulation/operand/parameter type tags and bidirectional index mappings.

**Operand symbols** (what we differentiate):
- `:va` - Voltage phase angles (DC PF, DC OPF, AC OPF)
- `:f` - Branch flows (DC PF, DC OPF)
- `:pg` or `:g` - Generator active power (DC OPF, AC OPF)
- `:qg` - Generator reactive power (AC OPF)
- `:lmp` - Locational marginal prices (DC OPF)
- `:vm` - Voltage magnitude (AC PF, AC OPF)
- `:im` - Current magnitude (AC PF)
- `:v` - Complex voltage phasor (AC PF)

**Parameter symbols** (what we differentiate w.r.t.):
- `:d` or `:pd` - Demand
- `:z` - Switching states
- `:cq`, `:cl` - Cost coefficients (DC OPF)
- `:fmax` - Flow limits (DC OPF)
- `:b` - Susceptances (DC OPF)
- `:p`, `:q` - Power injections (AC PF)

**Examples**:
```julia
# DC Power Flow
pf_state = DCPowerFlowState(net, d)
dva_dd = calc_sensitivity(pf_state, :va, :d)    # Sensitivity{DCPF, VoltageAngle, Demand}
df_dz = calc_sensitivity(pf_state, :f, :z)      # Sensitivity{DCPF, Flow, Switching}

# DC OPF (has LMP because it has duals)
prob = DCOPFProblem(net, d)
solve!(prob)
dlmp_dd = calc_sensitivity(prob, :lmp, :d)      # Sensitivity{DCOPF, LMP, Demand}
dpg_dcq = calc_sensitivity(prob, :pg, :cq)      # Sensitivity{DCOPF, Generation, QuadraticCost}

# AC Power Flow
dvm_dp = calc_sensitivity(ac_state, :vm, :p)    # Sensitivity{ACPF, VoltageMagnitude, ActivePower}

# AC OPF
ac_prob = ACOPFProblem(pm_data)
dvm_dz = calc_sensitivity(ac_prob, :vm, :z)     # Sensitivity{ACOPF, VoltageMagnitude, Switching}

# Invalid combinations throw ArgumentError
calc_sensitivity(pf_state, :lmp, :d)  # ERROR: no LMP for power flow
```

### Singleton Type API (Power Users)

Symbols map to singleton types internally. Power users can call singletons directly for type dispatch:

```julia
# Equivalent to calc_sensitivity(prob, :lmp, :d)
sens = calc_sensitivity(prob, LMP(), Demand())   # Sensitivity{DCOPF, LMP, Demand}

# Dispatch on result type
process(s::Sensitivity{DCOPF, LMP, Demand}) = "DC OPF LMP-demand"
process(s::Sensitivity{F, O, Switching}) where {F, O} = "Any switching sensitivity"
```

### Sensitivity{F,O,P} Return Type

`Sensitivity{F,O,P} <: AbstractMatrix{Float64}` — acts like a matrix with type tags:
- `F <: AbstractFormulation`: DCOPF, ACOPF, DCPF, ACPF
- `O <: AbstractOperand`: VoltageAngle, VoltageMagnitude, Generation, LMP, Flow, etc.
- `P <: AbstractParameter`: Demand, Switching, QuadraticCost, LinearCost, etc.

Fields:
- `matrix`: The sensitivity data
- `row_to_id`, `id_to_row`: Row index ↔ element ID
- `col_to_id`, `id_to_col`: Column index ↔ element ID

### Matrix Indexing Conventions

Sensitivity matrices use **sequential 1-based indexing** matching PowerModels keys:

- `S[i,j]` = ∂(operand element i) / ∂(parameter element j)
- Buses: indices 1:n → `net["bus"]["1"]` through `net["bus"]["$n"]`
- Branches: indices 1:m → `net["branch"]["1"]` through `net["branch"]["$m"]`
- Generators: indices 1:k → `net["gen"]["1"]` through `net["gen"]["$k"]`

To find connections:
- Generator i is at bus: `net["gen"]["$i"]["gen_bus"]`
- Branch j connects: `net["branch"]["$j"]["f_bus"]` → `net["branch"]["$j"]["t_bus"]`

Example: `dg_dz[i,j]` = ∂(generation at generator i) / ∂(switching state of branch j)

### DC OPF - B-theta Formulation

Uses susceptance-weighted Laplacian `L = A' * Diagonal(-b .* z) * A`:

- `DCNetwork`: Network data (topology `A`, susceptances `b`, switching `z`, limits, costs)
- `DCOPFProblem`: JuMP optimization wrapper with `SensitivityCache` for efficient KKT reuse
- `DCOPFSolution`: Primal (θ, g, f) and dual variables (ν_bal for LMPs)
- `DCPowerFlowState`: Non-OPF power flow (θ = L⁺ * p, no optimization)

### AC OPF - Polar Formulation

- `ACOPFProblem`: Full nonlinear AC OPF via Ipopt
- `ACOPFSolution`: Primal (va, vm, pg, qg) and dual variables
- Switching sensitivity via KKT implicit differentiation with ForwardDiff

### KKT Systems

For implicit differentiation via `dz/dp = -(dK/dz)⁻¹ * (dK/dp)`:

```julia
# DC OPF (analytical Jacobian)
kkt_dims(net)                  # Total KKT dimension
flatten_variables(sol, prob)   # Solution -> vector
calc_kkt_jacobian(prob)        # Sparse Jacobian d(KKT)/dz

# AC OPF (ForwardDiff Jacobian)
ac_kkt_dims(prob)              # Total KKT dimension
calc_ac_kkt_jacobian(prob)     # Dense Jacobian via ForwardDiff
```

## File Organization

```
src/
├── PowerModelsDiff.jl          # Main module with exports
├── types/
│   ├── abstract.jl             # Abstract type hierarchy + singleton tag bases
│   ├── tags.jl                 # Singleton type tags (DCOPF, LMP, Demand, etc.)
│   ├── dc_network.jl           # DCNetwork, DCPowerFlowState, DCOPFSolution + constructors
│   ├── dc_opf_problem.jl       # DCOPFProblem, SensitivityCache + constructors
│   ├── ac_network.jl           # ACNetwork, ACPowerFlowState
│   ├── ac_opf_problem.jl       # ACOPFProblem, ACOPFSolution + constructors
│   └── sensitivities.jl        # Sensitivity{F,O,P}, bundled types (DCPFSwitchingSens, ACOPFSwitchingSens)
├── prob/
│   ├── dc_opf.jl               # DC OPF solve!, update_demand!
│   ├── kkt_dc_opf.jl           # DC KKT system, Jacobians, cached parameter derivatives
│   ├── ac_opf_solve.jl         # AC OPF solve!, update_switching!
│   └── kkt_ac_opf.jl           # AC KKT system, ForwardDiff Jacobians, switching sensitivity
├── sens/
│   ├── interface.jl            # Symbol dispatch + singleton dispatch → Sensitivity{F,O,P}
│   ├── index_mapping.jl        # Bidirectional index mappings (bus/branch/gen)
│   ├── topology.jl             # DC PF switching/demand sensitivity
│   ├── demand.jl               # DC OPF demand sensitivity (cached)
│   ├── cost.jl                 # DC OPF cost sensitivity (cached)
│   ├── flowlimit.jl            # DC OPF flow limit sensitivity (cached)
│   ├── susceptance.jl          # DC OPF susceptance sensitivity (cached)
│   ├── voltage.jl              # AC voltage-power sensitivity
│   ├── current.jl              # AC current sensitivity
│   └── lmp.jl                  # LMP computation
├── pf/                         # Power flow equations
├── graphs/                     # Incidence matrices, Laplacian utilities
└── deprecated/                 # Legacy types

test/
├── runtests.jl                 # Main test runner
├── test_ac_opf_sens.jl         # AC OPF sensitivity tests
├── unified/
│   ├── test_interface.jl       # Unified API tests (singleton types)
│   └── test_sensitivity_verification.jl  # ForwardDiff verification
└── mwe_unified.jl              # Minimum working example (symbol API)
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
- `z` stores switching states in [0,1]; z=1 means branch closed, z=0 means open
- Derivatives are continuous (not discrete on/off)

**Default Solver**
- Clarabel for DC OPF, Ipopt for AC OPF
- Override: `DCOPFProblem(net, d; optimizer=Ipopt.Optimizer)`

**Sensitivity Verification**
- All sensitivities are verified against ForwardDiff in tests
- Run `test/unified/test_sensitivity_verification.jl` to check numerical correctness
