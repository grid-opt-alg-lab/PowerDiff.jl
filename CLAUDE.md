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

# Build documentation
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
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
├── DCPowerFlowState    # DC power flow (θ_r = B_r \ p_r)
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
calc_sensitivity(state, :operand, :parameter) → Sensitivity{T}
```

Returns a `Sensitivity{T}` result that acts like a matrix but carries formulation/operand/parameter as symbol fields, plus bidirectional index mappings.

**Operand symbols** (what we differentiate):
- `:va` - Voltage phase angles (DC PF, DC OPF, AC OPF)
- `:f` - Branch flows (DC PF, DC OPF)
- `:pg` or `:g` - Generator active power (DC OPF, AC OPF)
- `:psh` - Load shedding (DC OPF)
- `:qg` - Generator reactive power (AC OPF)
- `:lmp` - Locational marginal prices (DC OPF)
- `:vm` - Voltage magnitude (AC PF, AC OPF)
- `:im` - Current magnitude (AC PF)
- `:v` - Complex voltage phasor (AC PF)

**Parameter symbols** (what we differentiate w.r.t.):
- `:d` or `:pd` - Demand
- `:sw` - Switching states
- `:cq`, `:cl` - Cost coefficients (DC OPF)
- `:fmax` - Flow limits (DC OPF)
- `:b` - Susceptances (DC OPF)
- `:p`, `:q` - Power injections (AC PF)

**Examples**:
```julia
# DC Power Flow
pf_state = DCPowerFlowState(net, d)
dva_dd = calc_sensitivity(pf_state, :va, :d)    # Sensitivity{Float64}, .formulation == :dcpf
df_dsw = calc_sensitivity(pf_state, :f, :sw)     # Sensitivity{Float64}, .formulation == :dcpf

# DC OPF (has LMP because it has duals)
prob = DCOPFProblem(net, d)
solve!(prob)
dlmp_dd = calc_sensitivity(prob, :lmp, :d)      # .formulation == :dcopf, .operand == :lmp
dpg_dcq = calc_sensitivity(prob, :pg, :cq)      # .formulation == :dcopf, .operand == :pg

# AC Power Flow
dvm_dp = calc_sensitivity(ac_state, :vm, :p)    # .formulation == :acpf, .operand == :vm

# AC OPF
ac_prob = ACOPFProblem(pm_data)
dvm_dsw = calc_sensitivity(ac_prob, :vm, :sw)    # .formulation == :acopf, .operand == :vm

# Invalid combinations throw ArgumentError
calc_sensitivity(pf_state, :lmp, :d)  # ERROR: no LMP for power flow
```

### Sensitivity{T} Return Type

`Sensitivity{T} <: AbstractMatrix{T}` — acts like a matrix with symbol metadata:
- `T <: Number`: Element type (Float64 for most, ComplexF64 for `:v` operand)

Fields:
- `matrix`: The sensitivity data (Matrix{T})
- `formulation`: Symbol (:dcpf, :dcopf, :acpf, :acopf)
- `operand`: Symbol (:va, :vm, :pg, :qg, :f, :psh, :lmp, :im, :v)
- `parameter`: Symbol (:d, :sw, :cq, :cl, :fmax, :b, :p, :q)
- `row_to_id`, `id_to_row`: Row index ↔ element ID
- `col_to_id`, `id_to_col`: Column index ↔ element ID

### Non-Basic Network Support & IDMapping

All constructors (`DCNetwork`, `ACNetwork`, `DCOPFProblem`, `ACOPFProblem`, `ACPowerFlowState`) accept networks with arbitrary element IDs. Internally, IDs are translated to sequential 1-based indices via `IDMapping`:

```julia
struct IDMapping
    bus_ids::Vector{Int}           # sorted original bus IDs
    branch_ids::Vector{Int}        # sorted original branch IDs
    gen_ids::Vector{Int}           # sorted original gen IDs
    load_ids::Vector{Int}          # sorted original load IDs
    shunt_ids::Vector{Int}         # sorted original shunt IDs
    bus_to_idx::Dict{Int,Int}      # original ID → sequential index
    # ... (branch_to_idx, gen_to_idx, load_to_idx, shunt_to_idx)
end
```

- Stored in `network.id_map` on `DCNetwork` and `ACNetwork`
- Internal computation uses sequential indices; original IDs flow to `Sensitivity.row_to_id`/`col_to_id`
- For basic networks, IDMapping is identity (1:n → 1:n)

### Matrix Indexing Conventions

Sensitivity matrices use **sequential 1-based indexing** internally, but `row_to_id`/`col_to_id` map back to original element IDs:

- `S[i,j]` = ∂(operand element i) / ∂(parameter element j)
- `S.row_to_id[i]` gives the original element ID for row i
- `S.col_to_id[j]` gives the original element ID for column j

For basic networks (sequential IDs), `row_to_id == 1:n`. For non-basic networks (e.g., case5.m with bus IDs `[1,2,3,4,10]`), `row_to_id == [1,2,3,4,10]`.

Example: `dg_dsw[i,j]` = ∂(generation at generator `dpg.row_to_id[i]`) / ∂(switching state of branch `dpg.col_to_id[j]`)

### DC OPF - B-theta Formulation

Uses susceptance-weighted Laplacian `B = A' * Diagonal(-b .* sw) * A`:

- `DCNetwork`: Network data (topology `A`, susceptances `b`, switching `sw`, limits, costs, `c_shed`)
- `DCOPFProblem`: JuMP optimization wrapper with `DCSensitivityCache` for efficient KKT reuse
- `DCOPFSolution`: Primal (θ, g, f, psh) and dual variables (ν_bal for LMPs)
- `DCPowerFlowState`: Non-OPF power flow (θ_r = B_r \ p_r, no optimization)

### AC OPF - Polar Formulation

- `ACOPFProblem`: Full nonlinear AC OPF via Ipopt, with `ACSensitivityCache` for efficient KKT reuse
- `ACOPFSolution`: Primal (va, vm, pg, qg) and dual variables
- Switching sensitivity via KKT implicit differentiation with ForwardDiff
- Multiple operands (`:vm`, `:va`, `:pg`, `:qg`) share a single cached `dz_dsw` computation

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
│   ├── abstract.jl             # Abstract type hierarchy
│   ├── id_mapping.jl           # IDMapping (original ↔ sequential ID translation)
│   ├── dc_network.jl           # DCNetwork, DCPowerFlowState, DCOPFSolution + constructors
│   ├── dc_opf_problem.jl       # DCOPFProblem, DCSensitivityCache + constructors
│   ├── ac_network.jl           # ACNetwork, ACPowerFlowState
│   ├── ac_opf_problem.jl       # ACOPFProblem, ACOPFSolution, ACSensitivityCache + constructors
│   ├── sensitivities.jl        # Sensitivity{T} <: AbstractMatrix{T}
│   └── show.jl                 # Pretty-printing (Base.show methods)
├── prob/
│   ├── dc_opf.jl               # DC OPF solve!, update_demand!
│   ├── kkt_dc_opf.jl           # DC KKT system, Jacobians, cached parameter derivatives
│   ├── ac_opf_solve.jl         # AC OPF solve!, update_switching!
│   └── kkt_ac_opf.jl           # AC KKT system, ForwardDiff Jacobians, cached switching sensitivity
├── sens/
│   ├── interface.jl            # Symbol dispatch → Sensitivity{T}
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
└── graphs/                     # Incidence matrices, Laplacian utilities

test/
├── runtests.jl                 # Main test runner (~810 lines inline + includes below)
├── common.jl                   # Shared helpers: load_test_case, create_2bus_network, etc.
├── test_ac_opf_sens.jl         # AC OPF sensitivity tests
├── test_ac_pf_verification.jl  # AC PF finite-difference verification
├── test_sensitivity_coverage.jl # Exhaustive (operand, parameter) coverage tests
├── test_dc_opf_verification.jl # DC OPF finite-difference verification
├── test_update_switching.jl    # update_switching! correctness tests
├── test_psh.jl                 # Load shedding sensitivity tests
├── test_nonbasic.jl            # Non-basic network support (arbitrary element IDs)
├── test_jvp_vjp.jl             # JVP/VJP with ID-aware Dict I/O
├── unified/
│   ├── test_interface.jl       # Unified API tests (symbol-based Sensitivity{T})
│   └── test_sensitivity_verification.jl  # ForwardDiff verification
├── mwe_unified.jl              # Minimum working example (symbol API)
└── smoke_rts_gmlc.jl           # RTS-GMLC smoke test (manual, not in Pkg.test)

examples/
└── interactive_repl.jl         # Interactive REPL walkthrough (case14)

docs/
├── Project.toml                # Documenter.jl dependencies
├── make.jl                     # Documenter build script
└── src/
    ├── index.md                # Landing page
    ├── getting-started.md      # DC PF → DC OPF → AC PF → AC OPF walkthrough
    ├── sensitivity-api.md      # Operand/parameter tables, valid combinations
    ├── math.md                 # B-theta, KKT, implicit differentiation
    ├── advanced.md             # Type hierarchy, caching, solver config
    ├── api.md                  # Auto-generated API reference
    └── assets/                 # Logo files
```

## Important Conventions

**PowerModels Integration**
- Both basic and non-basic networks are accepted (arbitrary bus/branch/gen IDs)
- `make_basic_network()` is optional; non-basic networks are translated internally via `IDMapping`
- All network constructors use `PM.build_ref()` internally for consistent data access
- Access via string keys: `net["branch"]["1"]`, `net["gen"]["1"]`
- Module alias: `const PM = PowerModels`

**Matrix Orientations**
- Incidence matrix `A` is (m × n): rows are branches, columns are buses
- B-theta Laplacian: `B = A' * Diagonal(-b .* sw) * A`
- Sign convention: `b` stores Im(1/z) < 0 for inductive branches. The negation `-b > 0` makes `B` positive-semidefinite. This is the **negative** of PowerModels' `calc_susceptance_matrix`.

**Switching Variables**
- `sw` stores switching states in [0,1]; sw=1 means branch closed, sw=0 means open
- Derivatives are continuous (not discrete on/off)

**Default Solver**
- Clarabel for DC OPF, Ipopt for AC OPF
- Override: `DCOPFProblem(net, d; optimizer=Ipopt.Optimizer)`

**Testing**
- `runtests.jl` contains ~800 lines of inline tests plus `include()` calls for 9 additional test files
- Tests call `PowerModels.silence()` at startup to suppress solver output
- `test/common.jl` provides `load_test_case()`, `create_2bus_network()`, `create_3bus_congested_network()` helpers (used by included test files, not by runtests.jl which defines its own `load_test_case`)
- `test/test_nonbasic.jl` verifies all features work with non-basic networks (case5.m, bus IDs `[1,2,3,4,10]`)

**Sensitivity Verification**
- All sensitivities are verified against ForwardDiff or finite differences in tests
- `test/unified/test_sensitivity_verification.jl`: DC PF (ForwardDiff), DC OPF demand/switching (FD), AC PF
- `test/test_ac_pf_verification.jl`: AC PF (finite-difference against Newton re-solve)
- `test/test_dc_opf_verification.jl`: DC OPF for cost, flow limit, susceptance, and remaining combos
