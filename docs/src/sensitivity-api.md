# Sensitivity API

The primary interface for computing sensitivities is:

```julia
calc_sensitivity(state, :operand, :parameter) → Sensitivity{T}
```

## Operand Symbols

Operand symbols specify what quantity we differentiate.

| Symbol | Description | Formulations |
|--------|-------------|------------|
| `:va` | Voltage phase angles | DCPowerFlowState, DCOPFProblem, ACOPFProblem |
| `:f` | Branch flows | DCPowerFlowState, DCOPFProblem |
| `:pg` / `:g` | Generator active power | DCOPFProblem, ACOPFProblem |
| `:qg` | Generator reactive power | ACOPFProblem |
| `:lmp` | Locational marginal prices | DCOPFProblem |
| `:vm` | Voltage magnitude | ACPowerFlowState, ACOPFProblem |
| `:im` | Current magnitude | ACPowerFlowState |
| `:v` | Complex voltage phasor | ACPowerFlowState |

## Parameter Symbols

Parameter symbols specify what we differentiate with respect to.

| Symbol | Description | Formulations |
|--------|-------------|------------|
| `:d` / `:pd` | Demand | DCPowerFlowState, DCOPFProblem |
| `:z` | Switching states | DCPowerFlowState, DCOPFProblem, ACOPFProblem |
| `:cq`, `:cl` | Cost coefficients (quadratic, linear) | DCOPFProblem |
| `:fmax` | Flow limits | DCOPFProblem |
| `:b` | Susceptances | DCOPFProblem |
| `:p`, `:q` | Power injections (active, reactive) | ACPowerFlowState |

## Valid Combinations

### DC Power Flow (4 combinations)

| | `:d` | `:z` |
|---|---|---|
| `:va` | ✓ | ✓ |
| `:f` | ✓ | ✓ |

### DC OPF (24 combinations)

| | `:d` | `:z` | `:cq` | `:cl` | `:fmax` | `:b` |
|---|---|---|---|---|---|---|
| `:va` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `:pg` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `:f` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `:lmp` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### AC Power Flow (6 combinations)

| | `:p` | `:q` |
|---|---|---|
| `:vm` | ✓ | ✓ |
| `:v` | ✓ | ✓ |
| `:im` | ✓ | ✓ |

### AC OPF (4 combinations)

| | `:z` |
|---|---|
| `:vm` | ✓ |
| `:va` | ✓ |
| `:pg` | ✓ |
| `:qg` | ✓ |

## Symbol Aliases

- `:g` → `:pg` (generator active power)
- `:pd` → `:d` (demand)

## Matrix Indexing Conventions

All sensitivity matrices use sequential 1-based indexing matching PowerModels string keys:

```
S[i,j] = ∂(operand element i) / ∂(parameter element j)
```

- **Buses** `1:n` → `net["bus"]["1"]` through `net["bus"]["$n"]`
- **Branches** `1:m` → `net["branch"]["1"]` through `net["branch"]["$m"]`
- **Generators** `1:k` → `net["gen"]["1"]` through `net["gen"]["$k"]`

To find connections:
- Generator `i` is at bus: `net["gen"]["$i"]["gen_bus"]`
- Branch `j` connects: `net["branch"]["$j"]["f_bus"]` → `net["branch"]["$j"]["t_bus"]`

## Error Handling

Invalid operand/parameter combinations throw `ArgumentError` with a message listing all valid combinations for the given state type:

```julia
# DCPowerFlowState has no LMP
calc_sensitivity(pf_state, :lmp, :d)
# ArgumentError: calc_sensitivity(DCPowerFlowState, :lmp, :d) is not defined.
# Valid combinations for DCPowerFlowState:
#   :va w.r.t. :d
#   :f w.r.t. :d
#   ...
```

## `Sensitivity{T}` Return Type

[`Sensitivity{T}`](@ref) is an `AbstractMatrix{T}` with metadata fields:

- `matrix::Matrix{T}`: The raw sensitivity data
- `formulation::Symbol`: `:dcpf`, `:dcopf`, `:acpf`, or `:acopf`
- `operand::Symbol`: The operand symbol
- `parameter::Symbol`: The parameter symbol
- `row_to_id` / `id_to_row`: Bidirectional row index ↔ element ID mapping
- `col_to_id` / `id_to_col`: Bidirectional column index ↔ element ID mapping

Standard matrix operations work directly:

```julia
sens = calc_sensitivity(prob, :lmp, :d)
size(sens)            # (n, n)
sens[2, 3]            # element access
sens * v              # matrix-vector product
Matrix(sens)          # extract raw matrix
```
