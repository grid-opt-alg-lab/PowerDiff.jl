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
| `:psh` | Load shedding | DCOPFProblem |
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
| `:sw` | Switching states | DCPowerFlowState, DCOPFProblem, ACOPFProblem |
| `:cq`, `:cl` | Cost coefficients (quadratic, linear) | DCOPFProblem |
| `:fmax` | Flow limits | DCOPFProblem |
| `:b` | Susceptances | DCOPFProblem |
| `:p`, `:q` | Power injections (active, reactive) | ACPowerFlowState |

## Valid Combinations

### DC Power Flow (4 combinations)

| | `:d` | `:sw` |
|---|---|---|
| `:va` | ✓ | ✓ |
| `:f` | ✓ | ✓ |

### DC OPF (30 combinations)

| | `:d` | `:sw` | `:cq` | `:cl` | `:fmax` | `:b` |
|---|---|---|---|---|---|---|
| `:va` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `:pg` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `:f` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `:psh` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `:lmp` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### AC Power Flow (6 combinations)

| | `:p` | `:q` |
|---|---|---|
| `:vm` | ✓ | ✓ |
| `:v` | ✓ | ✓ |
| `:im` | ✓ | ✓ |

### AC OPF (4 combinations)

| | `:sw` |
|---|---|
| `:vm` | ✓ |
| `:va` | ✓ |
| `:pg` | ✓ |
| `:qg` | ✓ |

## Symbol Aliases

- `:g` → `:pg` (generator active power)
- `:pd` → `:d` (demand)

## Matrix Indexing Conventions

Sensitivity matrices use sequential 1-based indexing internally. The `Sensitivity{T}` type carries bidirectional ID mappings to translate between matrix indices and original element IDs:

```
S[i,j] = ∂(operand element i) / ∂(parameter element j)
S.row_to_id[i]  → original element ID for row i
S.col_to_id[j]  → original element ID for column j
S.id_to_row[id] → matrix row for original element ID
S.id_to_col[id] → matrix column for original element ID
```

For **basic networks** (sequential IDs), `row_to_id == 1:n`. For **non-basic networks** (arbitrary IDs, e.g., case5.m with bus IDs `[1,2,3,4,10]`), `row_to_id` contains the original IDs.

```julia
# Example: non-basic network with bus IDs [1,2,3,4,10]
S = calc_sensitivity(prob, :lmp, :d)
S.row_to_id          # [1, 2, 3, 4, 10]
S.id_to_row[10]      # 5 (bus 10 is at row 5)
S[5, 5]              # ∂LMP(bus 10) / ∂d(bus 10)
S[S.id_to_row[10], S.id_to_col[2]]  # ∂LMP(bus 10) / ∂d(bus 2)
```

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

## JVP / VJP

For ID-aware Jacobian-vector products, use [`jvp`](@ref) and [`vjp`](@ref). These accept `Dict{Int,Number}` keyed by original element IDs and return `Dict{Int,T}` keyed by original element IDs:

```julia
S = calc_sensitivity(prob, :lmp, :d)

# JVP: perturb demand at bus 10 by 0.1 MW
δlmp = jvp(S, Dict(10 => 0.1))
δlmp[10]  # LMP change at bus 10

# VJP: adjoint seed at bus 10
δd = vjp(S, Dict(10 => 1.0))
δd[2]     # adjoint contribution from bus 2
```

Missing keys are treated as zero; unknown IDs throw `ArgumentError`. Sequential vector inputs are also supported — they return `Dict` output:

```julia
jvp(S, randn(size(S, 2)))  # Vector in, Dict out
vjp(S, randn(size(S, 1)))  # Vector in, Dict out
```

For raw vector-in/vector-out, use `S * v` directly.

Conversion utilities [`dict_to_vec`](@ref) and [`vec_to_dict`](@ref) translate between `Dict{Int}` and dense vectors:

```julia
v = dict_to_vec(S, Dict(10 => 0.1), :col)   # parameter space
d = vec_to_dict(S, result_vec, :row)          # operand space
```
