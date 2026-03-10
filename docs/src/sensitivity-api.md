# Sensitivity API

The primary interface for computing sensitivities is:

```julia
calc_sensitivity(state, :operand, :parameter) → Sensitivity{T}
```

## Operand Symbols

Operand symbols specify what quantity we differentiate.

| Symbol | Description | Formulations |
|--------|-------------|------------|
| `:va` | Voltage phase angles | DCPowerFlowState, DCOPFProblem, ACPowerFlowState, ACOPFProblem |
| `:f` | Branch active power flows | DCPowerFlowState, DCOPFProblem, ACPowerFlowState |
| `:pg` / `:g` | Generator active power | DCOPFProblem, ACOPFProblem |
| `:psh` | Load shedding | DCOPFProblem |
| `:qg` | Generator reactive power | ACOPFProblem |
| `:lmp` | Locational marginal prices | DCOPFProblem, ACOPFProblem |
| `:vm` | Voltage magnitude | ACPowerFlowState, ACOPFProblem |
| `:im` | Current magnitude | ACPowerFlowState |
| `:v` | Complex voltage phasor | ACPowerFlowState |
| `:p` | Active power injection | ACPowerFlowState (Jacobian block) |
| `:q` | Reactive power injection | ACPowerFlowState (Jacobian block) |

## Parameter Symbols

Parameter symbols specify what we differentiate with respect to.

| Symbol | Description | Formulations |
|--------|-------------|------------|
| `:d` / `:pd` | Active demand | DCPowerFlowState, DCOPFProblem, ACPowerFlowState (via transform), ACOPFProblem |
| `:qd` | Reactive demand | ACPowerFlowState (via transform), ACOPFProblem |
| `:sw` | Switching states | DCPowerFlowState, DCOPFProblem, ACOPFProblem |
| `:cq`, `:cl` | Cost coefficients (quadratic, linear) | DCOPFProblem, ACOPFProblem |
| `:fmax` | Flow limits | DCOPFProblem, ACOPFProblem |
| `:b` | Susceptances | DCPowerFlowState, DCOPFProblem |
| `:p`, `:q` | Power injections (active, reactive) | ACPowerFlowState |
| `:va` | Voltage phase angle | ACPowerFlowState (Jacobian block parameter) |
| `:vm` | Voltage magnitude | ACPowerFlowState (Jacobian block parameter) |

## Valid Combinations

### DC Power Flow (6 combinations)

| | `:d` | `:sw` | `:b` |
|---|---|---|---|
| `:va` | ✓ | ✓ | ✓ |
| `:f` | ✓ | ✓ | ✓ |

### DC OPF (30 combinations)

| | `:d` | `:sw` | `:cq` | `:cl` | `:fmax` | `:b` |
|---|---|---|---|---|---|---|
| `:va` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `:pg` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `:f` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `:psh` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `:lmp` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### AC Power Flow (24 combinations: 14 native + 10 via transforms)

**Native combinations (14):**

| | `:p` | `:q` | `:va` | `:vm` |
|---|---|---|---|---|
| `:vm` | ✓ | ✓ | | |
| `:v` | ✓ | ✓ | | |
| `:im` | ✓ | ✓ | | |
| `:va` | ✓ | ✓ | | |
| `:f` | ✓ | ✓ | | |
| `:p` | | | ✓ | ✓ |
| `:q` | | | ✓ | ✓ |

**Transform-derived combinations (10):**

Via `∂/∂d = -∂/∂p` and `∂/∂qd = -∂/∂q` (since `p_net = pg - pd` with `pg` fixed in power flow). Only applies to operands that have `:p`/`:q` as a native parameter:

| | `:d` | `:qd` |
|---|---|---|
| `:vm` | ✓ | ✓ |
| `:v` | ✓ | ✓ |
| `:im` | ✓ | ✓ |
| `:va` | ✓ | ✓ |
| `:f` | ✓ | ✓ |

### AC OPF (30 combinations)

| | `:sw` | `:d` | `:qd` | `:cq` | `:cl` | `:fmax` |
|---|---|---|---|---|---|---|
| `:va` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `:vm` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `:pg` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `:qg` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `:lmp` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Power Flow Jacobian

The AC power flow Jacobian relates power injections to voltage state variables. The 4 standard blocks are available as sensitivity combinations:

```julia
state = ACPowerFlowState(pf_data)

J1 = calc_sensitivity(state, :p, :va)   # ∂P/∂θ  (n × n)
J2 = calc_sensitivity(state, :p, :vm)   # ∂P/∂|V| (n × n)
J3 = calc_sensitivity(state, :q, :va)   # ∂Q/∂θ  (n × n)
J4 = calc_sensitivity(state, :q, :vm)   # ∂Q/∂|V| (n × n)
```

These are **raw** Jacobian blocks for ALL buses (not reduced by bus type). For the reduced Newton-Raphson Jacobian, extract the non-reference bus submatrix:

```julia
non_ref = [i for i in 1:n if i != slack_bus]
J_NR = [Matrix(J1)[non_ref, non_ref]  Matrix(J2)[non_ref, non_ref];
        Matrix(J3)[non_ref, non_ref]  Matrix(J4)[non_ref, non_ref]]
```

The direct function `calc_power_flow_jacobian(state)` returns all 4 blocks at once as a NamedTuple.

### Bus-Type Enforcement

By default, the Jacobian contains raw partial derivatives for all buses. To get the Newton-Raphson form matching PowerModels' `calc_basic_jacobian_matrix`, pass a `bus_types` vector:

```julia
bus_types = [pf_data["bus"]["$i"]["bus_type"] for i in 1:n]
jac = calc_power_flow_jacobian(state; bus_types=bus_types)
```

Bus-type column modifications:
- **PQ (type 1)**: No modification — raw derivatives
- **PV (type 2)**: θ columns unchanged; |V| columns zeroed with `∂Q_j/∂|V_j| = 1`
- **Slack (type 3)**: Both θ and |V| columns become unit vectors (`e_j`)

The `calc_sensitivity` interface (`:p`/`:q` w.r.t. `:va`/`:vm`) always returns raw derivatives, which is correct for sensitivity analysis.

## Parameter Transforms

Some parameter symbols are derived from native parameters via type-specific transforms. These transforms are only valid for specific state types:

- **ACPowerFlowState**: `∂/∂d = -∂/∂p` and `∂/∂qd = -∂/∂q`
  - Valid because in power flow, `p_net = pg - pd` with `pg` fixed, so `∂p_net/∂pd = -1`
  - Does NOT apply to OPF, where demand sensitivity goes through KKT re-optimization

Transforms are transparent: `calc_sensitivity(state, :vm, :d)` automatically computes `-calc_sensitivity(state, :vm, :p)`.

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

Invalid operand/parameter combinations throw `ArgumentError` with a message listing all valid combinations (including transform-derived) for the given state type:

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

For ID-aware Jacobian-vector products, use [`jvp`](@ref) and [`vjp`](@ref). These accept a `Dict` keyed by original element IDs (e.g., `Dict(10 => 0.1)`) and return `Dict{Int,T}` keyed by original element IDs:

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
