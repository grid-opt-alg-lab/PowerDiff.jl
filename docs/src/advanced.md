# Advanced Topics

## Type Hierarchy

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

## Core Types

### DCNetwork

Stores the DC network topology and parameters.

| Field | Type | Description |
|-------|------|-------------|
| `n`, `m`, `k` | `Int` | Number of buses, branches, generators |
| `A` | `SparseMatrixCSC` | Incidence matrix (m × n) |
| `G_inc` | `SparseMatrixCSC` | Generator-bus incidence (n × k) |
| `b` | `Vector{Float64}` | Branch susceptances |
| `sw` | `Vector{Float64}` | Switching states in [0,1] |
| `fmax` | `Vector{Float64}` | Branch flow limits |
| `gmax`, `gmin` | `Vector{Float64}` | Generator limits |
| `angmax`, `angmin` | `Vector{Float64}` | Phase angle difference limits |
| `cq`, `cl` | `Vector{Float64}` | Cost coefficients (quadratic, linear) |
| `c_shed` | `Vector{Float64}` | Load shedding cost per bus |
| `demand` | `Vector{Float64}` | Real power demand aggregated per bus |
| `pg_init` | `Vector{Float64}` | Initial real generation aggregated per bus |
| `ref_bus` | `Int` | Preferred reference bus index (sequential) |
| `tau` | `Float64` | Regularization parameter |
| `id_map` | `IDMapping` | Bidirectional element ID mapping (original ↔ sequential) |
| `topology_cache` | `_DCTopologyCache` | Internal energized island cache (not part of the public API) |

Construct from a parsed MATPOWER network with `DCNetwork(parse_file("case14.m"))`, or
with explicit parameters: `DCNetwork(n, m, k, A, G_inc, b; ...)`.

Use `reference_buses(net)` to obtain the effective reference set. The choice is
deterministic: `ref_bus` is kept as the reference for its energized island, and
every other island (including an isolated bus) uses its lowest sequential bus
index.

`DCNetwork` precomputes an internal energized topology cache and refreshes it
when topology readers observe a direct `b` or `sw` change. This cache is not a
thread safety mechanism. Sharing a `DCNetwork` across threads is supported only
when topology fields are treated as read only; callers that mutate `b` or `sw`
directly must serialize the mutation and the next topology dependent read. For
`DCOPFProblem`, switch changes should go through [`update_switching!`](@ref), and
topology changing susceptance edits require rebuilding the problem so the JuMP
model and KKT layout keep the same reference constraints.

### ACNetwork

Stores the AC network with vectorized admittance representation.

| Field | Type | Description |
|-------|------|-------------|
| `n`, `m` | `Int` | Buses, branches |
| `A` | `SparseMatrixCSC` | Incidence matrix (m × n) |
| `incidences` | `Vector{Tuple}` | Edge list [(i,j), ...] (sequential indices) |
| `g`, `b` | `Vector{Float64}` | Branch conductances, susceptances |
| `g_shunt`, `b_shunt` | `Vector{Float64}` | Shunt admittances per bus |
| `sw` | `Vector{Float64}` | Switching states in [0,1] |
| `is_switchable` | `BitVector` | Which branches can be switched |
| `idx_slack` | `Int` | Slack bus index (sequential) |
| `vm_min`, `vm_max` | `Vector{Float64}` | Voltage magnitude limits per bus |
| `id_map` | `IDMapping` | Bidirectional element ID mapping (original ↔ sequential) |

## Sensitivity Caching

### DCSensitivityCache

The [`DCOPFProblem`](@ref) maintains a `DCSensitivityCache` that avoids redundant computation. Cached values include:

- `solution`: The last solved `DCOPFSolution`
- `kkt_factor`: LU factorization of the KKT Jacobian
- `dz_dd`, `dz_dsw`, `dz_dcl`, `dz_dcq`, `dz_dfmax`, `dz_db`: Full KKT derivative matrices

Calling `calc_sensitivity` with different operands for the same parameter reuses the cached KKT solve. For example, computing both `:va` and `:pg` w.r.t. `:d` only solves the KKT system once.

Cache invalidation happens automatically when `solve!`, `update_demand!`, `update_switching!`, or `update_fmax!` is called. Direct mutation of fields inside `prob.network` bypasses this contract.

### ACSensitivityCache

The [`ACOPFProblem`](@ref) maintains an `ACSensitivityCache` with:

- `solution`: The last solved `ACOPFSolution`
- `kkt_factor`: LU factorization of the KKT Jacobian
- `dz_dsw`, `dz_dd`, `dz_dqd`, `dz_dcq`, `dz_dcl`, `dz_dfmax`: Full KKT derivative matrices

All AC OPF operands (`:vm`, `:va`, `:pg`, `:qg`, `:lmp`, `:qlmp`) for the same parameter share a single cached `dz_d*` matrix. The KKT factorization is shared across all 6 parameter types.

## Solver Configuration

### DC OPF

Default solver is Ipopt. Override with any JuMP-compatible QP solver:

```julia
using HiGHS
prob = DCOPFProblem(dc_net, d; optimizer=HiGHS.Optimizer)
```

### AC OPF

The default `:jump` backend uses Ipopt. The opt-in CPU `:exa` backend uses
ExaModels and NLPModelsIpopt. Custom JuMP optimizer objects are accepted only
by `:jump`.

```julia
prob = ACOPFProblem(net; silent=true)
exa_prob = ACOPFProblem(net; backend=:exa, silent=true)
```

## KKT System Access (Qualified)

KKT internals are available via qualified access (`PowerDiff.function_name`), not exported:

```julia
using PowerDiff
const PD = PowerDiff

# DC OPF
z = PD.flatten_variables(sol, prob)     # Solution → vector
vars = PD.unflatten_variables(z, prob)  # Vector → named tuple
K = PD.kkt(z, prob, d)                  # KKT residuals
J = PD.calc_kkt_jacobian(prob)          # Sparse Jacobian dK/dz
dim = PD.kkt_dims(dc_net)              # KKT dimension
idx = PD.kkt_indices(dc_net)           # Named index ranges

# AC OPF — same unified API
z = PD.flatten_variables(sol, ac_prob)
J = PD.calc_kkt_jacobian(ac_prob)       # Sparse analytical Jacobian
dim = PD.kkt_dims(ac_prob)             # KKT dimension
idx = PD.kkt_indices(ac_prob)          # Named index ranges
```

## LMP Sign Conventions

DC OPF and AC OPF use **different LMP sign conventions** due to their constraint formulations. This is intentional and consistent within each formulation.

| Aspect | DC OPF | AC OPF |
|--------|--------|--------|
| **Power balance constraint** | `G*g + psh - d = B*θ` | `P_flow + P_d - P_g = 0` |
| **Demand sign in constraint** | Negative (subtracted) | Positive |
| **JuMP dual at optimum** | `ν_bal > 0` | `ν_p_bal < 0` |
| **`calc_lmp()` formula** | `return ν_bal` | `return -ν_p_bal` |
| **Sensitivity extraction** | `dz_dp[idx.nu_bal, :]` (no flip) | `-dz_dp[idx.nu_p_bal, :]` (negated) |

**Root cause:** The DC OPF constraint subtracts demand (`-d`), so increasing demand directly increases the dual. The AC OPF constraint adds demand (`+P_d`), so JuMP's Lagrangian `L = f - ν·h` produces a negative dual, requiring negation to get the positive marginal cost.

Both formulations produce **positive LMPs** at the API level: `calc_lmp()` and `calc_sensitivity(prob, :lmp, ...)` return values where a positive entry means "increasing demand at this bus increases cost."

See `src/sens/lmp.jl` for the authoritative sign convention documentation.
