# Advanced Topics

## Type Hierarchy

```
AbstractPowerNetwork
├── DCNetwork           # DC B-theta formulation
└── ACNetwork           # AC with vectorized admittance

AbstractPowerFlowState
├── DCPowerFlowState    # DC power flow (θ_r = L_r \ p_r)
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
| `cq`, `cl` | `Vector{Float64}` | Cost coefficients (quadratic, linear) |
| `c_shed` | `Vector{Float64}` | Load-shedding cost per bus |
| `ref_bus` | `Int` | Reference bus index |
| `τ` | `Float64` | Regularization parameter |

Construct from PowerModels data: `DCNetwork(net)` or with explicit parameters: `DCNetwork(n, m, k, A, G_inc, b; ...)`.

### ACNetwork

Stores the AC network with vectorized admittance representation.

| Field | Type | Description |
|-------|------|-------------|
| `n`, `m` | `Int` | Buses, branches |
| `A` | `SparseMatrixCSC` | Incidence matrix |
| `incidences` | `Vector{Tuple}` | Edge list [(i,j), ...] |
| `g`, `b` | `Vector{Float64}` | Conductances, susceptances |
| `g_shunt`, `b_shunt` | `Vector{Float64}` | Shunt admittances |
| `sw` | `Vector{Float64}` | Switching states |

## Sensitivity Caching

### DCSensitivityCache

The [`DCOPFProblem`](@ref) maintains a `DCSensitivityCache` that avoids redundant computation. Cached values include:

- `solution`: The last solved `DCOPFSolution`
- `kkt_factor`: LU factorization of the KKT Jacobian
- `dz_dd`, `dz_dsw`, `dz_dcl`, `dz_dcq`, `dz_dfmax`, `dz_db`: Full KKT derivative matrices

Calling `calc_sensitivity` with different operands for the same parameter reuses the cached KKT solve. For example, computing both `:va` and `:pg` w.r.t. `:d` only solves the KKT system once.

Cache invalidation happens automatically when `solve!`, `update_demand!`, or `update_switching!` is called.

### ACSensitivityCache

The [`ACOPFProblem`](@ref) maintains an `ACSensitivityCache` with:

- `solution`: The last solved `ACOPFSolution`
- `dz_dsw`: Full KKT derivative matrix w.r.t. switching

All AC OPF operands (`:vm`, `:va`, `:pg`, `:qg`) share a single cached `dz_dsw` computation.

## Solver Configuration

### DC OPF

Default solver is Clarabel (interior-point conic). Override with:

```julia
using Ipopt
prob = DCOPFProblem(dc_net, d; optimizer=Ipopt.Optimizer)
```

### AC OPF

Default solver is Ipopt. The `silent` keyword suppresses solver output:

```julia
prob = ACOPFProblem(net; silent=true)
```

## KKT System Access

For advanced users, the KKT system is directly accessible:

```julia
# DC OPF
z = flatten_variables(sol, prob)     # Solution → vector
vars = unflatten_variables(z, prob)  # Vector → named tuple
K = kkt(z, prob, d)                  # KKT residuals
J = calc_kkt_jacobian(prob)          # Sparse Jacobian dK/dz

# AC OPF
z = ac_flatten_variables(sol, prob)
J = calc_ac_kkt_jacobian(prob)       # Dense Jacobian via ForwardDiff
```
