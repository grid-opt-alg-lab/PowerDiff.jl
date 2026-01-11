# PowerModelsDiff.jl

A Julia package for differentiable power system analysis. Provides automatic differentiation capabilities for power flow equations, optimal power flow (OPF) problems, and sensitivity analysis of power networks.

## Features

- **DC OPF (B-theta formulation)**: Susceptance-weighted Laplacian formulation that preserves graphical structure
- **Implicit differentiation**: Compute sensitivities via KKT conditions without differentiating through the solver
- **LMP computation**: Locational Marginal Prices from dual variables
- **Demand sensitivity**: How OPF solutions change with demand perturbations
- **Topology sensitivity**: Sensitivities w.r.t. switching variables for topology optimization
- **AC power flow sensitivities**: Voltage sensitivity with respect to topology parameters

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/samtalki/PowerModelsDiff.jl")
```

## Quick Start

```julia
using PowerModelsDiff
using PowerModels

# Load a MATPOWER case
raw = PowerModels.parse_file("case14.m")
net = PowerModels.make_basic_network(raw)

# Create DC network and solve OPF
dc_net = DCNetwork(net)
d = calc_demand_vector(net)
prob = DCOPFProblem(dc_net, d)
sol = solve!(prob)

# Compute LMPs
lmps = calc_lmp(sol, dc_net)

# Compute demand sensitivities
sens = calc_sensitivity_demand(prob)

# Compute topology (switching) sensitivities
topo_sens = calc_sensitivity_switching(prob)
```

## Core Types

### DCNetwork
Network data for B-theta DC OPF formulation:
```julia
struct DCNetwork
    n::Int          # Number of buses
    m::Int          # Number of branches
    k::Int          # Number of generators
    A               # Branch incidence matrix (m x n)
    G_inc           # Generator-bus incidence matrix (n x k)
    b               # Branch susceptances
    z               # Switching states (1=closed, 0=open)
    fmax, gmax, gmin    # Flow and generation limits
    cq, cl          # Quadratic and linear cost coefficients
    ref_bus         # Reference bus index
    tau             # Regularization parameter
end
```

### DCOPFProblem
JuMP-based optimization problem:
```julia
mutable struct DCOPFProblem
    model::JuMP.Model
    network::DCNetwork
    theta           # Phase angle variables
    g               # Generation variables
    f               # Flow variables
    d               # Demand vector
    cons            # Constraint references
end
```

### DCOPFSolution
Solution container:
```julia
struct DCOPFSolution
    theta           # Optimal phase angles
    g               # Optimal generation
    f               # Optimal flows
    nu_bal          # Power balance duals (for LMPs)
    lambda_ub, lambda_lb  # Flow limit duals
    rho_ub, rho_lb  # Generation limit duals
    objective       # Optimal objective value
end
```

## Sensitivity Analysis

### Demand Sensitivity
Compute how the OPF solution changes with demand:

```julia
sens = calc_sensitivity_demand(prob)
# Returns DemandSensitivity with:
#   dtheta_dd: (n x n) sensitivity of angles to demand
#   dg_dd:     (k x n) sensitivity of generation to demand
#   df_dd:     (m x n) sensitivity of flows to demand
#   dlmp_dd:   (n x n) sensitivity of LMPs to demand
```

### Topology Sensitivity
Compute sensitivities w.r.t. switching variables:

```julia
topo_sens = calc_sensitivity_switching(prob)
# Returns TopologySensitivity with:
#   dtheta_ds: (n x m) sensitivity of angles to switching
#   dg_ds:     (k x m) sensitivity of generation to switching
#   df_ds:     (m x m) sensitivity of flows to switching
#   dlmp_ds:   (n x m) sensitivity of LMPs to switching
```

## KKT System

For advanced users, the package exposes the KKT system for custom sensitivity analysis:

```julia
# Flatten/unflatten primal-dual variables
z = flatten_variables(sol, prob)
vars = unflatten_variables(z, prob)

# Evaluate KKT residuals
K = kkt(z, prob, d)

# Compute analytical KKT Jacobian
J = calc_kkt_jacobian(prob)
```

## Solver Support

The default solver is Clarabel (interior-point conic solver). Other supported solvers:
- Ipopt (nonlinear)
- HiGHS (linear/quadratic)
- Gurobi (commercial)

```julia
using Ipopt
prob = DCOPFProblem(dc_net, d; optimizer=Ipopt.Optimizer)
```

## Dependencies

- [PowerModels.jl](https://github.com/lanl-ansi/PowerModels.jl): Power system modeling
- [JuMP.jl](https://github.com/jump-dev/JuMP.jl): Optimization modeling
- [Clarabel.jl](https://github.com/oxfordcontrol/Clarabel.jl): Default interior-point solver
- [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl): Automatic differentiation

## Mathematical Background

### B-theta Formulation
The DC OPF is formulated using the susceptance-weighted Laplacian:

```
min  (1/2) g' Cq g + cl' g + (tau^2/2) ||f||^2
s.t. G_inc * g - d = B * theta     (power balance)
     f = W * A * theta              (flow definition)
     -fmax <= f <= fmax             (flow limits)
     gmin <= g <= gmax              (gen limits)
     theta[ref] = 0                 (reference bus)
```

where:
- `B = A' * W * A` is the susceptance-weighted Laplacian
- `W = Diagonal(-b .* z)` includes switching variables
- `A` is the branch incidence matrix

### Implicit Differentiation
Sensitivities are computed via the implicit function theorem on KKT conditions:

```
dz/dp = -(dK/dz)^{-1} * (dK/dp)
```

where `z` is the primal-dual solution and `p` is the parameter (demand, switching, etc.).

## License

MIT License
