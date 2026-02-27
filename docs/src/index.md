# PowerModelsDiff.jl

A Julia package for differentiable power system analysis. Compute sensitivities of power flow solutions, optimal power flow dispatch, and locational marginal prices with respect to network parameters.

## Features

- **Unified sensitivity API**: `calc_sensitivity(state, :operand, :parameter)` returns `Sensitivity{T}` matrices with metadata
- **DC OPF (B-theta formulation)**: Susceptance-weighted Laplacian preserving network topology
- **DC power flow sensitivities**: Switching and demand sensitivity for non-OPF power flow
- **AC power flow sensitivities**: Voltage and current sensitivity w.r.t. power injections
- **AC OPF sensitivities**: Switching sensitivity via implicit differentiation of KKT conditions
- **LMP computation**: Locational marginal prices with energy/congestion decomposition
- **ForwardDiff verification**: All sensitivities verified against automatic differentiation

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/samtalki/PowerModelsDiff.jl")
```

## Quick Example

```julia
using PowerModelsDiff, PowerModels

# Load network
net = make_basic_network(parse_file("case14.m"))
dc_net = DCNetwork(net)
d = calc_demand_vector(net)

# DC OPF with sensitivity analysis
prob = DCOPFProblem(dc_net, d)
solve!(prob)

dlmp_dd = calc_sensitivity(prob, :lmp, :d)   # dLMP/dd (n x n)
dpg_dsw = calc_sensitivity(prob, :pg, :sw)   # dg/dsw (k x m)

dlmp_dd.formulation  # :dcopf
dlmp_dd.operand      # :lmp
dlmp_dd[2, 3]        # dLMP_2 / dd_3
```

See [Getting Started](@ref) for a full walkthrough.

## Contents

```@contents
Pages = [
    "getting-started.md",
    "sensitivity-api.md",
    "math.md",
    "advanced.md",
    "api.md",
]
Depth = 1
```
