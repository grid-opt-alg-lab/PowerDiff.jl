# API Reference

## Sensitivity Interface

```@docs
calc_sensitivity
Sensitivity
```

## ID Mapping

```@docs
IDMapping
```

## JVP / VJP

```@docs
jvp
jvp!
vjp
vjp!
kkt_dims
dict_to_vec
vec_to_dict
```

## Introspection

```@docs
operand_symbols
parameter_symbols
```

## DC Types

```@docs
DCNetwork
DCPowerFlowState
DCOPFProblem
DCOPFSolution
DCSensitivityCache
```

## DC Functions

```@docs
solve!
update_demand!
update_switching!
calc_demand_vector
calc_susceptance_matrix
calc_lmp
calc_congestion_component
calc_energy_component
calc_generation_participation_factors
calc_ptdf_from_sensitivity
ptdf_matrix
invalidate!
```

## AC Types

```@docs
ACNetwork
ACPowerFlowState
ACOPFProblem
ACOPFSolution
ACSensitivityCache
```

## AC Functions

```@docs
admittance_matrix
branch_current
branch_power
calc_power_flow_jacobian
```

## Abstract Types

```@docs
AbstractPowerNetwork
AbstractPowerFlowState
AbstractOPFSolution
AbstractOPFProblem
```
