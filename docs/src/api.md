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
vjp
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
calc_demand_vector
calc_susceptance_matrix
calc_lmp
calc_congestion_component
calc_energy_component
calc_generation_participation_factors
calc_ptdf_from_sensitivity
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

## KKT System

```@docs
kkt
kkt_dims
kkt_indices
flatten_variables
unflatten_variables
calc_kkt_jacobian
ac_kkt_dims
ac_kkt_indices
ac_flatten_variables
ac_unflatten_variables
calc_ac_kkt_jacobian
ac_kkt
```

## Abstract Types

```@docs
AbstractPowerNetwork
AbstractPowerFlowState
AbstractOPFSolution
AbstractOPFProblem
```
