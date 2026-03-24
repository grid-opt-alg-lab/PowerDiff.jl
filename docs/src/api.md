# API Reference

## Sensitivity Interface

```@docs
calc_sensitivity
calc_sensitivity_column
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
calc_qlmp
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

## Utilities

```@docs
silence
```

## Abstract Types

```@docs
AbstractPowerNetwork
AbstractPowerFlowState
AbstractOPFSolution
AbstractOPFProblem
```

## AcceleratedDCPowerFlows Extension

```@docs
to_apf_network
apf_ptdf
apf_lodf
compare_ptdf
materialize_apf_ptdf
```
