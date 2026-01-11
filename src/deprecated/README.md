# Deprecated Files

These files contain legacy implementations that have been superseded by the new consolidated implementation in `src/opf/`.

## dcopf.jl
PTDF-based DC OPF formulation. Superseded by the B-theta formulation in `src/opf/problem.jl` which preserves graphical structure.

## dcopf_B_theta.jl
Original B-theta formulation hardcoded to Gurobi solver with ParametricOptInterface. Superseded by the cleaner, solver-agnostic implementation in `src/opf/problem.jl`.

## Migration Guide

The new implementation uses:
- `DCNetwork` instead of `PowerNetwork` or `DCPowerManagementProblem.params`
- `DCOPFProblem` instead of `PowerManagementProblem` or `DCPowerManagementProblem`
- `DCOPFSolution` for solution storage
- `solve!(prob)` returns a `DCOPFSolution` struct
- Ipopt as default solver (solver-agnostic design)
