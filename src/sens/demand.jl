# Demand Sensitivity Analysis for DC OPF
# Uses implicit differentiation via KKT conditions
#
# Note: For DCOPFProblem, demand sensitivities are computed via the cached
# KKT system in kkt_dc_opf.jl. The functions here provide convenience wrappers.

using LinearAlgebra
using SparseArrays

"""
    calc_generation_participation_factors(prob::DCOPFProblem)

Compute generation participation factors from demand sensitivity.

The participation factor for generator i at bus j is dg_i/dd_j,
representing how much generator i output changes when demand at bus j increases by 1 MW.

# Returns
Matrix (k x n) of participation factors.
"""
function calc_generation_participation_factors(prob::DCOPFProblem)
    return _extract_sensitivity(prob, _get_dz_dd!(prob), :pg)
end

"""
    calc_ptdf_from_sensitivity(prob::DCOPFProblem)

Compute Power Transfer Distribution Factors from flow sensitivity.

PTDF[e, j] = df_e/dd_j represents how much flow on line e changes
when power is injected at bus j (and withdrawn at the slack bus).

For the B-theta formulation, this is directly available from the sensitivity analysis.

# Returns
Matrix (m x n) of PTDFs.
"""
function calc_ptdf_from_sensitivity(prob::DCOPFProblem)
    return _extract_sensitivity(prob, _get_dz_dd!(prob), :f)
end
