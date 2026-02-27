# Demand Sensitivity Analysis for DC OPF
# Uses implicit differentiation via KKT conditions
#
# Note: For DCOPFProblem, demand sensitivities are computed via the cached
# KKT system in kkt_dc_opf.jl. The functions here provide convenience wrappers.

"""
    calc_generation_participation_factors(prob::DCOPFProblem) → Sensitivity{Float64}

Compute generation participation factors from demand sensitivity.

The participation factor for generator i at bus j is dg_i/dd_j,
representing how much generator i output changes when demand at bus j increases by 1 MW.

# Returns
`Sensitivity{Float64}` (k × n) with formulation=:dcopf, operand=:pg, parameter=:d.
"""
function calc_generation_participation_factors(prob::DCOPFProblem)
    return calc_sensitivity(prob, :pg, :d)
end

"""
    calc_ptdf_from_sensitivity(prob::DCOPFProblem) → Sensitivity{Float64}

Compute Power Transfer Distribution Factors from flow sensitivity.

PTDF[e, j] = df_e/dd_j represents how much flow on line e changes
when power is injected at bus j (and withdrawn at the slack bus).

For the B-theta formulation, this is directly available from the sensitivity analysis.

# Returns
`Sensitivity{Float64}` (m × n) with formulation=:dcopf, operand=:f, parameter=:d.
"""
function calc_ptdf_from_sensitivity(prob::DCOPFProblem)
    return calc_sensitivity(prob, :f, :d)
end
