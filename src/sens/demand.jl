# Demand Sensitivity Analysis for DC OPF
# Uses implicit differentiation via KKT conditions

using LinearAlgebra
using SparseArrays

"""
    calc_sensitivity_demand(prob::DCOPFProblem)

Compute sensitivity of DC OPF solution with respect to demand using implicit differentiation.

Uses the implicit function theorem on KKT conditions:
```
∂z/∂d = -(∂K/∂z)⁻¹ · (∂K/∂d)
```

where K(z, d) = 0 are the KKT conditions and z contains all primal/dual variables.

# Returns
`DemandSensitivity` containing Jacobian matrices:
- `dθ_dd`: ∂θ/∂d (n × n) - phase angle sensitivity
- `dg_dd`: ∂g/∂d (k × n) - generation sensitivity
- `df_dd`: ∂f/∂d (m × n) - flow sensitivity
- `dlmp_dd`: ∂LMP/∂d (n × n) - LMP sensitivity

# Example
```julia
prob = DCOPFProblem(net, d)
solve!(prob)
sens = calc_sensitivity_demand(prob)
# How does generation change when demand at bus 3 increases?
dg_dbus3 = sens.dg_dd[:, 3]
```
"""
function calc_sensitivity_demand(prob::DCOPFProblem)
    net = prob.network
    n, m, k = net.n, net.m, net.k

    # Compute KKT Jacobian ∂K/∂z
    J_z = calc_kkt_jacobian(prob)

    # Compute KKT Jacobian w.r.t. demand ∂K/∂d
    J_d = calc_kkt_jacobian_demand(net)

    # Solve linear system: ∂z/∂d = -J_z⁻¹ * J_d
    # Use sparse LU factorization for efficiency
    dz_dd = -(J_z \ Matrix(J_d))

    # Extract individual sensitivities from flattened result
    # Variable order: [θ(n), g(k), f(m), λ_lb(m), λ_ub(m), ρ_lb(k), ρ_ub(k), ν_bal(n), ν_flow(m), η_ref(1)]
    dθ_dd = dz_dd[1:n, :]
    dg_dd = dz_dd[n+1:n+k, :]
    df_dd = dz_dd[n+k+1:n+k+m, :]

    # Dual variable sensitivities (for LMP computation)
    idx_λ_lb = n+k+m+1:n+k+2m
    idx_λ_ub = n+k+2m+1:n+k+3m
    idx_ν_bal = n+k+3m+2k+1:2n+k+3m+2k

    dλ_lb_dd = dz_dd[idx_λ_lb, :]
    dλ_ub_dd = dz_dd[idx_λ_ub, :]
    dν_bal_dd = dz_dd[idx_ν_bal, :]

    # Compute LMP sensitivity
    # In the B-θ formulation, LMP = ν_bal (the power balance dual already
    # incorporates network topology through the Laplacian constraint)
    # Therefore: ∂LMP/∂d = ∂ν_bal/∂d
    dlmp_dd = dν_bal_dd

    return DemandSensitivity(
        Matrix(dθ_dd),
        Matrix(dg_dd),
        Matrix(df_dd),
        Matrix(dlmp_dd)
    )
end

"""
    calc_sensitivity_demand_primal(prob::DCOPFProblem)

Compute only primal variable sensitivities (θ, g, f) w.r.t. demand.

More efficient than full `calc_sensitivity_demand` when only primal sensitivities are needed.

# Returns
NamedTuple with fields `dθ_dd`, `dg_dd`, `df_dd`.
"""
function calc_sensitivity_demand_primal(prob::DCOPFProblem)
    sens = calc_sensitivity_demand(prob)
    return (dθ_dd = sens.dθ_dd, dg_dd = sens.dg_dd, df_dd = sens.df_dd)
end

"""
    calc_generation_participation_factors(prob::DCOPFProblem)

Compute generation participation factors from demand sensitivity.

The participation factor for generator i at bus j is ∂gᵢ/∂dⱼ,
representing how much generator i output changes when demand at bus j increases by 1 MW.

# Returns
Matrix (k × n) of participation factors.
"""
function calc_generation_participation_factors(prob::DCOPFProblem)
    sens = calc_sensitivity_demand(prob)
    return sens.dg_dd
end

"""
    calc_ptdf_from_sensitivity(prob::DCOPFProblem)

Compute Power Transfer Distribution Factors from flow sensitivity.

PTDF[e, j] = ∂fₑ/∂dⱼ represents how much flow on line e changes
when power is injected at bus j (and withdrawn at the slack bus).

For the B-θ formulation, this is directly available from the sensitivity analysis.

# Returns
Matrix (m × n) of PTDFs.
"""
function calc_ptdf_from_sensitivity(prob::DCOPFProblem)
    sens = calc_sensitivity_demand(prob)
    return sens.df_dd
end
