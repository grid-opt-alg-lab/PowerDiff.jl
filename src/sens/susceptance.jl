# Susceptance Sensitivity Analysis for DC OPF
# Uses implicit differentiation via KKT conditions
#
# Note: For DCOPFProblem, susceptance sensitivities are computed via the cached
# KKT system in kkt_dc_opf.jl. This file contains the KKT Jacobian function
# for susceptance parameters.

"""
    calc_kkt_jacobian_susceptance(prob::DCOPFProblem, sol::DCOPFSolution)

Compute the Jacobian of KKT conditions with respect to susceptances dK/db.

# Arguments
- `prob`: DCOPFProblem
- `sol`: Pre-computed solution

# Returns
Sparse matrix of size (kkt_dims x m).

# Notes
Susceptance b affects:
- Susceptance matrix: B = A' * Diag(-b .* z) * A
- Weight matrix: W = Diag(-b .* z)
- Flow definition: f = W * A * theta

The affected KKT conditions are:
- K_theta = B' * nu_bal + (WA)' * nu_flow + e_ref * eta_ref
- K_power_bal = G_inc * g - d - B * theta
- K_flow_def = f - W * A * theta

Derivatives:
- dB/db_e = -z_e * A[e,:]' * A[e,:]
- d(WA)/db_e: row e becomes -z_e * A[e,:]
"""
function calc_kkt_jacobian_susceptance(prob::DCOPFProblem, sol::DCOPFSolution)
    net = prob.network
    n, m, k = net.n, net.m, net.k
    dim = kkt_dims(net)
    idx = kkt_indices(n, m, k)

    theta = sol.θ
    nu_bal = sol.ν_bal
    nu_flow = sol.ν_flow

    # Network parameters
    z = net.z
    A = net.A

    J_b = spzeros(dim, m)

    for e in 1:m
        # For each branch e, compute dK/db_e
        A_e = A[e, :]
        A_e_vec = Vector(A_e[:])  # Convert to dense vector
        Atheta_e = (A * theta)[e]  # Phase angle difference across branch e

        # 1. dK_theta/db_e: K_theta = B' * nu_bal + (WA)' * nu_flow + e_ref * eta_ref
        # dB'/db_e = -z_e * A[e,:]' * A[e,:] (symmetric)
        # d(WA')/db_e: column e becomes -z_e * A[e,:]'

        # Contribution from nu_bal through B:
        # dB/db_e * nu_bal = -z_e * A[e,:]' * (A[e,:] . nu_bal)
        Ae_dot_nu_bal = dot(A_e_vec, nu_bal)
        dK_theta_from_nu_bal = -z[e] * A_e_vec * Ae_dot_nu_bal

        # Contribution from nu_flow through WA':
        # d(WA')/db_e * nu_flow = -z_e * A[e,:]' * nu_flow[e]
        dK_theta_from_nu_flow = -z[e] * A_e_vec * nu_flow[e]

        J_b[idx.θ, e] = dK_theta_from_nu_bal + dK_theta_from_nu_flow

        # 2. dK_power_bal/db_e: K_power_bal = G_inc * g - d - B * theta
        # dK_power_bal/db_e = -dB/db_e * theta = -(-z_e * A[e,:]' * A[e,:]) * theta
        #                    = z_e * A[e,:]' * (A[e,:] . theta)
        #                    = z_e * A_e_vec * Atheta_e
        dK_power_bal_db_e = z[e] * A_e_vec * Atheta_e
        J_b[idx.ν_bal, e] = dK_power_bal_db_e

        # 3. dK_flow_def/db_e: K_flow_def = f - W * A * theta
        # dK_flow_def/db_e = -d(WA)/db_e * theta
        # d(WA)/db_e * theta: row e is -z_e * A[e,:] * theta = -z_e * Atheta_e
        # So dK_flow_def/db_e: row e is z_e * Atheta_e (note the sign flip)
        dK_flow_def_db_e = spzeros(m)
        dK_flow_def_db_e[e] = z[e] * Atheta_e
        J_b[idx.ν_flow, e] = dK_flow_def_db_e
    end

    return J_b
end
