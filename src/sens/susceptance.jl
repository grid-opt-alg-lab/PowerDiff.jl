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
- Susceptance matrix: B = A' * Diag(-b .* sw) * A
- Weight matrix: W = Diag(-b .* sw)
- Flow definition: f = W * A * theta

The affected KKT conditions are:
- K_theta = B' * nu_bal + (WA)' * nu_flow + e_ref * eta_ref
- K_power_bal = G_inc * g + psh - d - B * theta
- K_flow_def = f - W * A * theta

Derivatives:
- dB/db_e = -sw_e * A[e,:]' * A[e,:]
- d(WA)/db_e: row e becomes -sw_e * A[e,:]
"""
function calc_kkt_jacobian_susceptance(prob::DCOPFProblem, sol::DCOPFSolution)
    net = prob.network
    n, m, k = net.n, net.m, net.k
    dim = kkt_dims(net)
    idx = kkt_indices(n, m, k)

    theta = sol.va
    nu_bal = sol.nu_bal
    nu_flow = sol.nu_flow

    # Network parameters
    sw = net.sw
    A = net.A

    J_b = spzeros(dim, m)

    # Precompute A * theta once (invariant across branches)
    Atheta = A * theta

    for e in 1:m
        A_e_vec = Vector(A[e, :])
        Atheta_e = Atheta[e]

        # dK_theta/db_e = -sw_e * A[e,:]' * (dot(A[e,:], nu_bal) + nu_flow[e])
        Ae_dot_nu_bal = dot(A_e_vec, nu_bal)
        J_b[idx.va, e] = -sw[e] * A_e_vec * (Ae_dot_nu_bal + nu_flow[e])

        # dK_power_bal/db_e = sw_e * A[e,:]' * (A * theta)[e]
        J_b[idx.nu_bal, e] = sw[e] * A_e_vec * Atheta_e

        # dK_flow_def/db_e: only row e is nonzero
        J_b[idx.nu_flow[e], e] = sw[e] * Atheta_e
    end

    return J_b
end
