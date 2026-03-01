# Mathematical Background

## B-theta Formulation

The DC OPF uses the susceptance-weighted Laplacian to model power flow:

```math
\min_{g, \theta, f, \text{psh}} \quad g^\top C_q g + c_l^\top g + c_{\text{shed}}^\top \text{psh} + \frac{\tau^2}{2} \|f\|^2
```

subject to:

```math
\begin{aligned}
G_{\text{inc}} g + \text{psh} - d &= L \theta & (\nu_{\text{bal}}) \\
f &= W A \theta & (\nu_{\text{flow}}) \\
-f_{\max} \leq f &\leq f_{\max} & (\lambda_{\text{lb}}, \lambda_{\text{ub}}) \\
g_{\min} \leq g &\leq g_{\max} & (\rho_{\text{lb}}, \rho_{\text{ub}}) \\
0 \leq \text{psh} &\leq d & (\mu_{\text{lb}}, \mu_{\text{ub}}) \\
\theta_{\text{ref}} &= 0 & (\eta_{\text{ref}})
\end{aligned}
```

where:
- ``L = A^\top \text{diag}(-b \circ \text{sw}) A`` is the susceptance-weighted Laplacian
- ``W = \text{diag}(-b \circ \text{sw})`` is the branch weight matrix
- ``A`` is the ``m \times n`` incidence matrix (branches × buses)
- ``G_{\text{inc}}`` is the ``n \times k`` generator-bus incidence matrix
- ``c_{\text{shed}}`` is the load-shedding cost vector
- ``\tau`` is a small regularization parameter

## DC Power Flow

For non-OPF power flow with fixed generation, the voltage angles satisfy the
reduced system obtained by eliminating the reference (slack) bus:

```math
\theta_r = L_r^{-1} \, p_r
```

where ``L_r`` is the Laplacian with the reference bus row and column deleted
(invertible for a connected network), ``p_r`` is the net injection with the
reference entry removed, and ``\theta_{\text{ref}} = 0``.

Switching sensitivity follows from matrix perturbation theory:

```math
\frac{\partial \theta_r}{\partial \text{sw}_e}
  = -L_r^{-1} \frac{\partial L_r}{\partial \text{sw}_e} \, \theta_r
```

where ``\frac{\partial L_r}{\partial \text{sw}_e} = -b_e \, a_{e,r} \, a_{e,r}^\top``
is a rank-1 update from the incidence column of branch ``e`` restricted to non-reference buses.

## KKT System for Implicit Differentiation

OPF sensitivities are computed via the implicit function theorem applied to the KKT conditions. At an optimal solution ``z^*``, the KKT residual ``K(z^*, p) = 0`` where ``z`` collects all primal and dual variables and ``p`` is a parameter.

By the implicit function theorem:

```math
\frac{dz}{dp} = -\left(\frac{\partial K}{\partial z}\right)^{-1} \frac{\partial K}{\partial p}
```

### DC OPF KKT Variable Ordering

The KKT variable vector ``z`` is structured as:

```math
z = [\theta, g, f, \text{psh}, \lambda_{\text{lb}}, \lambda_{\text{ub}}, \rho_{\text{lb}}, \rho_{\text{ub}}, \mu_{\text{lb}}, \mu_{\text{ub}}, \nu_{\text{bal}}, \nu_{\text{flow}}, \eta_{\text{ref}}]
```

with total dimension ``5n + 4m + 3k + 1``.

The KKT Jacobian ``\partial K / \partial z`` is computed analytically as a sparse matrix. Parameter Jacobians ``\partial K / \partial p`` are computed for each parameter type (demand, switching, cost, flow limits, susceptances).

### AC OPF

The AC OPF uses the full nonlinear polar power flow equations with Ipopt as the solver. The KKT Jacobian is computed via ForwardDiff, and sensitivities are extracted for switching parameters via the same implicit differentiation framework.

## AC Admittance

The AC admittance matrix is:

```math
Y = A^\top \text{diag}(g + jb) A + \text{diag}(g_{\text{sh}} + jb_{\text{sh}})
```

where ``g`` and ``b`` are branch conductances and susceptances, and ``g_{\text{sh}}``, ``b_{\text{sh}}`` are shunt admittances.

## LMP Decomposition

Locational marginal prices are the power balance duals ``\nu_{\text{bal}}``, decomposed as:

```math
\text{LMP} = \text{energy} + \text{congestion}
```

where the energy component reflects the marginal cost of generation and the congestion component captures price differentiation due to binding flow constraints.
