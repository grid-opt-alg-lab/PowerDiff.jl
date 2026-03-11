# DC Optimal Power Flow

## B-theta Formulation

The DC OPF minimizes generation cost subject to linearized power flow constraints using the susceptance-weighted Laplacian:

```math
\min_{g, \theta, f, \text{psh}} \quad g^\top C_q g + c_l^\top g + c_{\text{shed}}^\top \text{psh} + \frac{\tau^2}{2} \|f\|^2
```

subject to:

```math
\begin{aligned}
G_{\text{inc}} g + \text{psh} - d &= B \theta & (\nu_{\text{bal}}) \\
f &= W A \theta & (\nu_{\text{flow}}) \\
-f_{\max} \leq f &\leq f_{\max} & (\lambda_{\text{lb}}, \lambda_{\text{ub}}) \\
g_{\min} \leq g &\leq g_{\max} & (\rho_{\text{lb}}, \rho_{\text{ub}}) \\
0 \leq \text{psh} &\leq d & (\mu_{\text{lb}}, \mu_{\text{ub}}) \\
\alpha_{\min} \leq A\theta &\leq \alpha_{\max} & (\gamma_{\text{lb}}, \gamma_{\text{ub}}) \\
\theta_{\text{ref}} &= 0 & (\eta_{\text{ref}})
\end{aligned}
```

where:
- ``B = A^\top \operatorname{diag}(-b \circ \mathrm{sw}) \, A`` is the susceptance-weighted Laplacian
- ``W = \operatorname{diag}(-b \circ \mathrm{sw})`` is the branch weight matrix
- ``A`` is the ``m \times n`` incidence matrix (branches × buses)
- ``G_{\text{inc}}`` is the ``n \times k`` generator-bus incidence matrix
- ``C_q = \operatorname{diag}(c_q)`` contains quadratic cost coefficients
- ``c_l`` contains linear cost coefficients
- ``c_{\text{shed}}`` is the load-shedding cost vector
- ``\tau`` is a small regularization parameter for numerical conditioning

## KKT System for Implicit Differentiation

OPF sensitivities are computed via the implicit function theorem applied to the KKT conditions. At an optimal solution ``z^*``, the KKT residual satisfies ``K(z^*, p) = 0`` where ``z`` collects all primal and dual variables and ``p`` is a parameter.

By the implicit function theorem:

```math
\frac{dz}{dp} = -\left(\frac{\partial K}{\partial z}\right)^{-1} \frac{\partial K}{\partial p}
```

### KKT Variable Ordering

The KKT variable vector ``z`` is structured as:

```math
z = [\theta, g, f, \text{psh}, \lambda_{\text{lb}}, \lambda_{\text{ub}}, \gamma_{\text{lb}}, \gamma_{\text{ub}}, \rho_{\text{lb}}, \rho_{\text{ub}}, \mu_{\text{lb}}, \mu_{\text{ub}}, \nu_{\text{bal}}, \nu_{\text{flow}}, \eta_{\text{ref}}]
```

with total dimension ``5n + 6m + 3k + 1``.

### KKT Conditions

The KKT residual ``K(z, p)`` consists of:

1. **Stationarity w.r.t. ``\theta``**: ``B^\top \nu_{\text{bal}} + (WA)^\top \nu_{\text{flow}} + e_{\text{ref}} \eta_{\text{ref}} + A^\top (\gamma_{\text{ub}} - \gamma_{\text{lb}}) = 0``
2. **Stationarity w.r.t. ``g``**: ``2 C_q g + c_l - G_{\text{inc}}^\top \nu_{\text{bal}} - \rho_{\text{lb}} + \rho_{\text{ub}} = 0``
3. **Stationarity w.r.t. ``f``**: ``\tau^2 f - \nu_{\text{flow}} - \lambda_{\text{lb}} + \lambda_{\text{ub}} = 0``
4. **Stationarity w.r.t. psh**: ``c_{\text{shed}} - \nu_{\text{bal}} - \mu_{\text{lb}} + \mu_{\text{ub}} = 0``
5. **Complementary slackness (flow bounds)**: ``\lambda_{\text{lb}} \circ (f + f_{\max}) = 0``, ``\lambda_{\text{ub}} \circ (f_{\max} - f) = 0``
5b. **Complementary slackness (angle differences)**: ``\gamma_{\text{lb}} \circ (A\theta - \alpha_{\min}) = 0``, ``\gamma_{\text{ub}} \circ (\alpha_{\max} - A\theta) = 0``
5c. **Complementary slackness (generation/shedding bounds)**: ``\rho \circ (\cdot) = 0``, ``\mu \circ (\cdot) = 0``
6. **Primal feasibility**: ``G_{\text{inc}} g + \text{psh} - d - B\theta = 0``
7. **Flow definition**: ``f - WA\theta = 0``
8. **Reference bus**: ``\theta_{\text{ref}} = 0``

### Analytical Sparse KKT Jacobian

The KKT Jacobian ``\partial K / \partial z`` is computed analytically as a sparse matrix, which is more efficient than automatic differentiation for large problems.

## Parameter Derivatives

For each parameter type, we compute ``\partial K / \partial p`` and then the full derivative via the IFT formula.

### Demand (``d``)

Demand enters the power balance and the upper shedding bound:

```math
\frac{\partial K_{\nu_{\text{bal}}}}{\partial d} = -I, \qquad
\frac{\partial K_{\mu_{\text{ub}}}}{\partial d} = \operatorname{diag}(\mu_{\text{ub}})
```

### Switching (``\mathrm{sw}``)

Switching affects the Laplacian ``B``, weight matrix ``W``, and flow definition through:

```math
\frac{\partial B}{\partial \mathrm{sw}_e} = -b_e \, a_e a_e^\top, \qquad
\frac{\partial W}{\partial \mathrm{sw}_e} = -b_e \, e_e e_e^\top
```

This propagates into the stationarity, power balance, and flow definition blocks of the KKT system.

### Cost Coefficients (``c_q``, ``c_l``)

Quadratic cost ``c_q`` enters stationarity w.r.t. ``g``:
```math
\frac{\partial K_g}{\partial c_{q,i}} = 2 g_i e_i
```

Linear cost ``c_l`` enters stationarity w.r.t. ``g``:
```math
\frac{\partial K_g}{\partial c_l} = I_k
```

### Flow Limits (``f_{\max}``)

Flow limits enter the complementary slackness conditions:
```math
\frac{\partial K_{\lambda_{\text{lb}}}}{\partial f_{\max}} = -\operatorname{diag}(\lambda_{\text{lb}}), \qquad
\frac{\partial K_{\lambda_{\text{ub}}}}{\partial f_{\max}} = \operatorname{diag}(\lambda_{\text{ub}})
```

### Susceptances (``b``)

Susceptances affect the same blocks as switching (through ``B`` and ``W``), but with different partial derivatives since ``B = A^\top \operatorname{diag}(-b \circ \mathrm{sw}) A``.

## LMP Decomposition

Locational marginal prices are the power balance duals ``\nu_{\text{bal}}``, decomposed as:

```math
\text{LMP} = \underbrace{\text{energy}}_{\text{uniform component}} + \underbrace{\text{congestion}}_{\text{flow-limit component}}
```

The congestion component is extracted by solving:

```math
\text{congestion}[\text{non-ref}] = B_r^{-1} \left(A_r^\top W (\lambda_{\text{ub}}^{\text{std}} - \lambda_{\text{lb}}^{\text{std}}) + A_r^\top (\gamma_{\text{ub}}^{\text{std}} - \gamma_{\text{lb}}^{\text{std}})\right)
```

where ``\lambda^{\text{std}}``, ``\gamma^{\text{std}}`` use the standard sign convention (non-negative for binding constraints). The ``\gamma`` terms capture congestion from binding phase angle difference limits. The energy component is uniform across all buses in a connected network and reflects the marginal cost of generation.
