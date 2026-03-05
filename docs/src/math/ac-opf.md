# AC Optimal Power Flow

## Formulation

The AC OPF minimizes generation cost subject to the full nonlinear AC power flow equations in polar coordinates:

```math
\min_{\theta, |V|, p_g, q_g} \quad \sum_{i=1}^{k} \left(c_{q,i} \, p_{g,i}^2 + c_{l,i} \, p_{g,i} + c_{c,i}\right)
```

subject to:

### Power Balance (Equality)

```math
\begin{aligned}
\sum_{(l,i,j) \in \mathcal{A}_i} p_{lij} + g_{s,i} |V_i|^2 &= \sum_{g \in \mathcal{G}_i} p_{g,g} - p_{d,i} & (\nu_{p,i}) \\
\sum_{(l,i,j) \in \mathcal{A}_i} q_{lij} - b_{s,i} |V_i|^2 &= \sum_{g \in \mathcal{G}_i} q_{g,g} - q_{d,i} & (\nu_{q,i})
\end{aligned}
```

where ``\mathcal{A}_i`` are the arcs incident to bus ``i``, ``\mathcal{G}_i`` are the generators at bus ``i``, and ``g_{s,i}``, ``b_{s,i}`` are shunt conductance/susceptance.

### Reference Bus (Equality)

```math
\theta_{\text{ref}} = 0 \qquad (\nu_{\text{ref}})
```

### Branch Flow Equations

For each branch ``l`` from bus ``f`` to bus ``t`` with switching state ``\mathrm{sw}_l``:

```math
\begin{aligned}
p_{fr,l} &= \mathrm{sw}_l \left[\frac{g_l + g_{fr}^{sh}}{t_m^2} |V_f|^2 + \frac{-g_l t_r + b_l t_i}{t_m^2} |V_f||V_t| \cos(\theta_f - \theta_t) + \frac{-b_l t_r - g_l t_i}{t_m^2} |V_f||V_t| \sin(\theta_f - \theta_t)\right] \\
p_{to,l} &= \mathrm{sw}_l \left[(g_l + g_{to}^{sh}) |V_t|^2 + \frac{-g_l t_r - b_l t_i}{t_m^2} |V_t||V_f| \cos(\theta_t - \theta_f) + \frac{-b_l t_r + g_l t_i}{t_m^2} |V_t||V_f| \sin(\theta_t - \theta_f)\right]
\end{aligned}
```

where ``g_l + jb_l`` is the branch admittance, ``t_r + jt_i`` is the complex tap ratio, ``t_m^2 = \mathrm{tap}^2``, and ``g_{fr}^{sh}``, ``b_{fr}^{sh}`` are the from-side shunt elements of the pi-model. Reactive flow equations ``q_{fr,l}`` and ``q_{to,l}`` follow the same structure.

### Inequality Constraints

```math
\begin{aligned}
p_{fr,l}^2 + q_{fr,l}^2 &\leq r_l^2 & (\lambda_{\text{th},fr,l}) & \quad \text{Thermal limits (from)} \\
p_{to,l}^2 + q_{to,l}^2 &\leq r_l^2 & (\lambda_{\text{th},to,l}) & \quad \text{Thermal limits (to)} \\
\theta_f - \theta_t &\geq \alpha_{\min,l} & (\lambda_{\angle,lb,l}) & \quad \text{Angle difference bounds} \\
\theta_f - \theta_t &\leq \alpha_{\max,l} & (\lambda_{\angle,ub,l}) \\
V_{\min,i} \leq |V_i| &\leq V_{\max,i} & (\mu_{lb,i}, \mu_{ub,i}) & \quad \text{Voltage bounds} \\
p_{g,\min,i} \leq p_{g,i} &\leq p_{g,\max,i} & (\rho_{pg,lb,i}, \rho_{pg,ub,i}) & \quad \text{Generation bounds} \\
q_{g,\min,i} \leq q_{g,i} &\leq q_{g,\max,i} & (\rho_{qg,lb,i}, \rho_{qg,ub,i}) \\
-r_l \leq p_{fr,l} &\leq r_l & (\sigma_{p,fr,lb,l}, \sigma_{p,fr,ub,l}) & \quad \text{Flow variable bounds} \\
-r_l \leq q_{fr,l} &\leq r_l & (\sigma_{q,fr,lb,l}, \sigma_{q,fr,ub,l})
\end{aligned}
```

(with analogous bounds for the to-side flows ``p_{to,l}``, ``q_{to,l}``).

## Reduced-Space Formulation

The implementation uses a **reduced-space** formulation where branch flows ``p_{fr}``, ``q_{fr}``, ``p_{to}``, ``q_{to}`` are treated as functions of the voltage state ``(\theta, |V|)`` rather than separate primal variables. This means:

- The flow definition constraints are eliminated
- Stationarity conditions automatically include all chain-rule terms via ForwardDiff on the reduced Lagrangian
- Flow bound complementary slackness uses the computed flow expressions

## KKT System

### Variable Ordering

The KKT variable vector ``z`` is structured as:

```math
z = [\theta, |V|, p_g, q_g, \; \nu_p, \nu_q, \nu_{\text{ref}}, \; \lambda_{\text{th},fr}, \lambda_{\text{th},to}, \lambda_{\angle,lb}, \lambda_{\angle,ub}, \; \mu_{lb}, \mu_{ub}, \; \rho_{pg,lb}, \rho_{pg,ub}, \rho_{qg,lb}, \rho_{qg,ub}, \; \sigma_{p,fr,lb}, \ldots, \sigma_{q,to,ub}]
```

with total dimension ``6n + 12m + 6k + n_{\text{ref}}``, where ``n`` is the number of buses, ``m`` is the number of branches, ``k`` is the number of generators, and ``n_{\text{ref}}`` is the number of reference buses.

### KKT Conditions

1. **Stationarity** (``2n + 2k`` conditions): Computed via `ForwardDiff.gradient` on the reduced-space Lagrangian ``\mathcal{L}(\theta, |V|, p_g, q_g)`` which automatically handles chain-rule terms through flow expressions.

2. **Primal feasibility** (``2n + n_{\text{ref}}`` conditions):
   - Active power balance at each bus
   - Reactive power balance at each bus
   - Reference bus angle constraint

3. **Complementary slackness** (``4m + 2n + 4k + 8m`` conditions):
   - Thermal limits: ``\lambda_{\text{th}} \cdot (p^2 + q^2 - r^2) = 0``
   - Angle differences: ``\lambda_\angle \cdot (\theta_f - \theta_t - \alpha) = 0``
   - Voltage bounds: ``\mu \cdot (|V| - V_{\text{bound}}) = 0``
   - Generation bounds: ``\rho \cdot (x_g - x_{\text{bound}}) = 0``
   - Flow variable bounds: ``\sigma \cdot (f \pm r) = 0``

## Implicit Differentiation

### KKT Jacobian

The KKT Jacobian ``\partial K / \partial z`` is computed via ForwardDiff (dense matrix). This is applied to the KKT operator which includes stationarity via the reduced-space Lagrangian, making it self-consistent with the flow chain-rule terms.

### Parameter Jacobians

For each parameter ``p``, the parameter Jacobian ``\partial K / \partial p`` is also computed via ForwardDiff. The parameter is injected into the KKT operator through keyword arguments, and ForwardDiff propagates dual numbers through the entire computation.

The full derivative is:

```math
\frac{dz}{dp} = -\left(\frac{\partial K}{\partial z}\right)^{-1} \frac{\partial K}{\partial p}
```

## Supported Parameters

The AC OPF supports sensitivity w.r.t. 6 parameter types:

| Symbol | Parameter | Dimension | How it enters the KKT system |
|--------|-----------|-----------|------------------------------|
| `:sw` | Switching state | ``m`` | Multiplies all branch flow expressions |
| `:d` | Active demand | ``n`` | Enters power balance constraints |
| `:qd` | Reactive demand | ``n`` | Enters reactive power balance constraints |
| `:cq` | Quadratic gen cost | ``k`` | Enters objective (stationarity conditions) |
| `:cl` | Linear gen cost | ``k`` | Enters objective (stationarity conditions) |
| `:fmax` | Flow limits (rate_a) | ``m`` | Enters thermal limits and flow bound constraints |

Each parameter type requires its own ``\partial K / \partial p`` computation but shares the same KKT Jacobian factorization.

## Caching Strategy

The `ACSensitivityCache` implements a two-level caching hierarchy:

1. **KKT factorization** (shared across all parameters): One LU factorization of ``\partial K / \partial z`` is computed once and reused for all parameter types.

2. **Parameter derivatives** (shared across operands): For each parameter ``p``, the full ``dz/dp`` matrix is computed once. Different operand queries (e.g., `:va`, `:vm`, `:pg`, `:qg`, `:lmp`) simply extract different row blocks from the same cached matrix.

This means that querying all 5 operands for the same parameter costs essentially the same as querying 1 operand. And querying a second parameter type only requires the ``\partial K / \partial p`` computation (the expensive KKT factorization is reused).

## Operand Extraction

Given the full derivative ``dz/dp``, individual operand sensitivities are extracted by selecting the appropriate row indices:

| Operand | KKT rows | Description |
|---------|----------|-------------|
| `:va` | ``1 \ldots n`` | Voltage angles |
| `:vm` | ``n{+}1 \ldots 2n`` | Voltage magnitudes |
| `:pg` | ``2n{+}1 \ldots 2n{+}k`` | Active generation |
| `:qg` | ``2n{+}k{+}1 \ldots 2n{+}2k`` | Reactive generation |
| `:lmp` | ``2n{+}2k{+}1 \ldots 3n{+}2k`` | Active power balance duals (``\nu_p``) |

## Supported Combinations

The AC OPF supports all 30 operand × parameter combinations:

| | `:sw` | `:d` | `:qd` | `:cq` | `:cl` | `:fmax` |
|------|-------|------|--------|--------|--------|---------|
| `:va` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `:vm` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `:pg` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `:qg` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `:lmp` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## LMP Computation

In the AC OPF, locational marginal prices are the active power balance duals ``\nu_{p,i}``. These capture the marginal cost of serving an additional unit of active demand at bus ``i``, accounting for network losses, congestion, and voltage constraints.

```julia
lmps = calc_sensitivity(ac_prob, :lmp, :d)   # dLMP/dd
```

## Solver

The AC OPF uses [Ipopt](https://github.com/coin-or/Ipopt) as the default nonlinear programming solver, accessed via JuMP. Clarabel (a convex solver) is explicitly rejected since AC OPF is nonconvex.
