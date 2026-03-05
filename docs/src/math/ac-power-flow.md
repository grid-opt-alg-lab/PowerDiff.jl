# AC Power Flow

## Admittance Matrix

The bus admittance matrix in rectangular form is:

```math
Y = A^\top \operatorname{diag}(g + jb) \, A + \operatorname{diag}(g_{\text{sh}} + jb_{\text{sh}})
```

where ``g`` and ``b`` are branch conductances and susceptances, and ``g_{\text{sh}}``, ``b_{\text{sh}}`` are shunt admittances.

## Power Injection Equations

In polar coordinates, the power injections at bus ``i`` are:

```math
\begin{aligned}
P_i &= \sum_k |V_i| |V_k| \bigl(G_{ik} \cos(\theta_i - \theta_k) + B_{ik} \sin(\theta_i - \theta_k)\bigr) \\
Q_i &= \sum_k |V_i| |V_k| \bigl(G_{ik} \sin(\theta_i - \theta_k) - B_{ik} \cos(\theta_i - \theta_k)\bigr)
\end{aligned}
```

where ``G_{ik} + jB_{ik} = Y_{ik}`` is the ``(i,k)`` element of the admittance matrix.

In compact notation using complex voltages ``V = |V| e^{j\theta}``:

```math
P + jQ = V \cdot \overline{Y V} = \overline{\bar{V} Y V}
```

so ``P = \operatorname{Re}(\bar{V} Y V)`` and ``Q = -\operatorname{Im}(\bar{V} Y V)`` (standard convention).

## Newton-Raphson Jacobian

The power flow Jacobian used in Newton-Raphson iteration has four blocks in rectangular coordinates:

```math
J = \begin{bmatrix} \partial P / \partial v_{\text{re}} & \partial P / \partial v_{\text{im}} \\ \partial Q / \partial v_{\text{re}} & \partial Q / \partial v_{\text{im}} \end{bmatrix}
```

The Jacobian is built from the admittance matrix ``Y`` and bus current injections ``I = Y V`` at the operating point, with the slack bus row and column deleted.

## Voltage-Power Sensitivity

The voltage sensitivity to power injections is computed via implicit differentiation on the power flow equations. At a solved operating point where ``K(V, p) = 0``:

```math
\frac{\partial V}{\partial p} = -J^{-1} \frac{\partial K}{\partial p}
```

This yields four sensitivity matrices:
- ``\partial |V| / \partial P``: voltage magnitude sensitivity to active power
- ``\partial |V| / \partial Q``: voltage magnitude sensitivity to reactive power
- ``\partial \theta / \partial P``: voltage angle sensitivity to active power
- ``\partial \theta / \partial Q``: voltage angle sensitivity to reactive power

The magnitude and angle sensitivities are extracted from the complex phasor sensitivity ``\partial V / \partial p``:

```math
\frac{\partial |V_i|}{\partial p_k} = \operatorname{Re}\!\left(\frac{\partial V_i}{\partial p_k} \cdot \frac{\bar{V}_i}{|V_i|}\right), \qquad
\frac{\partial \theta_i}{\partial p_k} = \operatorname{Im}\!\left(\frac{\partial V_i}{\partial p_k} \cdot \frac{\bar{V}_i}{|V_i|^2}\right)
```

### Jacobian Block Sensitivities

The unified interface also supports querying individual Jacobian blocks directly. For example, ``\partial P / \partial \theta`` and ``\partial P / \partial |V|`` can be obtained as `calc_sensitivity(state, :p, :va)` and `calc_sensitivity(state, :p, :vm)`.

## Current Sensitivity

Branch current sensitivities are computed via the chain rule through voltage sensitivities:

```math
\frac{\partial I_\ell}{\partial p_k} = Y_{ft} \left(\frac{\partial V_f}{\partial p_k} - \frac{\partial V_t}{\partial p_k}\right)
```

where ``I_\ell = Y_{ft} (V_f - V_t)`` is the current on branch ``\ell`` connecting buses ``f`` and ``t``. Current magnitude sensitivity uses:

```math
\frac{\partial |I_\ell|}{\partial p_k} = \frac{\operatorname{Re}\!\left(\frac{\partial I_\ell}{\partial p_k} \cdot \bar{I}_\ell\right)}{|I_\ell|}
```

## Branch Flow Sensitivity

Active power flow sensitivity on branch ``\ell`` uses the product rule on ``P_\ell = \operatorname{Re}(V_f \bar{I}_\ell)``:

```math
\frac{\partial P_\ell}{\partial p_k} = \operatorname{Re}\!\left(\frac{\partial V_f}{\partial p_k} \bar{I}_\ell + V_f \overline{\frac{\partial I_\ell}{\partial p_k}}\right)
```

## Parameter Transforms

The power flow formulation uses power injections ``(p, q)`` as native parameters. To obtain sensitivities w.r.t. demand ``(d, q_d)``, the transform ``p = g - d`` yields:

```math
\frac{\partial (\cdot)}{\partial d} = -\frac{\partial (\cdot)}{\partial p}, \qquad
\frac{\partial (\cdot)}{\partial q_d} = -\frac{\partial (\cdot)}{\partial q}
```

These transforms are applied automatically by the unified interface when using `:d` or `:qd` as the parameter symbol.
