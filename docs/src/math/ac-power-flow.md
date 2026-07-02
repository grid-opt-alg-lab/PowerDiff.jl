# AC Power Flow

## Admittance Matrix

The bus admittance matrix is assembled from the branch admittances and the incidence matrix:

```math
Y = A^\top \operatorname{diag}\bigl(\mathrm{sw} \circ (g + jb)\bigr) \, A + \operatorname{diag}(g_{\text{sh}} + jb_{\text{sh}})
```

where ``A`` is the ``m \times n`` incidence matrix (row ``e``, written ``a_e^\top``, has ``+1`` at branch ``e``'s from-bus and ``-1`` at its to-bus), ``g`` and ``b`` are branch conductances and susceptances, ``\mathrm{sw}_e \in [0,1]`` is the switching state that scales branch ``e``'s admittance (``\mathrm{sw}_e = 0`` removes the branch), and ``g_{\text{sh}}``, ``b_{\text{sh}}`` are shunt admittances. An off-nominal transformer replaces each branch's rank-1 contribution ``a_e a_e^\top`` by a ``2 \times 2`` primitive built from the complex tap ratio.

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

## Admittance and Topology Sensitivity

The sections above differentiate the power flow with respect to power injections. The same implicit function machinery differentiates it with respect to the **network admittance parameters**: branch conductance ``g_e`` and susceptance ``b_e``. This is the basis for topology and admittance control, predicting how the voltage state moves when a branch's admittance changes, or when a line is switched, *without* solving the power flow equations again.

At a solved operating point the power balance residual ``F(V, \beta) = 0`` depends on a parameter ``\beta \in \{g_e, b_e\}`` only through the admittance matrix ``Y = A^\top \operatorname{diag}(\mathrm{sw} \circ (g + jb)) \, A``. Implicit differentiation reuses the same Jacobian ``J`` used for voltage sensitivity to power injections (same factorization as above, with a different right side):

```math
\frac{\partial V}{\partial \beta} = -J^{-1} \frac{\partial F}{\partial \beta}, \qquad
\frac{\partial F}{\partial \beta} = \begin{bmatrix} \partial P / \partial \beta \\ \partial Q / \partial \beta \end{bmatrix}
```

Each branch enters ``Y`` as a rank-1 update, so the parameter derivatives are sparse:

```math
\frac{\partial Y}{\partial g_e} = \mathrm{sw}_e \, a_e a_e^\top, \qquad
\frac{\partial Y}{\partial b_e} = j \, \mathrm{sw}_e \, a_e a_e^\top
```

(shown for unit tap ratio; an off-nominal complex tap replaces the outer product by the branch's ``2 \times 2`` primitive). Let ``\Delta V = A V`` be the vector of edge voltage drops (so ``\Delta V_e = V_f - V_t`` across branch ``e``). Still at unit tap, define the matrix ``M`` with entries

```math
M_{i,e} = \mathrm{sw}_e \, \bar{V}_i \,(a_e)_i \, \Delta V_e ,
```

which is nonzero only at branch ``e``'s two endpoints. Using ``P = \operatorname{Re}(\bar V \circ YV)`` and ``Q = -\operatorname{Im}(\bar V \circ YV)``, the injection derivatives are:

```math
\frac{\partial P}{\partial g} = \operatorname{Re}(M), \quad
\frac{\partial Q}{\partial g} = -\operatorname{Im}(M); \qquad
\frac{\partial P}{\partial b} = -\operatorname{Im}(M), \quad
\frac{\partial Q}{\partial b} = -\operatorname{Re}(M).
```

The conductance and susceptance columns share the same ``M`` and differ only in which part is taken: the factor ``j`` in ``\partial Y / \partial b_e`` rotates the real and imaginary parts into each other. Solving for ``\partial V / \partial \beta`` and projecting onto magnitude and angle exactly as in the voltage sensitivity case,

```math
\frac{\partial |V_i|}{\partial \beta} = \operatorname{Re}\!\left(\frac{\partial V_i}{\partial \beta} \cdot \frac{\bar V_i}{|V_i|}\right), \qquad
\frac{\partial \theta_i}{\partial \beta} = \operatorname{Im}\!\left(\frac{\partial V_i}{\partial \beta} \cdot \frac{\bar V_i}{|V_i|^2}\right),
```

gives the admittance sensitivities `calc_sensitivity(state, :vm, :g)`, `calc_sensitivity(state, :va, :b)`, and so on. Branch current and flow sensitivities to ``g`` and ``b`` carry a **direct** term in addition to this voltage chain rule, because the perturbed branch's own admittance enters its flow. For a shuntless branch ``I_\ell = (g_\ell + j b_\ell)(V_f - V_t)``,

```math
\frac{\partial I_\ell}{\partial \beta}
  = \underbrace{\frac{\partial (g_\ell + j b_\ell)}{\partial \beta}\,(V_f - V_t)}_{\text{direct (branch } \ell)}
  + \underbrace{(g_\ell + j b_\ell)\left(\frac{\partial V_f}{\partial \beta} - \frac{\partial V_t}{\partial \beta}\right)}_{\text{indirect (through the voltage change)}},
```

which mirrors the direct and indirect split of the DC switching sensitivity. The direct term is nonzero only when ``\beta`` is branch ``\ell``'s own ``g`` or ``b`` (then ``\partial (g_\ell + j b_\ell)/\partial g_\ell = 1`` or ``\partial (g_\ell + j b_\ell)/\partial b_\ell = j``); for every other branch only the voltage term remains. Power flow sensitivities follow from ``S_\ell = V_f \bar I_\ell`` by the product rule, and the implementation adds this direct contribution for `:im` and `:f` queries.

In AC OPF the analogous topology parameter is the switching state ``\mathrm{sw}``, which scales the whole branch admittance primitive (``\partial Y / \partial \mathrm{sw}_e = (g_e + jb_e)\, a_e a_e^\top`` at unit tap). Its sensitivities come from implicitly differentiating the KKT system (see [AC Optimal Power Flow](@ref)).

## References

The implicit differentiation framework for the power flow equations, including the guarantee that the derivative exists for generic radial or meshed networks and its application to voltage, current, and power flow sensitivities under admittance and topology changes, follows:

- S. Talkington, D. Turizo, S. A. Dorado-Rojas, R. K. Gupta & D. K. Molzahn, ["Differentiating Through Power Flow Solutions for Admittance and Topology Control,"](https://arxiv.org/abs/2510.17071) 2025.
