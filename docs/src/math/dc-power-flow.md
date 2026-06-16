# DC Power Flow

## Reduced System

For non-OPF power flow with fixed generation, the DC approximation linearizes
the power flow equations using the susceptance-weighted Laplacian. The voltage
angles satisfy the reduced system obtained by eliminating the reference (slack) bus:

```math
\theta_r = B_r^{-1} \, p_r
```

where:
- ``B_r`` is the susceptance-weighted Laplacian with the reference bus row and column deleted (invertible for a connected network)
- ``p_r = g_r - d_r`` is the net injection vector with the reference entry removed
- ``\theta_{\text{ref}} = 0`` by convention

The susceptance-weighted Laplacian is:

```math
B = A^\top \operatorname{diag}(-b \circ \mathrm{sw}) \, A
```

where ``A`` is the ``m \times n`` incidence matrix (row ``e``, written ``a_e^\top``, has ``+1`` at branch ``e``'s from-bus and ``-1`` at its to-bus, so ``a_e^\top \theta = (A\theta)_e`` is the angle difference across the branch), and ``b`` stores the imaginary part of the inverse impedance (``b_e = \operatorname{Im}(1/z_e) < 0`` for inductive branches, so ``-b > 0``).

## Switching Sensitivity

Switching sensitivity follows from matrix perturbation theory. For a branch ``e`` with switching state ``\mathrm{sw}_e \in [0,1]``:

```math
\frac{\partial \theta_r}{\partial \mathrm{sw}_e}
  = -B_r^{-1} \frac{\partial B_r}{\partial \mathrm{sw}_e} \, \theta_r
```

where the perturbation is a rank-1 update from row ``a_e^\top`` of branch ``e`` restricted to the non-reference buses (denoted ``a_{e,r}``):

```math
\frac{\partial B_r}{\partial \mathrm{sw}_e} = -b_e \, a_{e,r} \, a_{e,r}^\top
```

### Flow Sensitivity to Switching

Branch flows are ``f = W A \theta`` with ``W = \operatorname{diag}(-b \circ \mathrm{sw})``, so branch ``e`` carries ``f_e = -b_e \, \mathrm{sw}_e \,(a_e^\top \theta)``. Both the weight ``W`` and the angles ``\theta`` depend on ``\mathrm{sw}_e``, so the product rule splits the sensitivity into an **indirect** part (every branch feels the change in angles) and a **direct** part (only branch ``e``'s own weight changes):

```math
\frac{\partial f}{\partial \mathrm{sw}_e}
  = \underbrace{W A \frac{\partial \theta}{\partial \mathrm{sw}_e}}_{\text{indirect (all branches)}}
  + \underbrace{\frac{\partial W}{\partial \mathrm{sw}_e} A \theta}_{\text{direct (branch } e \text{ only)}}
  = W A \frac{\partial \theta}{\partial \mathrm{sw}_e} \;-\; b_e \,(a_e^\top \theta)\, e_e
```

The direct term uses ``\partial W / \partial \mathrm{sw}_e = -b_e \, e_e e_e^\top``. Only the diagonal entry of ``W`` for branch ``e`` depends on ``\mathrm{sw}_e``, so it adds ``-b_e \,(a_e^\top \theta) = -b_e \,(A\theta)_e`` to the flow on branch ``e`` alone, where ``e_e \in \mathbb{R}^m`` is the unit vector for branch ``e``.

Substituting the angle sensitivity ``\partial \theta / \partial \mathrm{sw}_e`` we get

```math
\frac{\partial \theta_r}{\partial \mathrm{sw}_e}
  = -B_r^{-1} \frac{\partial B_r}{\partial \mathrm{sw}_e}\, \theta_r
  = b_e \,(a_{e,r}^\top \theta_r)\, B_r^{-1} a_{e,r}
```

(embedded into the full bus vector with the reference entry held at zero) which makes the flow sensitivity fully explicit:

```math
\frac{\partial f}{\partial \mathrm{sw}_e}
  = b_e \,(a_e^\top \theta) \Big(
    \underbrace{W A_r\, B_r^{-1} a_{e,r}}_{\text{indirect (all branches)}}
    \;-\; \underbrace{e_e}_{\text{direct (branch } e \text{ only)}}
  \Big)
```

where ``A_r`` is the incidence matrix with its reference bus column removed (so ``A \,\partial\theta/\partial \mathrm{sw}_e = A_r \,\partial\theta_r/\partial \mathrm{sw}_e``), and we used ``a_{e,r}^\top \theta_r = a_e^\top \theta`` since ``\theta_{\text{ref}} = 0``. The shared scalar ``b_e \,(a_e^\top \theta) = b_e \,(A\theta)_e`` is exactly the negative of branch ``e``'s flow when it is in service (``f_e = -b_e (A\theta)_e`` at ``\mathrm{sw}_e = 1``), so the whole flow sensitivity scales with how heavily branch ``e`` is loaded.

## Demand Sensitivity

Since ``p = g - d`` and generation is fixed, ``\partial p / \partial d = -I``. The angle sensitivity to demand is:

```math
\frac{\partial \theta}{\partial d} = -B_r^{-1}
```

embedded in the non-reference block (with zero rows/columns for the reference bus). The flow sensitivity follows as:

```math
\frac{\partial f}{\partial d} = W A \frac{\partial \theta}{\partial d}
```
