# DC Power Flow

## Reduced System

For non-OPF power flow with fixed generation, the DC approximation linearizes
the power flow equations using the susceptance-weighted Laplacian. The voltage
angles satisfy the reduced system obtained by eliminating one reference bus per
energized island:

```math
\theta_r = B_r^{-1} \, p_r
```

where:
- ``B_r`` is the susceptance-weighted Laplacian with one reference row and column deleted per energized island, including isolated buses
- ``p_r = g_r - d_r`` is the net injection vector with the reference entries removed
- ``\theta_{\text{refs}} = 0`` by convention

The susceptance-weighted Laplacian is:

```math
B = A^\top \operatorname{diag}(-b \circ \mathrm{sw}) \, A
```

where ``A`` is the ``m \times n`` incidence matrix and ``b`` stores the imaginary part of the inverse impedance (``b_e = \operatorname{Im}(1/z_e) < 0`` for inductive branches, so ``-b > 0``).

`DCNetwork` caches the energized island partition used to choose reference
buses. Constructors initialize this cache, and topology dependent readers
refresh it if direct `b` or `sw` edits change which branches are energized. The
cache assumes serialized topology mutation: a shared `DCNetwork` may be read
from multiple threads only while its topology fields are not being mutated.

## Switching Sensitivity

Switching sensitivity follows from matrix perturbation theory. For a branch ``e`` with switching state ``\mathrm{sw}_e \in [0,1]``:

```math
\frac{\partial \theta_r}{\partial \mathrm{sw}_e}
  = -B_r^{-1} \frac{\partial B_r}{\partial \mathrm{sw}_e} \, \theta_r
```

where the perturbation is a rank-1 update from the incidence column of branch ``e`` restricted to non-reference buses:

```math
\frac{\partial B_r}{\partial \mathrm{sw}_e} = -b_e \, a_{e,r} \, a_{e,r}^\top
```

These derivatives apply while the energized island partition is fixed. Toggling
a bridge (a branch whose energization changes the island partition) adds or
removes a reference bus and changes the dimension of the reduced system ``B_r``,
so sensitivities are nonsmooth at that split or merge boundary.

### Flow Sensitivity to Switching

Branch flows are ``f = W A \theta`` where ``W = \operatorname{diag}(-b \circ \mathrm{sw})``. The flow sensitivity has both indirect (via angle changes) and direct (via the switching coefficient) contributions:

```math
\frac{\partial f}{\partial \mathrm{sw}_e} = W A \frac{\partial \theta}{\partial \mathrm{sw}_e} + \text{direct effect on edge } e
```

## Demand Sensitivity

Since ``p = g - d`` and generation is fixed, ``\partial p / \partial d = -I``. The angle sensitivity to demand is:

```math
\frac{\partial \theta}{\partial d} = -B_r^{-1}
```

embedded in the non-reference block (with zero rows/columns for the reference buses). The flow sensitivity follows as:

```math
\frac{\partial f}{\partial d} = W A \frac{\partial \theta}{\partial d}
```
