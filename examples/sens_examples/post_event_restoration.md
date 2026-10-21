# Example Use Case: Post-Event Transmission Restoration with PowerDiff

## Running the example

Run the example from the root of the PowerDiff repository so Julia uses the
package versions specified by `Project.toml`:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/sens_examples/post_event_restoration.jl
```

The example resolves whatever `Project.toml` pins. If an untracked `Manifest.toml`
holds an older PowerIO, regenerate it from `Project.toml` before running.

## Example Problem Setup

After a major disturbance, several transmission lines may be simultaneously damaged. The system operator knows which lines are damaged, but repair resources (crews
and equipment) are limited. If only one line can be restored first, the operator
must decide which repair will recover the most unserved electricity demand.

The example in [`post_event_restoration.jl`](post_event_restoration.jl) studies
this decision on the IEEE 300-bus test system. Branches 3, 6, 16, 17, and 37 are
treated as severely damaged by setting their relaxed switching states to 0.01.
Their nearly open switching states cause load shedding. PowerDiff is used to
assign local sensitivity-based priority weights to the five possible repairs.

## Network and damage model

Let:

- \(\mathcal{B}\) be the set of buses;
- \(\mathcal{E}\) be the set of transmission branches;
- \(\mathcal{D}\subseteq\mathcal{E}\) be the known damaged branches;
- \(d_i\) be demand at bus \(i\); and
- \(s_\ell\in[0,1]\) be the continuous switching state of branch \(\ell\).

The physical branch status is binary: \(s_\ell=1\) represents a closed branch
and \(s_\ell=0\) represents an open branch. PowerDiff relaxes this switching
state to the continuous interval \([0,1]\) for sensitivity analysis. An
intermediate value is a continuous relaxation of the switching decision, not a
probability that the branch is available. The example uses

\[
\mathcal{D}=\{3,6,16,17,37\},
\qquad
s_3=s_6=s_{16}=s_{17}=s_{37}=0.01.
\]

The small positive switching state represents a nearly open damaged branch
while retaining a differentiable point with the same energized island
partition. All undamaged branches have switching state one.

## Operating problem

For a fixed continuous switching vector \(s\), PowerDiff solves a DC optimal
power flow (DC OPF). The model chooses generator outputs, bus voltage angles,
branch flows, and any unavoidable load shedding.

A simplified representation is

\[
\begin{aligned}
\min_{p^g,\theta,f,p^{\mathrm{sh}}}\quad
    & C(p^g) + c^{\mathrm{sh}}\sum_{i\in\mathcal{B}}p_i^{\mathrm{sh}} \\
\text{subject to}\quad
    & \text{nodal power balance}, \\
    & f_\ell=-b_\ell s_\ell(\theta_{i(\ell)}-\theta_{j(\ell)}),
      && \ell\in\mathcal{E}, \\
    & -\overline f_\ell\leq f_\ell\leq\overline f_\ell,
      && \ell\in\mathcal{E}, \\
    & \underline p_g\leq p_g^g\leq\overline p_g,
      && g\in\mathcal{G}, \\
    & 0\leq p_i^{\mathrm{sh}}\leq\max(d_i,0),
      && i\in\mathcal{B}.
\end{aligned}
\]

Here, \(p_i^{\mathrm{sh}}\) is demand that cannot be served at bus \(i\).
PowerDiff assigns load shedding a high cost, so the OPF uses it only when the
available generation and transmission network cannot serve all demand.

The quantity of interest is total unserved demand:

\[
S(s)=\sum_{i\in\mathcal{B}}p_i^{\mathrm{sh}*}(s),
\]

where the star indicates the optimal DC OPF solution at switching state \(s\).

## Restoration decision

Suppose one damaged branch can be fully restored. The exact best-first repair
would be

\[
\ell^*=
\arg\min_{\ell\in\mathcal{D}}
S\!\left(s+(1-s_\ell)e_\ell\right),
\]

where \(e_\ell\) changes only branch \(\ell\). Evaluating this expression
directly requires a separate OPF solve for every repair candidate.

PowerDiff instead supplies a local screening score for all branches from the
damaged operating point:

\[
q_\ell = \frac{\partial S}{\partial s_\ell}.
\]

If \(q_\ell<0\), a small increase in the continuous switching state of branch
\(\ell\) locally reduces unserved demand. The derivative is a local marginal
influence score; it is not a direct prediction of the effect of fully closing
the branch. Define the nonnegative restoration-priority score

\[
r_\ell=\max\{0,-q_\ell\}.
\]

The scores are normalized over the damaged set to form a probability-like
restoration-priority distribution:

\[
\pi_\ell
=
\frac{r_\ell}{\sum_{j\in\mathcal D}r_j},
\qquad
\sum_{\ell\in\mathcal D}\pi_\ell=1.
\]

The weight \(\pi_\ell\) is not a statistically calibrated probability that
branch \(\ell\) is the best full repair. It is the share of local sensitivity
priority assigned to that candidate. The highest-priority branch is

\[
\widehat\ell=\arg\max_{\ell\in\mathcal D}\pi_\ell.
\]

## The PowerDiff sensitivity matrix

PowerDiff can construct the bus-by-branch sensitivity matrix

\[
J_{i\ell}
=
\frac{\partial p_i^{\mathrm{sh}*}}{\partial s_\ell}.
\]

For IEEE300, \(J\) has 300 rows and 411 columns. It can be calculated with

```julia
J = PowerDiff.calc_sensitivity(
    damaged_problem,
    :psh,
    :sw,
)
```

In the symbol interface:

- `:psh` is the operand, or output being differentiated;
- `:sw` is the parameter representing continuous branch switching state; and
- `J[i, l]` describes how load shedding at bus `i` responds locally to branch
  switching state `l`.

Because total load shedding is the sum over buses, its branch sensitivity is

\[
q = J^\mathsf{T}\mathbf{1},
\]

where \(\mathbf{1}\) is a vector of 300 ones. Using the full matrix, this could
be computed as

```julia
J = PowerDiff.calc_sensitivity(damaged_problem, :psh, :sw)
q = Matrix(J)' * ones(base_network.n)
```

## Why the implementation uses `vjp`

The example does not require the individual bus-level entries of \(J\); it only
needs their aggregate \(J^\mathsf{T}\mathbf{1}\). It therefore uses PowerDiff's
vector-Jacobian product:

```julia
total_shed_sensitivity = PowerDiff.vjp(
    damaged_problem,
    :psh,
    :sw,
    ones(base_network.n),
)
```

This is mathematically equivalent to the full-matrix calculation:

```julia
J = PowerDiff.calc_sensitivity(damaged_problem, :psh, :sw)
q_from_matrix = Matrix(J)' * ones(base_network.n)
q_from_vjp = PowerDiff.vjp(
    damaged_problem,
    :psh,
    :sw,
    ones(base_network.n),
)

q_from_matrix ≈ q_from_vjp
```

The `vjp` form avoids constructing and storing the entire sensitivity matrix.
This distinction matters for large networks or repeated sensitivity analyses.

## Solution procedure

The implemented solution has seven steps:

1. Load IEEE300 and solve the healthy-network DC OPF.
2. Set the continuous switching states of branches 3, 6, 16, 17, and 37 to
   0.01 and solve the damaged-grid DC OPF.
3. Measure total unserved demand in the damaged solution.
4. Use `PowerDiff.vjp` to compute \(\partial S/\partial s_\ell\) for every
   branch.
5. Convert the derivatives into nonnegative scores
   \(r_\ell=\max\{0,-q_\ell\}\) and normalize them into priority weights
   \(\pi_\ell\).
6. Rank the damaged branches by their local priority weights.
7. Increase or fully restore each candidate branch and re-solve the OPF to
   compare the local ranking with the discrete full-repair outcomes.

For a small switching-state change \(\Delta s_\ell\), the first-order prediction is

\[
S(s+\Delta s_\ell e_\ell)
\approx
S(s)+q_\ell\Delta s_\ell.
\]

The predicted amount of recovered load is therefore

\[
\text{predicted recovery}_\ell
=
-q_\ell\Delta s_\ell.
\]

The example tests this approximation by increasing a damaged branch's switching
state from 0.01 to 0.02. It also tests a full repair by setting the branch's
switching state to one and solving a fresh OPF.

## IEEE300 result

The healthy IEEE300 network serves all demand. Setting the continuous switching
states of branches 3, 6, 16, 17, and 37 to 0.01 creates 54.050 MW of unserved
demand. The computed results are:

| Branch | \(\partial S/\partial s_\ell\) | Priority weight | Predicted recovery for \(+0.01\) | Actual recovery for \(+0.01\) | Actual full-repair recovery |
|---:|---:|---:|---:|---:|---:|
| 6 | -278.863 MW | 39.8% | 2.789 MW | 2.789 MW | 27.211 MW |
| 3 | -238.987 MW | 34.1% | 2.390 MW | 2.390 MW | 22.440 MW |
| 37 | -100.079 MW | 14.3% | 1.001 MW | 1.001 MW | 2.799 MW |
| 16 | -46.781 MW | 6.7% | 0.468 MW | 0.432 MW | 0.432 MW |
| 17 | -36.297 MW | 5.2% | 0.363 MW | 0.363 MW | 1.167 MW |

Branch 6 receives the highest local sensitivity-based priority weight. Explicitly
restoring each branch and re-solving the OPF shows that branch 6 also provides
the largest full-repair benefit in this example. The relative full-repair
results for branches 16 and 17 do not follow their local priority order,
illustrating why the priority weights are screening information rather than
probabilities of the best discrete repair.

## Interpretation and limitation

The PowerDiff derivative is local: it describes behavior near the damaged
operating point. It accurately predicts the effect of the small \(+0.01\)
switching-state change in this example. A complete repair from 0.01 to 1.0 is a
large and potentially nonlinear change, so its benefit should not be treated as
exactly equal to the first-order prediction.

The normalized sensitivities are therefore used as probability-like screening
weights. They allocate relative local priority across repair candidates but are
not calibrated probabilities that a candidate will be the best full repair.
The highest-priority candidates are verified with explicit OPF re-solves. This
combines scalable derivative information with exact evaluation of the final
discrete repair decision.
