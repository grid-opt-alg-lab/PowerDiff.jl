# Example Use Case: Post-Event Transmission Restoration using PowerDiff

## Running the example

Run the example from the root of the PowerDiff repository so Julia uses the
package versions specified by `Project.toml`:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/sens_examples/post_event_restoration.jl
```

**Note:** PowerDiff currently requires PowerIO 0.8.2 or later within the compatible 0.8 release series. If an older untracked `Manifest.toml` pins PowerIO 0.1.x, it must be regenerated from the current `Project.toml` before running the example.

## Problem Setup: *Prioritizing Line Repairs Following Disturbance*

This example application involves prioritizing ltransmission ine restoration following extreme weather, equipment failure, or other disruptive grid event scenarios where multiple transmission lines may be simultaneously damaged, leading to unserved demand in the network. While the locations of damaged lines may be known by the system operator, limited resources (crew, equipment) may prevent all lines from being repaired at once. It is then of interest to prioritize restoration of lines based on the amount of unserved electricity demand recovered by a repair. This example considers the following decision problem:
<p align="center">
  <em>“Given a set of known outaged transmission lines, which line should be restored first to induce the greatest reduction in total load shed?”</em>
</p>

The example in [`post_event_restoration.jl`](post_event_restoration.jl) studies
this decision on the IEEE 300-bus test system. Branches 3, 6, 16, 17, and 37 (arbitrarily chosen) are
treated as severely damaged by setting their relaxed switching states to 0.01.
Their nearly open switching states cause load shedding. PowerDiff is used to
assign local sensitivity-based priority weights to the five possible repairs.

## Network and damage model

Let:

- $\mathcal{B}$ be the set of buses;
- $\mathcal{G}$ be the set of generators;
- $\mathcal{L}$ be the set of transmission branches;
- $\mathcal{D}\subseteq\mathcal{L}$ be the set of damaged/outaged branches;
- $d_i$ be demand at bus $i$; and
- $s_\ell\in[0,1]$ be the continuous switching state of branch $\ell$.

While physical transmission line status is binary, with $s_\ell=1$ indicating a closed or energized branch and $s_\ell = 0$ indicating an open or outaged branch, we relax this branch status to a continuous $[0,1]$ to allow for calculation of derivatives with respect to branch switching. An intermediate value is a continuous relaxation of the binary switching state. To represent a severely damaged or "nearly open" branch while retaining a differentiable operating point, each damaged branch is assigned a small positive switching state $\epsilon$: 
$$
    s_\ell =     
    \begin{cases}
        \epsilon, & \ell\in\mathcal{D},\\
        1, & \ell\notin\mathcal{D},
    \end{cases}
    \qquad \epsilon=0.01.
    \tag{1}
    
$$
Undamaged branches have $s_\ell=1$, and a complete repair changes the switching state of branch $\ell$ from $\epsilon$ to one. 

## Operating problem

For a fixed continuous switching vector $s$ defined by (1), PowerDiff solves a DC optimal
power flow (DC OPF) to find the post-disturbance operating point. This consists of generator outputs, bus voltage angles,
branch flows, and any load shedding.

A simplified representation is

$$
\begin{aligned}
\min_{p^g,\theta,f,p^{\mathrm{sh}}}\quad
    & C(p^g) + c^{\mathrm{sh}}\sum_{i\in\mathcal{B}}p_i^{\mathrm{sh}} \\
\text{subject to}\quad
    & \text{nodal power balance}, \\
    & f_\ell=-b_\ell s_\ell(\theta_{i(\ell)}-\theta_{j(\ell)}),
      && \ell\in\mathcal{L}, \\
    & -\overline f_\ell\leq f_\ell\leq\overline f_\ell,
      && \ell\in\mathcal{L}, \\
    & \underline p_g\leq p_g^g\leq\overline p_g,
      && g\in\mathcal{G}, \\
    & 0\leq p_i^{\mathrm{sh}}\leq\max(d_i,0),
      && i\in\mathcal{B}.
\end{aligned}
$$

Here, $p_i^{\mathrm{sh}}$ is the load shed at bus $i$.
PowerDiff assigns load shedding a high cost, so the OPF uses it only when the
available generation and transmission network cannot serve all demand.

The quantity of interest is total unserved demand, or load shed, of the grid:

$$
    U(s)=\sum_{i\in\mathcal{B}}
    p_i^{\mathrm{sh}*}(s)
    \tag{2}
$$

where the star indicates the optimal DC OPF solution at switching state $s$.

## PowerDiff Sensitivity Matrix

Suppose only one damaged branch can be fully restored. A direct, exhaustive approach to finding the most effective line repair could be to perform a separate OPF solve for every repair candidate. This may be practical for a small dataset, but its computational cost grows linearly with the number of repair candidates.

PowerDiff can instead supplies a local screening score for all branches from the
damaged operating point by differentiating the optimal load-shedding solution with respect to the continuous branch switching states. The bus-by-branch load shedding sensitivity matrix is 
$$
    J_{i\ell} = \frac{\partial p_i^{\mathrm{sh}*}}{\partial s_\ell}
    \tag{3}
$$ 
Each entry at row $i$ and column $l$ represents the local change in optimal load shed at bus $i$ caused by a small change in the continuous switching state $s_l$ of branch $l$. The Jacobian matrix can be calculated using: 
```julia
J = PowerDiff.calc_sensitivity(
    damaged_grid,
    :psh,
    :sw,
)
```
In the symbol interface:

- `:psh` specifies optimal load shedding as the the operand, or output being differentiated;
- `:sw` ispecifies continuous branch switching state as the parameter; and
- `J[i, l]` describes how load shedding at bus `i` responds locally to branch
switching state `l`.


For this example, we are concerned with the local sensitivity of the total load shedding across all buses $\mathcal{B}$ with respect to branch switching states. This desired local sensitivity is defined as

$$
\kappa_\ell =  \frac{\partial U(s)}{\partial s_\ell}
    =
    \sum_{i\in\mathcal{B}}J_{i\ell}.
  \tag{4}
$$

A negative $\kappa_\ell$ indicates that a small increase in the switching state of branch $\ell$ locally reduces the total load shed $U(s)$. Therefore, the magnitude of $\kappa_\ell$ can be interpreted as the local marginal "importance" of repairing branch $\ell  \in \mathcal{D}$ to reducing the total load shed in the damaged network.

## PowerDiff Vector-Jacobian Product

The complete sensitivity matrix $J$ is not required for this example; we only need their aggregates $J^\mathsf{T}\mathbf{1}$. We can therefore use the vector-Jacobian product provided by PowerDiff: 


```julia
total_shed_sensitivity = PowerDiff.vjp(
    damaged_grid,
    :psh,
    :sw,
    ones(base_network.n),
)
```
where `base_network.n` represents theh number of buses in the network.

This is equivalent to the full-matrix calculation:

```julia
J = PowerDiff.calc_sensitivity(damaged_grid, :psh, :sw)
total_shed_sensitivity = Matrix(J)' * ones(base_network.n)
```

However, `vjp` does not construct or store a complete bus-by-branch sensitivity matrix, instead directly calculating the local sensitivity of total load shed with respect to all switching parameters.
This significantly reduces computational cost for large networks or repeated sensitivity analyses.

## Restoration Priority Quantification
The resulting sensitivities $\kappa_\ell$ are signed local derivatives, neither probabilities nor exact predictions of the benefits of complete line repairs. We convert $\kappa_\ell$ into nonnegative local restoration-priority weights that are normalized over the damaged set $\mathcal{D}$:

$$
\pi_\ell = \frac{\max \{ 0, - \kappa_\ell\}}{\displaystyle\sum_{j\in\mathcal{D}}\max \{ 0, - \kappa_j\} }
\tag{5}
$$

A branch $\ell$ receives positive weight $\pi_\ell$ when increasing its continuous switching state $s_\ell$ is locally associated with reduced total load shed in the grid. Provided that at least one branch has a positive restoration score, the weights satisfy $\pi_\ell \geq 0$ and $\sum_{\ell\in\mathcal{D}} \pi_\ell =1$. These weights are probability-like values quantifying restoration priority, and they describe the share of local marginal importance assigned to each damaged branch. They are not true probabilities that brach $\ell$ is the best full repair. 


## Solution Procedure

The implemented solution has seven steps:

1. Load IEEE300 and solve the healthy-network DC OPF.
2. Arbitrarily choose $\mathcal{D}={3,6,16,17,37}$. Set the continuous switching states of damaged branches in $D$ to 0.01 and solve the damaged-grid DC OPF.
3. Measure total unserved demand in the damaged solution.
4. Use `PowerDiff.vjp` to compute $\partial U/\partial s_\ell$ 
5. Convert the derivatives into nonnegative scores
   $\max\{0,-\kappa_\ell\}$ and normalize them into priority weights
   $\pi_\ell$.
6. Rank the damaged branches by their local priority weights.
7. Increase or fully restore each candidate branch and re-solve the DC OPF to
   compare the local ranking with the discrete full-repair outcomes.

For a sufficiently small switching-state perturbation $\Delta s_\ell$, the first-order approximation of the change in total load shed is $U(s)+\kappa_\ell \Delta s_\ell$, where the locally predicted reduction in load shedding is $-\kappa_\ell \Delta s_\ell$. This approximation only describes behavior near the damaged operating point and does not directly describe the effect of changing branch $\ell$ from $s_\ell =\epsilon$ to $s_\ell=1$. A complete repair is a large change that could alter generator dispatch, power flow patterns, and locations of load shed. 

These priority weights can be used to screen candidate repairs, where the candidates with highest $\pi_\ell$  are evaluated by individually setting their $s_\ell=1$ and solving new DC OPFs. The final discrete restoration decision can be based on these explicit full-repair DC OPF simulation outcomes.

## Local Prediction and Full-Repair Verification using IEEE300 Test Case

The example tests PowerDiff's approximation by increasing a damaged branch's switching
state from 0.01 to 0.02 on the IEEE-300 test system. It also tests a full repair by setting the branch's switching state to one and solving a fresh OPF.

The damaged branch set is arbitrarily set as:
$$\mathcal{D}=\{3,6,16,17,37\}, \qquad s_\ell = \begin{cases} 0.01, & \ell\in\mathcal{D},\\ 1, & \ell\notin\mathcal{D}. \end{cases}$$
The given branches are labeled using their provided branch IDs. The original IEEE-300 network has no load shed while the damaged configuration has 54.05 MW of load shed. The local sensitivity results and explicit repair evaluations are shown below:

| Branch | $\kappa_\ell$ (MW) | Priority weight | Predicted \(+0.01\) | Actual \(+0.01\) | Full-repair recovery (MW) |
|---:|---:|---:|---:|---:|---:|
| 6 | -278.863 MW | 39.8% | 2.789 MW | 2.789 MW | 27.211 MW |
| 3 | -238.987 MW | 34.1% | 2.390 MW | 2.390 MW | 22.440 MW |
| 37 | -100.079 MW | 14.3% | 1.001 MW | 1.001 MW | 2.799 MW |
| 16 | -46.781 MW | 6.7% | 0.468 MW | 0.432 MW | 0.432 MW |
| 17 | -36.297 MW | 5.2% | 0.363 MW | 0.363 MW | 1.167 MW |

 Column 2 reports local sensitivity $\kappa_\ell$; column 3 reports priority weight $\pi_\ell$; column 4 reports predicted reduction in load shed (MW), $-\kappa_\ell \Delta s$, when $\Delta s_\ell = 0.01$; column 5 reports the actual reduction in load shed (MW) from DC OPF results for $\Delta s_\ell=0.01$; and column 6 reports the total load shed reduction (MW) when the branch is fully repaired to $s_\ell=1$. As shown, branch 6 received the highest local restoration-priority weight. Explicitly restoring each branch and solving the DC OPF also verifies branch 6 as the line repair that results in the largest reduction in load shed in this example. However, the ordering of complete-repair outcomes does not perfectly align with the local priority ordering, illustrating why $\pi_\ell$ are screening information rather than theh probabilities of the best full repair. In particular, branch 16 has a larger local priority weight than branch 17, but a full repair of branch 17 recovers more load shed. Discrepancies like this are expected because $\kappa_\ell$ describes local sensitivity near $s_\ell=0.01$ while a full repair to $s_\ell=1$ can move the DC OPF into different active-constraint regions.

Therefore, obtaining weights from this sensitivity analysis should be interpreted as a scalable method for screening large sets of damaged lines to identify a smaller subset of candidate lines to evaluate using a full DC OPF solve. The sensitivity analysis should not replace the full evaluation of a final discrete repair.


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
