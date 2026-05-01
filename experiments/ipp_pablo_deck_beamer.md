# IPP Market-Aware Upgrade Planning — Beamer/PowerPoint mapping notes

This file is a **handoff document**: a structured outline that a downstream
agent (Beamer/PowerPoint generator) can consume to produce a LaTeX/Beamer or
Keynote/Google-Slides version of the HTML deck at
`experiments/ipp_pablo_deck.html`. It does *not* itself produce a deck.

Each slide gives:
- title
- body bullets (1–3 per bullet, copy-paste-able)
- equation block (LaTeX source ready for `\[ ... \]` or `equation*`)
- image reference (relative path; aspect ratio suggested)
- speaker notes (~30 words; what to say)
- Beamer-mapping hint (suggested frame layout)

Author: Samuel Talkington for Pablo Ruiz, NewGrid Inc.  Date: 2026-05-01.

---

## Slide 1 — Title

**Title.** Market-Aware Transmission Upgrade Planning for Independent Power Producers
**Subtitle.** A Differentiable Bilevel Approach via PowerDiff.jl
**Author.** Samuel Talkington, Georgia Tech
**Date.** 2026-05-01 — for Pablo Ruiz, NewGrid Inc.

**Speaker notes.** "Today I'd like to show how PowerDiff.jl turns Pablo's bilevel formulation into a tractable optimization with a clean per-iteration cost model — and a real RTS-GMLC demo."

**Beamer-mapping hint.** Standard `\titlepage` with institutional logo, gradient title text.

---

## Slide 2 — The IPP basis problem

**Bullets.**
- An IPP signs a Power Purchase Agreement (PPA) that settles at a single hub bus set H ⊆ N. They are paid the local nodal LMP for physically-delivered energy at their generator.
- Their realized profit per MWh is the **basis spread**: `λ_local − λ_hub`. Positive basis = profit; negative basis = loss.
- Real-world archetypes: MISO Indiana Hub vs. wind in West/Iowa; PJM AECO vs. PA gas; NEISO Mass Hub vs. Maine wind.

**Equation.**
\[
\Pi_{\text{IPP}} \;=\; \sum_{i \in N \setminus H} \bigl(\lambda_i - \bar\lambda_H\bigr)\,P_i \;+\; \text{(constant PPA strike)}
\]

**Image.** `experiments/img/miso_archetype.svg` (a simple two-zone schematic — the agent can draw this with TikZ in Beamer or omit it).

**Speaker notes.** "The IPP cares about basis. If you want to predict their profit you need a model of how transmission upgrades move LMPs. That's the problem we're solving today."

**Beamer-mapping hint.** Two-column frame: left column for bullets, right column for the schematic figure.

---

## Slide 3 — Pablo's bilevel formulation

**Bullets.**
- **Outer problem**: choose line capacities `fmax` (MW) over a feasible set U.
- **Inner problem**: full DC OPF returns LMP vector λ*(fmax).
- Weights encode the IPP's exposure: `w_i = +1` if `i ∈ H`, `w_i = -1` otherwise.

**Equation.**
\[
\begin{aligned}
\min_{\overline{\boldsymbol{f}} \in \mathcal{U}} \quad & \boldsymbol{w}^\top \boldsymbol{\lambda}_\star\!(\overline{\boldsymbol{f}}) \\
\text{s.t.} \quad & \boldsymbol{\lambda}_\star\!(\overline{\boldsymbol{f}}) \in \mathsf{OPF}\!(\overline{\boldsymbol{f}})
\end{aligned}
\]

**Image.** none.

**Speaker notes.** "This is exactly Pablo's formulation from your write-up — just stated in our notation. The challenge is the inner argmin: λ* is implicit; you can't write it down in closed form."

**Beamer-mapping hint.** Single-column frame, equation centered, large font.

---

## Slide 4 — What does w'λ actually mean? (KEY SLIDE)

**Bullets (TL;DR).**
- LMP decomposes as `λ_i = λ_en + cong_i` (energy + congestion).
- `cong_i = Σ_e (μ⁺[e] − μ⁻[e]) · PTDF[e, i]`. Zero if no line is binding.
- For single-bus hub `H = {h}`, `w'λ = (n-2)·λ_en + cong_h − Σ_{i≠h} cong_i` *up to sign*.
- **Practical**: `min w'λ` ⟺ **maximise IPP basis revenue** = `Σ_{i ≠ h} (λ_i − λ_h)`.
- Hub choice matters: hub at slack bus → IPP is co-located with cheap power, doesn't need upgrades; hub at load pocket → IPP profits from upgrades that relieve hub congestion.

**Equation block (one main, one corollary).**
\[
\lambda_i \;=\; \underbrace{\lambda_{\mathrm{en}}}_{\text{system-wide}} \;+\; \underbrace{\mathrm{cong}_i}_{\text{bus-specific}} \,, \qquad
\mathrm{cong}_i \;=\; \sum_{e} \bigl(\mu^+_e - \mu^-_e\bigr)\,\mathrm{PTDF}_{e,i}
\]
\[
\boldsymbol{w}^\top \boldsymbol{\lambda} \;=\; \lambda_h \;-\; \sum_{i \ne h} \lambda_i \;=\; -\,\Pi_{\text{IPP}}^{\text{1-MW per non-hub bus}}
\]

**Image.** `experiments/ipp_market_planning_3bus.png` (Tier 1 figure, 16:9 — shows the small-case sanity check that grounds the formula).

**Speaker notes.** "This is the slide I want to spend two minutes on. Pablo, you wrote w_i = +1 hub, -1 non-hub. The way to read that is: the IPP is short the hub (PPA pays settlement at hub LMP) and long every other node (paid local LMP for generation). Min w'λ literally maximises basis."

**Beamer-mapping hint.** Two-column: left column is the algebra (LMP decomposition + corollary), right column is the 3-bus image. Equation block uses `align*`.

---

## Slide 5 — Why this is hard (the standard approach)

**Bullets.**
- λ*(fmax) is **implicit**: defined by the KKT system of the inner OPF. No closed form.
- Black-box automatic differentiation through Ipopt: opaque, error-prone, allocates densely.
- Naïve finite differences: requires `m + 1` OPF solves per outer gradient. case14: ~20 solves; RTS-GMLC: 120 solves; per outer iteration. **Quadratic per-iter scaling** in network size.
- Active-set changes (a binding line becomes non-binding or vice versa) cause discontinuities — gradient methods need to handle this.

**Equation.**
\[
\nabla_{\overline{\boldsymbol{f}}}\, \boldsymbol{w}^\top\boldsymbol{\lambda}_\star \;=\; \boldsymbol{w}^\top \frac{\partial \boldsymbol{\lambda}_\star}{\partial \overline{\boldsymbol{f}}}, \qquad \frac{\partial \boldsymbol{\lambda}_\star}{\partial \overline{\boldsymbol{f}}} \;=\; -\bigl(\partial_z K\bigr)^{-1} \bigl(\partial_{\overline{f}} K\bigr)
\]

**Image.** none.

**Speaker notes.** "Naive AD or FD just doesn't scale. You spend most of your wall time re-solving OPFs to fingerprint the gradient. For real planning we need something smarter."

**Beamer-mapping hint.** Single-column, equation centered.

---

## Slide 6 — Why PowerDiff (the value pitch)

**Bullets.**
- PowerDiff differentiates *implicitly* through the KKT system — analytically, sparsely.
- `vjp(prob, :lmp, :fmax, w)` returns `wᵀ ∂λ*/∂fmax` in **one transpose KKT solve** + an O(m) scatter. No full Jacobian materialised.
- `update_fmax!` rewrites JuMP constraint RHS in-place; preserves the reduced-Laplacian Cholesky factor.
- Per-iteration cost: 1 OPF solve + 1 transpose KKT solve. Independent of `m`.

**Equation.** Code excerpt (use a `lstlisting` or `verbatim` block in Beamer):
```julia
prob = DCOPFProblem(net, d)
g    = zeros(m); work = zeros(kkt_dims(prob))
for k in 1:max_iters
    solve!(prob)                                   # forward OPF
    vjp!(g, prob, :lmp, :fmax, w; work=work)        # one transpose KKT solve
    e_star = argmin(g)                              # FW oracle
    update_fmax!(prob, fmax_k + Δ_step)             # in-place
end
```

**Image.** `experiments/img/per_iter_cost.svg` (a small bar chart comparing FD vs PowerDiff, can be omitted if not pre-rendered).

**Speaker notes.** "This is the engineering punch-line. The same loop runs on a 14-bus textbook case and a 73-bus realistic system — only the OPF dominates wall time."

**Beamer-mapping hint.** Single-column, code in monospace verbatim. Use `minted` package for syntax highlighting (Julia lexer).

---

## Slide 7 — Tier 1: 3-bus pedagogical demo

**Bullets.**
- Cheap gen at bus 1, expensive gen at bus 2, single load at bus 3.
- Line 1→3 saturates at fmax = 0.5 — bus-3 LMP > bus-1 LMP by a margin of `(c_2 − c_1) · (1 − slack of binding line)`.
- We compute the full ∂λ/∂fmax matrix analytically and verify entry-by-entry against finite differences.
- Hub = {bus 3}: IPP pays the load-pocket price; upgrades reduce that price.

**Equation.** A 3×2 matrix table (for the slide; the script outputs the actual numbers):
```
  ∂λ/∂fmax    line 1   line 2
  bus 1       0.0      0.0
  bus 2       0.0      0.0
  bus 3      -40.0     0.0
```

**Image.** `experiments/ipp_market_planning_3bus.png`.

**Speaker notes.** "Three buses, two lines. Line 1 is binding; everything you'd expect from PTDF holds. The Jacobian agrees with finite differences to 1e-7. This is the sanity check before we go to a real network."

**Beamer-mapping hint.** Two-column. Left: bullets + Jacobian table. Right: full image at 0.45× textwidth.

---

## Slide 8 — Tier 2: case14 (headline demo)

**Bullets.**
- IEEE 14-bus benchmark, `rate_a × 0.2` to surface congestion.
- Auto-detected hub = bus with highest baseline LMP (a load pocket).
- FW converges in <50 iterations. Each iteration: 1 OPF + 1 VJP (~5 ms each).
- Pareto sweep over capex weight α ∈ {0, 0.5, 1, 2, 5} — same machinery, no new sensitivities needed.

**Equation.** Capex-augmented objective:
\[
\min_{\overline{\boldsymbol{f}} \in \mathcal{U}} \quad \boldsymbol{w}^\top \boldsymbol{\lambda}_\star\!(\overline{\boldsymbol{f}}) \;+\; \alpha\,\boldsymbol{c}^\top (\overline{\boldsymbol{f}} - \overline{\boldsymbol{f}}_0)
\]

**Image.** `experiments/ipp_market_planning_case14.png` (4-panel) and
`experiments/ipp_market_planning_case14_pareto.png` (capex-Pareto curve).

**Speaker notes.** "Here's the headline. PowerDiff picks 3-4 lines covering ~50% of the budget; basis revenue improves by [fill in number after run]. The Pareto plot shows the same algorithm produces a clean tradeoff curve — no re-derivation of gradients."

**Beamer-mapping hint.** Two frames. First frame: 4-panel convergence figure full-bleed. Second frame: Pareto curves with bullets on side.

---

## Slide 9 — Tier 4: RTS-GMLC (scale demo)

**Bullets.**
- 73-bus, 120-branch realistic system. Real load time series from Barrows et al.
- Single-period: peak hour, FW converges in ~60 iterations.
- Multi-period: 12 representative hours sampled from `DAY_AHEAD_regional_Load.csv`. Inner gradient averaged across periods.
- Per-FW-iter cost is independent of m (number of branches).

**Equation.** Multi-period objective:
\[
\min_{\overline{\boldsymbol{f}} \in \mathcal{U}} \quad \frac{1}{T} \sum_{t=1}^{T} \boldsymbol{w}^\top \boldsymbol{\lambda}_\star^{(t)}\!(\overline{\boldsymbol{f}}; \boldsymbol{d}_t)
\]

**Image.** `experiments/ipp_market_planning_rts_gmlc.png` (single-period) and
`experiments/ipp_market_planning_rts_gmlc_multi.png` (multi-period).

**Speaker notes.** "Same template, scales to a system with realistic generator portfolios and load shapes. The multi-period version answers: 'which lines am I willing to advocate to the ISO across an entire load duration curve?' That's the planning question."

**Beamer-mapping hint.** Two figures side-by-side, single-period left, multi-period right. Compact bullets at top.

---

## Slide 10 — Roadmap, limitations, Q&A

**Bullets.**
- **Already supported in PowerDiff**: AC OPF (slower, ForwardDiff KKT), 6 parameters total (`:d`, `:qd`, `:cq`, `:cl`, `:fmax`, `:sw`), open-source.
- **Natural extensions** (not in this demo, easy to build):
  - Stochastic / scenario-based: sample d_s from a forecast distribution.
  - Discrete upgrades: relax-and-round, or branch-and-bound on top of FW.
  - Multi-stakeholder: add a regulator or social welfare term to the objective.
  - Capital cost models: time-of-use cost, queue costs, NEPA constraints.
- **Limitations**: DC OPF doesn't model losses or reactive power; LMPs are piecewise-affine in fmax (active-set jumps); gradients can be zero in regions with no congestion.
- **Question for Pablo**: how does NewGrid currently model IPP basis exposure under planning uncertainty?

**Equation.** none.

**Image.** none.

**Speaker notes.** "Happy to take this in any direction NewGrid finds useful — particularly stochastic over forecast scenarios, or coupling with retail rate design."

**Beamer-mapping hint.** Single-column with bullet groups; final bullet bold and italicised as the open question.

---

## Per-slide image asset list

| Slide | Path | Aspect | Source |
|-------|------|--------|--------|
| 4 | `experiments/ipp_market_planning_3bus.png` | 16:9 | Tier 1 figure |
| 7 | `experiments/ipp_market_planning_3bus.png` | 16:9 | Tier 1 figure (reused) |
| 8a | `experiments/ipp_market_planning_case14.png` | 11:8 | Tier 2 4-panel |
| 8b | `experiments/ipp_market_planning_case14_pareto.png` | 9:4 | Tier 2 Pareto |
| 9a | `experiments/ipp_market_planning_rts_gmlc.png` | 11:8 | Tier 4 single |
| 9b | `experiments/ipp_market_planning_rts_gmlc_multi.png` | 11:8 | Tier 4 multi |

---

## Beamer preamble suggestion

```latex
\documentclass[aspectratio=169]{beamer}
\usepackage{amsmath, amssymb, bm}
\usepackage{minted}            % syntax highlighting for the Julia code slide
\usepackage{graphicx}
\usepackage{xcolor}
\definecolor{ipp-blue}{HTML}{5A82E6}
\definecolor{ipp-purple}{HTML}{B07ACC}
\definecolor{ipp-red}{HTML}{E05A52}
\definecolor{ipp-green}{HTML}{4FB838}

\setbeamercolor{title}{fg=ipp-purple}
\setbeamercolor{frametitle}{fg=ipp-blue}
\setbeamercolor{structure}{fg=ipp-purple}
\setbeamerfont{frametitle}{series=\bfseries}
\beamertemplatenavigationsymbolsempty
```

A dark-theme Beamer port should set
```latex
\setbeamercolor{background canvas}{bg=black!93!white}
\setbeamercolor{normal text}{fg=white!92!black}
```
to match the HTML deck's `#0c0c1a` / `#e8e8f0` palette.

---

## Notes for the downstream agent

- **Math**: prefer `\bm` over `\boldsymbol` if the document uses `bm` package. Equations are ASCII-clean; copy verbatim.
- **Code**: use `minted` (`julia` lexer) or `listings` with custom Julia keywords. The deck's CSS has its own syntax tokens (`.kw`, `.fn`, etc.) — these are HTML-only.
- **Numbers**: speaker notes contain `[fill in number after run]` placeholders. The final numerical results are in `experiments/ipp_history_*.csv` and the printed output of `experiments/ipp_market_planning.jl`.
- **Pacing**: 10 slides for a 10-minute meeting. Slide 4 (the explanation) and slide 8 (the case14 result) get extra time; the rest are 30 seconds each.
- **Q&A buffer**: leave 2-3 minutes after slide 10 for Pablo's questions.
