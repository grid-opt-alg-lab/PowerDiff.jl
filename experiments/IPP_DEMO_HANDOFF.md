# IPP demo handoff

Working draft of a market-aware transmission upgrade demo for Pablo Ruiz at NewGrid.
Bilevel DC-OPF where the IPP optimizes line capacities to maximize basis revenue
between a hub and the rest of the system.

## What&rsquo;s here

| Path | Contents |
|---|---|
| `experiments/ipp_market_planning.jl` | Self-contained Julia script. Tier 1 (3-bus), Tier 2 (case14), Tier 4 (RTS-GMLC, single + 12-hour). |
| `experiments/Project.toml` | Pinned env. `develop`s local PowerDiff. |
| `experiments/ipp_pablo_deck.html` | 11-slide HTML deck. Embeds the PNG plots from the script. |
| `experiments/ipp_pablo_deck_beamer.md` | Per-slide outline for porting to Beamer/PowerPoint. |
| `experiments/ipp_market_planning_*.{pdf,png}` | Figures emitted by each tier. |
| `experiments/ipp_history_*.csv` | Per-iter Frank-Wolfe logs. |

## Run

```bash
# First time only — fetches CairoMakie + Makie (~10 min, mostly precompile)
julia --project=experiments -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'

# Run all tiers (~5-10 min)
julia --project=experiments experiments/ipp_market_planning.jl

# Run individual tiers from a REPL
julia --project=experiments
julia> include("experiments/ipp_market_planning.jl")
julia> run_tier1()                # 3-bus, < 5 s
julia> run_tier2()                # case14, ~10 s
julia> run_tier4()                # RTS-GMLC, ~3-5 min (single + 12-hour multi)
julia> run_tier4(run_multi_period=false)  # skip multi-period
```

The script will write `*.png` and `*.pdf` figures next to the script and
`ipp_history_*.csv` log files alongside.

## RTS-GMLC dataset

Tier 4 reads `~/Datasets/RTS-GMLC/RTS_Data/FormattedData/MATPOWER/RTS_GMLC.m` and
`~/Datasets/RTS-GMLC/RTS_Data/timeseries_data_files/Load/DAY_AHEAD_regional_Load.csv`
directly. On the Mac:

```bash
git clone --depth 1 https://github.com/GridMod/RTS-GMLC.git ~/Datasets/RTS-GMLC
```

If you put it elsewhere, edit `RTS_PATH` and `RTS_LOAD_CSV` near the top of
`run_tier4` in the script.

## View the HTML deck

```bash
# Mac:
open experiments/ipp_pablo_deck.html

# Or any browser, file:// URL.
# Keys: Space / ↓ / PageDown next, ↑ / PageUp prev. Side dots also clickable.
```

## Export to PowerPoint or Beamer

The HTML deck has a `@media print` style that flattens animations and forces
one slide per page. So:

### HTML &rarr; PDF (works everywhere)

```bash
# Open the HTML in Chrome / Safari / Firefox.
# Cmd+P → Save as PDF → Layout: Landscape → Background graphics: ON.
```

That PDF is good enough for emailing or projecting directly.

### PDF &rarr; PowerPoint (Mac)

Open the PDF in **Keynote**. Keynote imports each PDF page as a slide
(File → Open). Then File → Export To → PowerPoint. Editable slides on the other
side. Most reliable round-trip on macOS.

### Markdown &rarr; PowerPoint (alternate path)

`experiments/ipp_pablo_deck_beamer.md` is structured plain Markdown
(per-slide blocks with title, bullets, equations, image refs). Two options:

- **pandoc**: `pandoc -s ipp_pablo_deck_beamer.md -o deck.pptx` &mdash;
  heading-based, good for fast editing in PowerPoint, equations rendered via
  Office Math. (Some manual cleanup expected.)
- **Beamer / LaTeX**: the same Markdown is annotated with Beamer-mapping hints
  (frame layout per slide, equation blocks ready for `\[...\]`). Hand it to a
  Beamer generator agent or wrap by hand.

### HTML &rarr; PPTX direct

If you want one-shot conversion preserving layout, try
[`decktape`](https://github.com/astefanutti/decktape) or
[`marp`](https://marp.app/):

```bash
npx decktape automatic ipp_pablo_deck.html deck.pdf  # then PDF→PPTX as above
```

The Mac shortcut is still: print to PDF, open in Keynote, export to PPT.

## Status / what&rsquo;s done vs. open

**Done.**
- Tier 1 sanity FD verification (`max |analytical − FD| ≈ 3e-9`).
- Tier 2 case14 (rate_a × 0.10, gmin bump 0.01) Frank-Wolfe with Armijo
  backtracking and best-iterate tracking. Converges in ~80 iters, profit
  improvement ~+382 over baseline.
- Tier 2 capex-aware Pareto sweep over $\alpha \in \{0, 0.5, 1, 2, 5\}$.
- Tier 4 RTS-GMLC single + multi-period (multi-period uses 12 hour-of-day
  means from `DAY_AHEAD_regional_Load.csv`).

**Known issues / future improvements.**
- FW with linear cost ($c_q = 0$) gives identically-zero gradients within an
  active set. Tier 1 inlines a 3-bus net with $c_q = 1$ to demonstrate. The
  presentation flags this on slide 5.
- Active-set kinks make $w^\top \lambda_\star$ piecewise-affine; the simple
  $\gamma = 2/(k+2)$ FW step overshoots. We added Armijo backtracking and best-
  iterate tracking. Convergence is noisy but the best iterate is monotone.
  An exact line search (piecewise-linear minimizer along the FW segment) is a
  cleaner fix.
- Pablo&rsquo;s literal weights $w_i = +1\,(H), -1\,(\text{else})$ minimize the
  signed sum-spread, not strictly per-MWh portfolio profit (those differ by a
  factor of $|N|-|H|$ on the hub coordinate). For an actual IPP portfolio you
  may want $w_i = +1\,(\text{non-hub gen sites}), -k\,(\text{hub})$ where $k$
  matches contracted hub MW. Slide 4 explains.
- Tier 4 multi-period uses uniform per-bus scaling of demand by total
  hour-of-day load. RTS has 3 zones so this loses spatial detail. A zone-aware
  scaling using `bus["zone"]` would be a small refactor.
- The auto hub detector picks the highest-LMP bus. For a different IPP, set
  `HUB_OVERRIDE` at the top of the script (vector of original bus IDs).

## On committing the figures

The PDFs/PNGs in `experiments/` are committed-friendly (deterministic given
the seed; no PII). Privacy concerns:

- `~/Datasets/RTS-GMLC/` is **not** committed (it&rsquo;s local; clone separately).
- The script paths reference `~/Datasets/...` &mdash; only your home dir is mentioned,
  no credentials. Safe.
- The HTML deck&rsquo;s footer carries Sam&rsquo;s name and email (talkington@pm.me).
  Already public.

## One-line summary for Pablo

> Differentiable LMP layer + matrix-free transpose KKT solve + budget Frank-Wolfe.
> Per outer iteration: 1 OPF + 1 KKT solve, independent of network size.
> 73-bus RTS-GMLC, 12 hour-of-day periods, in &lt; 5 minutes wall clock.
