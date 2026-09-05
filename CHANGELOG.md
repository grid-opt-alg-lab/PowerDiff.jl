# Changelog

## Unreleased

Tracks **PowerIO 0.11** (powerio C ABI 7). The `[compat]` bound is now
`PowerIO = "0.11"`; the binding gates its ABI handshake on equality, so it and its
binaries move together and the bound cannot be relaxed.

PowerIO 0.11 replaces the accessor-and-JSON-payload layer with typed element
tables, so this is a rewrite of the seam rather than a version bump. PowerDiff is a
consumer of PowerIO: every electrical quantity PowerIO states is now read from it,
and nothing that PowerIO computes is computed again here.

### Changed

- **Breaking:** `parse_file` and `parse_matpower` return a
  `PowerIO.PioModule{PowerIO.BalancedNetwork}` rather than a bare network. The
  module carries the reader's diagnostics, the source record and the history
  alongside the case, and it is what `PowerIO.emit` writes back out. Every network
  constructor accepts the module or the network inside it, so `DCNetwork(net)`,
  `ACNetwork(net)`, `DCOPFProblem(net)`, `ACOPFProblem(net)`,
  `DCPowerFlowState(net)` and `calc_demand_vector(net)` are unchanged at the call
  site.
- **Breaking:** a failure inside PowerIO reaches the caller as a
  `PowerIO.PowerIOError` instead of being flattened into an `ArgumentError`. It
  carries the diagnostic `code` and the records behind it, both of which wrapping
  discarded. PowerDiff's own refusals stay `ArgumentError`.
- **Breaking:** `parse_matpower_struct` is removed. It was a compatibility alias
  for `parse_matpower` with no callers.
- **Breaking:** `network_findings` is removed, one release after it was added and
  before it ever shipped. It existed because PowerIO 0.9 reached its findings only
  through a normalize pass PowerDiff had to own, deduplicate and label. In 0.11 they
  are `m.diagnostics`, a property of the module `parse_file` already returns, and
  a `Diagnostic` is a record with a `code` and a `severity` rather than a line of
  text. Wrapping a property in an exported function is surface, not integration.
- The series conductance and susceptance are read back from PowerIO's terminal
  admittance coefficients rather than derived from `r` and `x`, and each branch
  terminal's charging admittance is carried on its own side. PowerDiff previously
  summed the two sides, split the total evenly and discarded the charging
  conductance, which lost fidelity on every source that states the terminals
  separately. Nothing in `src/` inverts a branch impedance any more.
- The branch-by-bus incidence matrix is stated once and shared by both network
  types, which previously assembled one each.
- Out-of-service and isolated elements are selected out here rather than by
  PowerIO. `to_powerdata` is unfiltered in 0.11: every row carries a `status` and
  its source row number, which is exactly the `IDMapping` index, so the two-pass
  reconciliation that used to recover those numbers is gone. Isolated (`type == 4`)
  buses and everything standing on them are dropped, as before.
- A generator's cost model is read off the element rather than inferred from the
  shape of a converted row, so a piecewise linear cost is refused for what it is.
- The format vocabulary is PowerIO's rather than a copy of it: an unrecognized
  token goes to PowerIO, which answers with what the linked library actually reads,
  and a reader PowerIO gains is reachable without a PowerDiff release. PowerDiff's
  historical short spellings still resolve. A bare `json` is still refused as
  ambiguous. The hand-maintained distribution-format blocklist is replaced by a
  check on what the source actually parsed to, which also covers time series,
  scenario sets and calculation instances.
- `docs/powerio-integration.md` moves to `docs/src/` and is published; it stated the
  ingest contract but sat outside the docs build, so it was never rendered.

### Fixed

- A branch whose source states no thermal limit takes the synthesized limit whether
  the source spells that `0` or `Inf`. PowerIO 0.11 carries MATPOWER's `rate_a == 0`
  out as `Inf`, which the previous `rate_a > 0` test accepted, so every unrated
  branch would have reached the solver with an unbounded flow. Stock IEEE 300 leaves
  all 411 branches unrated.
- Generator reactive limits may be absent. PowerIO carries a bound the case does not
  state as `±Inf` rather than refusing the case, and stock case9241pegase leaves them
  off seven generators. PowerDiff leaves the bound off the solver model and its KKT
  complementarity row reads `ρ = 0`, the multiplier of a constraint that is not
  there and the value a solver reports for a bound it was never given. Without this
  the upgrade would have turned a clean refusal into `0 * Inf` — a `NaN` in the
  residual and an `Inf` in the Jacobian. The KKT sparsity pattern is unchanged.

### Added

- A test that the ingest still reproduces MATPOWER's network, comparing
  `ptdf_matrix` against `PowerModels.calc_basic_ptdf_matrix` on three PGLib cases.
  It shares no code with the ingest and pins the topology, the branch susceptances
  and the reference bus together.

## 0.1.0

First release on the PowerIO 0.6.x data layer, and the first cut through the
one-click `register.yml` release workflow.

- Migrated to PowerIO 0.6.x: the `[compat]` bound is now `PowerIO = "0.6"`, and
  the deprecated `PowerIO.Network` type was replaced by `PowerIO.BalancedNetwork`
  (renamed upstream in PowerIO 0.3.0). The parsing/data contract is unchanged —
  `PowerIO.to_powerdata`'s row schema is identical across the two versions, so
  the `_network_data` adapter and everything downstream are unaffected.
- Release tooling: TagBot now carries the `contents: write` permission it needs
  to publish GitHub releases (previously it produced a bare tag with no release
  object), and a `register.yml` workflow performs the version bump and
  JuliaRegistrator trigger in one manual dispatch, drawing release notes from
  this changelog.
