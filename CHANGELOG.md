# Changelog

## Unreleased

Tracks **PowerIO 0.9** (powerio C ABI 5). The `[compat]` bound is now
`PowerIO = "0.9"`; 0.9 gates its C ABI handshake on equality, so the binding and its
binaries move together and this bound cannot be relaxed.

- The format vocabulary is PowerIO's rather than a copy of it. `parse_file` hands an
  unrecognized `from` token to PowerIO instead of refusing it against a five-format
  allowlist, so every transmission reader PowerIO ships — pandapower, PyPSA, PSLF,
  PowerWorld binary, gridfm, GO Challenge 3, Surge, OPFData, PSS/E 34 and 35 — is
  reachable without a PowerDiff release, and an unknown token is answered by PowerIO
  with the set the linked library actually reads. PowerDiff's own short spellings
  (`:m`, `:raw`, `:aux`, `:pm`, `:powermodels`, `:egret`) resolve as before. A bare
  `json` is still refused as ambiguous, now naming more of the readers it could mean,
  and a distribution token (`:dss`, `:pmd`, `:bmopf`) is refused with the lowering
  step to run first.
- One normalize pass per network, memoized. PowerDiff runs `PowerIO.to_normalized`
  itself and reads `to_powerdata` off the normalized network, which skips its own
  pass. The tables are cached on the parsed network, so `DCNetwork(net)` and
  `ACNetwork(net)` share one ingest — they can no longer describe different cases, the
  JSON payload is materialized once, and the second constructor is free.
- New exported `network_findings(net)`: what PowerIO reported about a case, as data.
  Returns `(; reader, normalize)` — the parser's fidelity findings and the normalize
  pass's — as `CODE: message` lines. PowerIO 0.9 raises the normalize findings as one
  `@warn` per distinct code from inside `to_powerdata`; PowerDiff owns that pass now,
  so it reports them itself, once per network rather than once per constructor, and
  under `PowerDiff.silence()`.
- Generator reactive limits may be absent. PowerIO 0.9 carries a bound the case does
  not state as `±Inf` rather than refusing the case (stock case9241pegase leaves them
  off seven generators). PowerDiff leaves the bound off the solver model and its KKT
  complementarity row reads `ρ = 0`, the multiplier of a constraint that is not there
  and the value a solver reports for a bound it was never given. Without this the
  upgrade would have turned a clean refusal into `0 * Inf` — a `NaN` in the residual
  and an `Inf` in the Jacobian. The KKT sparsity pattern is unchanged.
- A branch whose rating is non-finite is treated as unrated, taking the same
  synthesized thermal limit as `rate_a == 0`.
- Every other value read out of a case must be finite, and a non-finite one is now
  named with its element and field rather than reaching a factorization. `ACNetwork`
  applies the same rule to a caller-built table.
- `PowerIO.source_format` reports the lowercase token every `from` argument accepts
  (`"powermodels-json"`, not `"PowerModelsJson"`), a powerio 0.9 change; the parser
  tests assert the new spelling.

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
