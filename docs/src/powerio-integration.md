# The PowerIO Contract

PowerIO is PowerDiff's parser and data layer, and the only one. There is no parser
backend switch, no second in-memory representation of a case, and nothing to
configure: a network reaches PowerDiff the way PowerIO hands it over, and the seam
between them is one function.

PowerDiff tracks **PowerIO 0.9** (powerio C ABI 5).

## Reading a case

```julia
using PowerDiff

net = parse_file("case14.m")     # -> PowerIO.BalancedNetwork
dc  = DCNetwork(net)
ac  = ACNetwork(net)
```

`PowerDiff.parse_file` resolves the path (including a case-library artifact, via
`library=`) and calls `PowerIO.parse_file`. PowerIO infers a path's format from its
extension unless `from` is given; a stream has no extension, so pass `from` (MATPOWER
is assumed if you do not).

**The format vocabulary is PowerIO's, not a copy of it.** `from` takes PowerIO's own
tokens, and a token PowerDiff does not recognize is handed over rather than refused,
so a reader PowerIO ships is usable from `parse_file` without a PowerDiff release.
An unknown token is answered by PowerIO with the set the linked library actually
reads. Two answers stay local because PowerIO cannot give them:

- A bare `json` names a container, not a reader, and PowerIO has several. Name the
  one you mean: `from=:powermodels`, `:egret`, `:pandapower`, `:goc3`, `:surge`,
  `:opfdata`.
- A distribution token (`:dss`, `:pmd`, `:bmopf`) parses to a
  `PowerIO.MulticonductorNetwork`, which is not a balanced transmission network.
  Lower it first — `PowerIO.to_package`, `PowerIO.lower_multiconductor_to_balanced`,
  `PowerIO.from_package` — and pass the balanced result in.

PowerDiff's own short spellings — `:m`, `:raw`, `:aux`, `:pm`, `:powermodels`,
`:egret` — still resolve to the PowerIO token they always did.

## One pass, one set of tables

`_network_data` is the whole seam. It runs PowerIO's normalize pass itself and reads
`PowerIO.to_powerdata` off the normalized network — one pass, not two, because
`to_powerdata` recognizes an already-normalized input and skips its own.

The result is memoized on the parsed network, so `DCNetwork(net)` and `ACNetwork(net)`
share a single ingest. They therefore cannot describe different cases, and the second
constructor is free.

Normalization gives PowerDiff per-unit scaling by `base_mva`, degree-to-radian
conversion, out-of-service and isolated-element filtering, bus-type inference, per-bus
load/shunt aggregation, and polynomial cost rescaling. PowerDiff layers on only the
OPF modeling PowerIO leaves to its consumer:

- **Polynomial cost interpretation.** The constant, linear, and quadratic
  coefficients come straight off the normalized generator rows, already per-unit and
  right-aligned. Piecewise-linear (model 1) costs are rejected; higher-order
  polynomials are rejected by PowerIO itself. A generator with no cost record is
  cost-free.
- **A finite thermal limit on every branch.** A case that states no limit — `0`, or
  PowerIO 0.9's non-finite spelling — gets one synthesized from the endpoint voltage
  limits and the angle window. That is the largest flow the network physically
  admits, so it bounds the problem without ever binding.
- **Default angle-difference bounds**, on the MATPOWER/PowerModels convention.
- **Absent reactive limits**, below.
- **Rejection of records PowerDiff does not model**: storage and HVDC/dcline. Both
  are checked against the raw network, so a case that declares them out of service is
  still refused rather than silently accepted.

The DC susceptance PowerDiff builds, `b = -x / (r² + x²)`, is the negated form of
powerio 0.9's default `SeriesImpedance` convention: the two read the same whole series
impedance and differ only by the Laplacian sign convention each documents.

## What PowerIO reports, as data

PowerIO has two things to say about a case, and `network_findings` returns both:

```julia
findings = network_findings(net)
findings.reader      # what the source format could not represent
findings.normalize   # what the normalize pass found or had to assume
```

Every line reads `CODE: message`. Split at the first `": "` and branch on the code;
the prose carries no stability promise.

PowerIO 0.9 raises the normalize findings as one `@warn` per distinct code from inside
`to_powerdata`. PowerDiff takes that pass over, so it reports them itself: same
one-per-code rule, said once for the network rather than once per constructor, and
under `PowerDiff.silence()`. Two are worth knowing by name:

- `CANONICALIZE.NORMALIZE.GEN_COST_ABSENT` — the case states no generator cost data,
  so any cost objective built from it is identically zero.
- `CANONICALIZE.NORMALIZE.REFERENCE_DESIGNATED` — the case named no reference bus, so
  one was chosen.

Reader findings are returned and never logged, matching PowerIO, which leaves them on
the parsed network for the consumer to read.

## Bounds a case may leave unstated

PowerIO 0.9 carries an absent numeric bound through as `±Inf` instead of refusing the
case. That is how MATPOWER, PowerModels, pandapower and PyPSA all spell "no limit",
and stock cases carry it — case9241pegase leaves the reactive limits off seven
generators.

PowerDiff's KKT layout is fixed, with one complementarity row per bound in a structure
that must not move with the data. An absent bound therefore cannot be dropped from the
system, and it cannot be carried either: `ρ · (qg − qmin)` with `qmin == -Inf` is
`0 · Inf`, a `NaN` in the residual and an `Inf` in the Jacobian.

So the row stays and states the right thing. **Generator reactive limits** model
absence end to end: the bound is left off the solver model, and the KKT row reads
`ρ = 0` — the multiplier of a constraint that is not there, which is exactly what a
solver reports for a bound it was never given. Its derivative is constant, so the
fixed-regime sensitivity of an absent bound is zero, as it should be.

Everywhere else a non-finite value is a modeling error, and PowerDiff names the
element and the field rather than letting it reach a factorization.

## Provenance

Bus rows carry the source bus id on `bus_i`, so `IDMapping.bus_ids` — and any
bus-indexed sensitivity `row_to_id` — map back to the input network. Generator and
branch `index` values are source row numbers among the unfiltered PowerIO rows, so
out-of-service rows leave gaps instead of renumbering the active ones.

## Tests

`test/test_parser_parity.jl` asserts path and IO parsing land on identical tables
through this single PowerIO path. `test/test_non_matpower_parsers.jl` round-trips a
MATPOWER case through PowerModels JSON, Egret JSON and PSS/E RAW and holds each to the
MATPOWER baseline. `test/test_powerio_integration.jl` covers the seam itself: format
routing, the one-pass ingest, findings, and absent bounds.
