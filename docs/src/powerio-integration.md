# PowerIO Integration

PowerIO is PowerDiff's parser and data layer, and the only one. PowerDiff has no
parser backend switch and no second representation of a case, and it derives no
electrical quantity that PowerIO already states.

## Parsing

`PowerDiff.parse_file(path)` returns a `PowerIO.PioModule{PowerIO.BalancedNetwork}`.
The module carries more than the network: `m.value` is the case, `m.diagnostics`
holds the reader's findings, `m.sources[1].format` names the reader that ran, and
`m.history` records what has been done to it. Pass the module, or the network
inside it, to [`DCNetwork`](@ref) or [`ACNetwork`](@ref).

For paths the extension selects the reader unless `from` is given. For streams
pass `from`, because a stream has no extension; MATPOWER is assumed when neither
is given.

The format vocabulary is PowerIO's rather than a copy of it. PowerDiff normalizes
its own historical short spellings (`:m`, `:raw`, `:aux`, `:pm`, `:powermodels`,
`:egret`) and hands every other token straight over, so a reader PowerIO gains is
reachable without a PowerDiff release, and an unknown token is answered by PowerIO
with what the linked library actually reads. Two refusals stay local because
PowerIO cannot make them: a bare `json` names a container rather than a reader, and
a source that parses to anything other than a balanced transmission network is
refused by naming what it holds.

A failure inside PowerIO reaches the caller as a `PowerIO.PowerIOError`, which
carries the diagnostic `code` and the records behind it. PowerDiff does not wrap
it; wrapping would discard both.

## What PowerIO states and what PowerDiff adds

The network constructors read `PowerIO.to_powerdata`. It supplies per-unit powers
and ratings, radian branch angles and angle bounds, per-bus aggregated load and
shunt, right-aligned per-unit polynomial cost coefficients, each branch terminal's
own charging admittance, and the four terminal admittance coefficients from which
the series conductance and susceptance are read back exactly.

It is **unfiltered**: every generator and branch row carries a `status`, isolated
buses are present, and each row's `i` is its source row number. PowerDiff selects
what it models -- rows whose status is set, on buses that are not isolated -- and
that source row number becomes the [`IDMapping`](@ref) index, so an out-of-service
row leaves a gap instead of renumbering the rows that remain.

On top of that selection sit the four modeling decisions PowerIO leaves to its
consumer:

- polynomial cost interpretation, reading the cost model off the generator and
  refusing piecewise linear costs. Higher-than-quadratic polynomials are refused by
  `to_powerdata` itself.
- a finite thermal limit when the source states none. MATPOWER spells that `0` and
  PowerIO carries it out as `Inf`; both take the same synthesized limit, the largest
  flow the endpoint voltage limits and the angle window physically admit.
- default angle-difference bounds, mapping MATPOWER's `±360` and `0, 0` spellings to
  a `±60°` window.
- refusal of storage and HVDC records, which PowerDiff does not model.

## What PowerDiff computes and why it is not a duplicate

Two objects look like something PowerIO offers and are not.

`admittance_matrix(net, sw)` assembles `Y` from `g`, `b`, `tap`, `shift` and the
four terminal terms on every call. `PowerIO.calc_admittance_matrix` tabulates `Y`
for one network at its parsed values with a boolean in-service flag; PowerDiff
needs `Y` as a function of parameters a caller perturbs, because `sw` is continuous
in `[0, 1]` and `calc_sensitivity(state, :vm, :b)` and its nine siblings
differentiate through `g` and `b`. The same argument applies to the susceptance
Laplacian `B = AᵀWA` with `W = -b .* sw`, and to the branch flow matrix.

The branch-by-bus incidence matrix is stated once, in `_incidence_matrix`, and
shared by both network types. `PowerIO.calc_incidence_matrix` covers in-service
branches in table order against all buses and returns a bare sparse matrix with no
branch or bus index map, so relabeling it into PowerDiff's sorted-source-id space
costs more than stating it.

## Reading what PowerIO found

`m.diagnostics` is a `Vector{PowerIO.Diagnostic}`, available as soon as a case is
parsed and without emitting it again. A `Diagnostic` is a record, not a line of
text: branch on `d.code` and `d.severity`, and treat `d.message` as prose that
carries no stability promise.

```julia
m = parse_file("case14.m")
any(d -> d.severity === :warning, m.diagnostics)
```

PowerDiff logs none of it. Writing a case back out in another format goes through
PowerIO directly, `PowerIO.emit(m, "psse", path)`.
