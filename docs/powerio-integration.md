# PowerIO Parser Contract

PowerIO is PowerDiff's parser and data layer. PowerDiff does not expose a parser
backend switch.

`PowerDiff.parse_file(path)` resolves the path, requires a MATPOWER `.m` file, and
returns a `PowerIO.Network` via `PowerIO.parse_file`. `PowerDiff.parse_file(io)`
reads the stream and calls `PowerIO.parse_str(text, "matpower")`. Pass the result to
[`DCNetwork`](@ref) or [`ACNetwork`](@ref).

The network constructors build directly from `PowerIO.to_powerdata(net)`, which
already returns normalized data: per-unit scaling by `base_mva`, degree-to-radian
conversion, out-of-service and isolated-element filtering, bus-type inference,
per-bus load/shunt aggregation, and polynomial cost rescaling. PowerDiff layers on
only the OPF modeling it owns:

- polynomial cost interpretation: constant, linear, and quadratic costs; PWL and
  higher-order polynomials are rejected. Costs are read from PowerIO's raw generator
  records because `to_powerdata` does not preserve coefficients declared with
  `ncost > 3`.
- a finite `rate_a` fallback when the source leaves the thermal limit at `0`
- default angle-difference bounds
- a reference bus chosen as the largest generator's bus when the source marks none

PowerDiff rejects networks carrying storage or HVDC/dcline records, which it does
not model.

The parser tests assert path and IO parity through this single PowerIO path.
