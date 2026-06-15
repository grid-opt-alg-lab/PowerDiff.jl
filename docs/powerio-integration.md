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

- polynomial cost interpretation: it reads the constant, linear, and quadratic
  coefficients straight from `to_powerdata`'s generator rows (already per-unit and
  right-aligned). PWL costs are rejected; higher-order polynomials are rejected by
  `to_powerdata` itself. A generator with no cost record is treated as cost-free.
- a finite `rate_a` fallback when the source leaves the thermal limit at `0`
- default angle-difference bounds

PowerDiff rejects networks carrying storage or HVDC/dcline records, which it does
not model.

The parser tests assert path and IO parity through this single PowerIO path.
