# PowerIO Parser Contract

PowerIO is PowerDiff's parser and data layer. PowerDiff does not expose a parser
backend switch.

`PowerDiff.parse_file(path)` resolves the path, requires a MATPOWER `.m` file, and
calls `PowerIO.parse_file(path)`. `PowerDiff.parse_file(io; filetype="m")` reads
the stream and calls `PowerIO.parse_str(text, "matpower")`.

PowerIO returns a raw, lossless `Network`: MW/MVAr, degrees, original bus ids, raw
bus types, loads and shunts as first class records, and out of service elements
retained. PowerDiff then maps that `Network` into its own `ParsedCase` and keeps
the normalization it already owns:

- per unit scaling by `base_mva`
- degree to radian conversion
- bus type inference and slack selection
- out of service and isolated element filtering
- tap `0` to `1`
- angle bound normalization
- generator cost rescaling and padding
- `rate_a` fallback

PowerDiff rejects PowerIO networks carrying storage or HVDC/dcline records because
the current `ParsedCase` model has no fields for them.

The parser tests assert path and IO parity through this single PowerIO path.
