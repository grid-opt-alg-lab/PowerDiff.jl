# Contributing to PowerDiff.jl

Thanks for your interest in PowerDiff.jl. Julia 1.10 is the minimum supported
version; CI tests 1.10 and 1.11.

## Running the tests

The full suite spins up a temporary environment with the test-target
dependencies:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

To iterate in a REPL, load the package and include a test file. Test files
depend on PowerModels, ForwardDiff and Statistics, so they only run cleanly
once those are on the load path. `Pkg.test()` is the reliable path; for fast
iteration, `dev` the test dependencies into the project environment first, then:

```bash
julia --project=. -e 'include("test/runtests.jl")'   # runs everything
```

## Building the docs

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

`docs/Project.toml` points PowerDiff at the repository root, so there is no
separate develop step. The build renders `CHANGELOG.md` into
`docs/src/changelog.md`; that file is generated and is not tracked.

## Benchmarks

```bash
julia --project=benchmark -e 'include("benchmark/benchmarks.jl"); run(SUITE)'
```

The `benchmark` check is **skipped on pull requests from a fork**, by design.
A fork pull request receives a read-only `GITHUB_TOKEN`, so the step that posts
the comparison comment cannot succeed. Skipping it keeps the check neutral
instead of failing your pull request for a reason that has nothing to do with
your change. A maintainer can run the benchmarks on a branch of this repository
if a change needs them.

## Labels that change what CI runs

- `full-matrix` on a pull request runs the full test matrix (Julia 1.10 and
  1.11 on Linux, plus macOS/arm64). Add it whenever you change a
  `Project.toml` `[compat]` bound.
- `changelog skip` marks a pull request that needs no changelog entry, and
  keeps it out of the generated release notes.

## Style

There is no separate lint or format step. Follow the conventions already in the
file you are editing.

## Releases

Maintainers: see [RELEASING.md](RELEASING.md).
