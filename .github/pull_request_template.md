## What this changes

<!-- What the change does, and why. Link any issue it closes. -->

## Changelog

- [ ] Added a bullet under `## [Unreleased]` in `CHANGELOG.md`.
- [ ] Not user-visible; labelled `changelog skip` instead.

## Checks

- [ ] `julia --project=. -e 'using Pkg; Pkg.test()'` passes locally.
- [ ] New exported names have docstrings and an entry in `docs/src/api.md`.
- [ ] A `Project.toml` `[compat]` change carries the `full-matrix` label.
