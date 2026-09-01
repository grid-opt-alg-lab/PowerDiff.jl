# Changelog

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
