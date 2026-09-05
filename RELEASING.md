# Releasing PowerDiff.jl

For maintainers. PowerDiff is registered in the Julia General registry, so a
release is a registration; the tag and the GitHub release follow from it
automatically.

Every pre-1.0 minor bump is a breaking release.

## 1. Open a release pull request

In one pull request:

- set `version = "X.Y.Z"` in `Project.toml`;
- rename `## [Unreleased]` in `CHANGELOG.md` to `## [X.Y.Z] - YYYY-MM-DD`.

The renamed section needs at least one `- ` or `* ` bullet, and its text must
mention **breaking** or **changelog**. General's AutoMerge runs the
`guideline_breaking_explanation` check and holds a breaking release whose notes
do not explain the break. The Register workflow enforces both rules before it
triggers anything, so a mistake here costs a re-run, not a bad release.

Optionally, refresh the link list at the bottom of `CHANGELOG.md` first:

```bash
julia --project=docs -e 'using Changelog; Changelog.generate(Changelog.CommonMark(), "CHANGELOG.md"; repo = "grid-opt-alg-lab/PowerDiff.jl")'
```

That rewrites `CHANGELOG.md` in place, so run it locally and commit the result.

## 2. Review and merge

Merge the pull request and wait for CI to go green on `main`. That commit is
what gets registered, and the Register workflow re-runs the suite against it.

## 3. Read the tip of main

```bash
git fetch origin && git rev-parse origin/main
```

## 4. Run the Register workflow

Actions -> Register -> Run workflow, and paste the SHA from step 3 into
`expected_sha`. The workflow refuses anything that is not the current tip of
`main`, checks the version and changelog, checks the version against General
and against the existing tags, runs the tests, and posts one
`@JuliaRegistrator register` comment carrying the changelog section as the
release notes.

Set `dry_run` to run every check and print the comment without posting it.

## 5. Wait for AutoMerge

JuliaRegistrator opens a pull request against `JuliaRegistries/General`.
AutoMerge merges it after about ten minutes if the checks pass. Failures show
up as replies on the commit comment thread and on the General pull request.

## 6. Wait for TagBot

TagBot pushes the `vX.Y.Z` tag and publishes the GitHub release, usually within
the hour. If it does not, run Actions -> TagBot -> Run workflow; TagBot
re-scans every registered version and backfills whatever it missed.

## 7. Verify

- the `vX.Y.Z` tag exists and has a GitHub release attached;
- the release notes contain the changelog section;
- the Documentation workflow built docs for the tag (the `tags: ['v*']`
  trigger fires because TagBot pushes over SSH);
- `Pkg.add("PowerDiff")` resolves the new version in a clean environment.

---

If step 4 fails, fix whatever it names and run it again. It mutates nothing,
and it skips the trigger when the commit already carries one, so re-running is
safe and will not double-post.
