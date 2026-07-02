# Testing & Dependency Strategy

How and why we test pyvespa and keep dependencies current. Update when the
strategy changes, not just the code.

## Guiding principle

> If the full test suite passes against the latest allowed dependency versions,
> the risk of taking those updates is low enough to merge.

We don't review dependency diffs by hand; the test suite is the gate. It is only
as good as our coverage, so invest in coverage.

## What we test

| Suite | Location | CI workflow | Python | Needs |
|-------|----------|-------------|--------|-------|
| Unit | `tests/unit/` | `pyvespa-unit-tests.yml` | 3.10-3.13 x {ubuntu, windows, macos} | none (mocked) |
| Integration (local) | `tests/integration/` (docker, grouping, queries, evaluation) | `integration-except-cloud.yml` | 3.10 | Docker + Vespa |
| Integration (cloud) | `tests/integration/` (vespa_cloud*, mteb) | `integration-cloud.yml` | 3.10 | Vespa Cloud secrets |
| Doctests | `tests/mktestdocs/` | `mktestdocs.yml` | 3.10 | none |
| Notebooks (local) | `docs/` notebooks | `notebooks-except-cloud.yml` | 3.11 | none |
| Notebooks (cloud) | `docs/` notebooks | `notebooks-cloud.yml` | 3.11 | Vespa Cloud + model API keys |

Unit, local integration, doctests, and local notebooks run on every
`pull_request` to `master`. The two **cloud** suites are path-filtered: they
trigger only on PRs touching their own workflow, a few specific files, or
`uv.lock` (added so dependency PRs run them); otherwise they run on a weekly cron.
A dependency PR changes `uv.lock`, so it runs the full matrix. "Full test suite"
means all of the above green on a PR.

## Lockfile and install styles

pyvespa is a published library: consumers resolve from our `pyproject.toml`
ranges and never see `uv.lock`. The lockfile governs only our dev/CI.

CI uses two install styles on purpose:

- **`pip install -e .[extra]`** (unit matrix, integration-except-cloud): latest
  versions allowed by `pyproject.toml`. Our real-world canary for what users get.
- **`uv sync`** (mktestdocs, notebooks-cloud, integration-cloud, copilot-setup):
  locked versions from `uv.lock`, for reproducibility.

We keep `uv.lock` committed: it gives reproducible dev/CI and is the single
artifact we bump (one re-lock updates every transitive dep in one PR). The
trade-off is that pinned CI can hide breakage users on floating ranges hit, which
is why the unit matrix stays floating on `pip install`.

## Keeping dependencies up to date

Goal: few, batched, test-gated updates, not a stream of per-dependency PRs.

- **Python** (`update-deps.yml`): weekly `uv lock --upgrade`, unit-test gate, one
  PR on `deps/weekly-uv-upgrade`. The `uv.lock` change triggers the full matrix
  (cloud suites included, via their `uv.lock` path filter); merge once green. Uses
  a short-lived GitHub App token (`DEPS_BOT_APP_ID` / `DEPS_BOT_APP_PRIVATE_KEY`),
  since PRs made with `GITHUB_TOKEN` don't trigger CI.
- **GitHub Actions** (`dependabot.yml`): SHA-pinned in workflows; Dependabot keeps
  them current, grouped into one weekly PR.
- **Security**: enable Dependabot security *updates* (not just alerts) in repo
  settings. These open targeted CVE PRs independent of `dependabot.yml`, so a
  vulnerable Python dep is still bumped immediately even though we don't run a
  `pip` ecosystem for routine version updates (that noise is what the weekly
  re-lock replaces).

## Pinning policy in `pyproject.toml`

- Prefer floating lower bounds (`>=`) so the floating CI stays meaningful.
- Exact pins (`==`) only where reproducibility matters (build toolchain:
  `setuptools`, `build`, `twine`) or to dodge a known-bad release.
- `requests` / `httpx` are transitional (see CLAUDE.md, `httpr` migration); remove
  when done.

## Open items

- Consider moving remaining `pip install` workflows to `uv`, keeping the unit
  matrix floating.
- Consider auto-merge on the weekly deps PR once we trust the gate.
