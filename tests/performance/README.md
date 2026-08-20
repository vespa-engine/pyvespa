# Performance Tests

Load tests for regression tracking of pyvespa-relevant serving paths, run
against a persistent Vespa Cloud application dedicated to this purpose:
`vespa-team.pyvespa-performance.default` (prod, `aws-us-east-1c`).

The zone is chosen to sit close to GitHub-hosted runners (US regions), which
generate the load in CI. This keeps network RTT small and stable, which matters
because the workload uses a closed model where throughput is gated by
round-trip time.

## Layout

| File | Purpose |
| --- | --- |
| `conftest.py` | Session fixture: resolves the mTLS and token endpoints of the deployed instance via the Vespa Cloud control plane, hands them to the tests, and deletes all fed documents in teardown so every run starts from a clean slate. |
| `k6_token_vs_mtls.js` | k6 workload: feeds documents through the mTLS and token endpoints with identical VU schedules and records per-endpoint latency, throughput, and error-rate metrics. |
| `test_token_vs_mtls.py` | Runs the k6 script as a subprocess, parses the summary, and asserts absolute and relative thresholds (throughput floors, error-rate ceiling, token-vs-mTLS latency and throughput ratios). |
| `test_deploy_performance_instance.py` | Manual, one-time deploy of the application package backing the tests. `@unittest.skip`'d so it never runs in CI. |

## Prerequisites

1. **k6** installed and on `PATH`
2. **Environment variables**:
   - `VESPA_TEAM_API_KEY`
   - `VESPA_CLOUD_SECRET_TOKEN`
   - `VESPA_CLIENT_TOKEN_ID` (optional, deploy only)
   defaults to the id used by the integration tests.
3. **mTLS data-plane certificate** for the application in
   `~/.vespa/vespa-team.pyvespa-performance.default/`. 

## Running locally

```bash
uv run pytest tests/performance/ -m performance -v
```

The k6 output is streamed to the console and the assertions print a short
`=== Results ===` summary at the end.

Note that every run feeds documents into (and afterwards deletes them from)
the shared prod instance, so avoid running concurrently with another run.
Local runs from Europe also see a higher RTT to `aws-us-east-1c` than CI
does, so absolute local numbers are not comparable with CI numbers.

## Running in CI

`.github/workflows/performance-cloud.yml` runs the suite weekly, on pushes to
master, on manual dispatch, and on PRs that touch the workflow file itself.
The runner image is pinned (`ubuntu-24.04`, the fixed 4 vcpu / 16 GB public-repo
flavor) and the k6 version is pinned, so the load-generator side stays constant
across runs. Required repository secrets: `VESPA_TEAM_API_KEY`,
`VESPA_CLOUD_SECRET_TOKEN`, and the data-plane pair
`VESPA_PERFORMANCE_MTLS_CERT` / `VESPA_PERFORMANCE_MTLS_KEY` (contents of the
`data-plane-public-cert.pem` / `data-plane-private-key.pem` files).

## One-time instance lifecycle

The application is deployed **once** and reused across all runs. Redeploy only
when the application package in `test_deploy_performance_instance.py` changes,
and by hand:

```bash
# Remove the @unittest.skip first, then:
uv run pytest tests/performance/test_deploy_performance_instance.py -s -v
```

## Follow-ups / known comparability gaps

- **Storage type is not pinned**: `<resources>` supports `storage-type`
  (`local`/`remote`) and `disk-speed`, which we leave unset, so the
  provisioner chooses. Pinning it (one more redeploy) would remove a disk
  performance variable from the feed-heavy workload.
- **Instance generation is not pinnable**: Vespa Cloud can move the node to a
  newer hardware generation with identical declared resources (e.g. on host
  retirement). If the trend graphs show an unexplained step change, this might
  be the cause.
