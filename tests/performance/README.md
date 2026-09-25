# Performance tests

Compare k6's HTTP baseline with four pyvespa feed APIs against the persistent
`vespa-team.pyvespa-performance.default` application in `aws-us-east-1c`.
Each test measures the token and the mTLS endpoint one after the other, so
every number is the instance's ceiling through that path rather than a share.

## Running

Install k6 on `PATH`, set `VESPA_TEAM_API_KEY` and `VESPA_CLOUD_SECRET_TOKEN`,
and place the authorized data-plane certificate/key pair in
`~/.vespa/vespa-team.pyvespa-performance.default/`.

```bash
uv run pytest tests/performance/ -m performance -s -v
```

The suite takes roughly 60–65 minutes and deletes documents in the shared
application at setup and teardown. Avoid overlapping runs; let the instance
settle after cleanup before comparing another run.

The instance is deployed once using `test_deploy_performance_instance.py`
(remove its `@unittest.skip` for a manual deployment). Normal tests reuse it.
CI runs through `.github/workflows/performance-cloud.yml`; mTLS credentials
come from `VESPA_PERFORMANCE_MTLS_CERT` and `VESPA_PERFORMANCE_MTLS_KEY`.

## What is measured

- k6 runs first and last, bracketing the pyvespa tests to measure instance drift.
- pyvespa tests `sync_feed_data_point`, `async_feed_data_point`, `feed_iterable`,
  and `feed_async_iterable`, with eight processes per transport to avoid a
  single Python process limiting throughput.
- Both lanes use the same payload, in-flight request count and connection
  count, with retries and compression disabled and a 120-second timeout.
  Each process/client multiplexes requests over one HTTP/2 connection.
- A 60-second k6 warmup on mTLS estimates capacity from its successful
  requests. Together with measured network RTT, it sets the concurrency of
  the active transport to keep about 200 requests queued in the instance:
  in flight = 200 + throughput × RTT.
- Closed loops count completions inside a 150-second window after 30 seconds
  of warmup. Batch APIs count completions while all worker processes are
  feeding; they have no per-request latency measurements.
- Tests wait for instance CPU to settle between workloads. Documents remain
  until teardown because deleting between tests causes background compaction.

Load settings and thresholds live in `utils/config.py`. `utils/k6_lane.py` and
`utils/pyvespa_lane.py` are the two lanes; `utils/cpu_probes.py` samples the
load generator's and the instance's CPU.

## Reading results

Both lanes enforce throughput floors, error limits and token/mTLS ratio
bounds. A run fails validity checks if more than 1% of requests receive 429,
the client held less than 85% of the configured concurrency in flight
(rps × mean latency, Little's law: the direct test of whether the client kept
the instance's queue full), runner CPU exceeds 90%, or container CPU is below
75%. Missing probes are reported as unknown and do not fail the run. pyvespa
must also reach 80% of the opening k6 throughput when that baseline is present.

If both lanes slow down, investigate the instance. If k6 stays steady and
pyvespa falls behind, investigate the client path and its CPU cost per request.
Treat gaps smaller than the opening-to-closing k6 drift as noise. Because the
transports run one at a time, token rps and mTLS rps are each an absolute
ceiling and their ratio is the capacity cost of the token path's auth hop; the
reported token extra latency is that hop in milliseconds. Compare CI runs with
CI, rather than local absolute numbers.

CI uploads JUnit XML, k6 summaries, per-method `*records.json`, `k6_drift.json`,
and `metrics.prom`. Set `PERFORMANCE_REPORT_DIR` to collect the same reports
locally. Thresholds are about 30% below CI run #34 (2026-09-23); recalibrate
load and floors when the application or its hardware changes.
