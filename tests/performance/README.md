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

- k6 runs first and last, bracketing the pyvespa tests. The pyvespa tests run
  10 to 50 minutes after the opening k6 run, and over that time the instance
  itself can change (compaction as the corpus grows, memory pressure, a noisy
  host neighbour). The closing run measures that movement with the same
  client, so a pyvespa-versus-k6 gap can be told apart from instance drift.
- pyvespa runs `feed_iterable` and `feed_async_iterable` the way a user calls
  them: one process, one call per transport, with the knobs in
  `utils/config.py`. The knobs (workers, connections, queue size) come from a
  local sweep for the fastest single-process throughput; nothing internal is
  tuned. An eight-process variant of this lane matched k6 within 1 to 4%, so
  a gap in this lane is Python-side cost per request, not the wire path.
- Both lanes use the same payload and HTTP/2, with retries and compression
  disabled and a 120-second timeout.
- A 60-second k6 warmup on mTLS estimates capacity from its successful
  requests. Together with measured network RTT, it sets the k6 concurrency of
  the active transport to keep about 200 requests queued in the instance:
  in flight = 200 + throughput × RTT.
- k6 counts completions inside a 150-second window after 30 seconds of
  warmup. pyvespa measures one batch of documents whole, after an untimed
  warmup batch; the batch APIs report no per-request latency.
- Tests wait for instance CPU to settle between workloads. Documents remain
  until teardown because deleting between tests causes background compaction.

Load settings and thresholds live in `utils/config.py`. `utils/k6_lane.py` and
`utils/pyvespa_lane.py` are the two lanes; `utils/cpu_probes.py` samples the
load generator's and the instance's CPU.

## Reading results

Both lanes enforce throughput floors, error limits and token/mTLS ratio
bounds. k6 must also pass the validity checks that prove the instance was the
bottleneck: no more than 1% of requests receive 429, the client held at least
85% of the configured concurrency in flight (rps × mean latency, Little's law:
the direct test of whether the client kept the instance's queue full), runner
CPU at most 90%, container CPU at least 75%. pyvespa prints the same evidence
without asserting it, since one Python process is the bottleneck by design,
and reports the share of the session's k6 ceiling each method delivers.
Missing probes are reported as unknown and do not fail the run.

If both lanes slow down, investigate the instance. If k6 stays steady and
pyvespa falls behind, investigate the client path and its CPU cost per request.
Treat gaps smaller than the opening-to-closing k6 drift as noise. Because the
transports run one at a time, token rps and mTLS rps are each an absolute
ceiling and their ratio is the capacity cost of the token path's auth hop; the
difference between k6's token and mTLS latency is that hop in milliseconds.
Compare CI runs with CI, rather than local absolute numbers.

CI uploads JUnit XML, k6 summaries, per-method `*records.json`, `k6_drift.json`,
and `metrics.prom`. Set `PERFORMANCE_REPORT_DIR` to collect the same reports
locally. Thresholds are about 30% below CI run #34 (2026-09-23); recalibrate
load and floors when the application or its hardware changes.

## Metrics reference

`.github/scripts/reports_to_prom.py` writes `metrics.prom` from the reports.
Nothing ships it to a Prometheus yet; whoever wires that up should attach the
commit, workflow run id and runner CPU model (see `runner.md`) as labels, since
the exporter does not.

`perf_<field>{source, lane, method, transport, http, concurrency}`, one sample
per records file, lane, method and transport. Labels: `source` is the records
file (the opening and closing k6 runs differ only here), `lane` is `k6` or
`pyvespa`, `method` is `http_post` or the pyvespa method, `concurrency` the
configured in-flight requests for k6 and `max_workers` for pyvespa. Fields:

| field | unit | meaning |
| --- | --- | --- |
| `rps` | 1/s | requests completed per second inside the measurement window |
| `error_rate` | 0..1 | share of those requests not answered 200 |
| `requests` | count | requests counted in the window |
| `duration_s` | s | window length: the 150 s hold for k6, the batch's wall time for pyvespa |
| `p50_ms`, `p95_ms`, `p99_ms`, `mean_ms` | ms | latency, k6 only; the batch APIs time nothing per request |
| `achieved_in_flight` | requests | k6 only: `rps × mean_ms / 1000` (Little's law); divided by `concurrency` it says whether the client kept the instance's queue full |
| `cpu_ms_per_request` | ms | load-generator CPU per request, pyvespa only; depends on the runner CPU model |
| `rate_limited_rate` | 0..1 | share of requests answered 429 |
| `client_cpu_fraction` | 0..1 | load-generator CPU busy share during the window |
| `server_container_cpu_util`, `server_content_cpu_util` | 0..1 | instance cluster CPU, peak sample covering the window |

`perf_instance_drift_pct` (no labels): closing k6 run's token + mTLS rps
relative to the opening one, in percent; the instance's own movement during
the session and therefore the noise floor for that run.

`k6_<metric>_<field>{source}`: every numeric field of the raw k6 summaries,
for diagnosis only; the `perf_*` series carry the same information.

## Graph suggestions

- One panel per method: `perf_rps` for both lanes and transports, one point
  per run. Reference line: rolling median of the last five runs. Flag a run
  when it is below the median by more than that run's `perf_instance_drift_pct`.
- Skip or annotate runs outside the validity limits (`rate_limited_rate` > 0.01,
  `achieved_in_flight / concurrency` < 0.85, `client_cpu_fraction` > 0.90,
  `server_container_cpu_util` < 0.75). Those runs are invalid, not regressions.
- Share of the ceiling per pyvespa method, from the same run:
  `sum(perf_rps{lane="pyvespa",method="feed_iterable"}) / sum(perf_rps{source="k6_token_vs_mtls"})`.
  A drop here with a steady k6 line is a pyvespa regression.
- `perf_cpu_ms_per_request` per method, grouped by runner CPU model: the
  client-efficiency trend, the place a pyvespa regression shows first.
- Token path cost per method: the exporter writes no derived series, Grafana
  subtracts the two transports itself:
  `perf_p95_ms{transport="token"} - ignoring(transport) perf_p95_ms{transport="mtls"}`.
