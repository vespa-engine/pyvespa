# Performance tests

Compare k6's HTTP baseline with the two pyvespa batch feed APIs against the persistent
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

The suite takes roughly 45–70 minutes and deletes documents in the shared
application at setup and teardown. Avoid overlapping runs; let the instance
settle after cleanup before comparing another run.

The instance is deployed once using `test_deploy_performance_instance.py`
(remove its `@unittest.skip` for a manual deployment). Normal tests reuse it.
CI runs through `.github/workflows/performance-cloud.yml`; mTLS credentials
come from `VESPA_PERFORMANCE_MTLS_CERT` and `VESPA_PERFORMANCE_MTLS_KEY`.

## What is measured

- A 30-second k6 probe with one request in flight per transport opens the
  session. Token p50 minus mTLS p50 there is the token path's extra hop
  without queueing; the runner's round trip cancels in the difference, so a
  slower token endpoint shows here first and independently of load.
- k6 runs first and last, bracketing the pyvespa tests. The pyvespa tests run
  10 to 50 minutes after the opening k6 run, and over that time the instance
  itself can change (compaction as the corpus grows, memory pressure, a noisy
  host neighbour). The closing run measures that movement with the same
  client, so a pyvespa-versus-k6 gap can be told apart from instance drift.
- pyvespa runs `feed_iterable` and `feed_async_iterable` the way a user calls
  them: one process, one call per transport, with the parameters in
  `utils/config.py`. The parameters (workers, connections, queue size) come from a
  CI sweep for the fastest single-process throughput; nothing internal is
  tuned. The worker counts are set high enough that the process stays
  CPU-bound rather than latency-bound, since hosted runners sit 4 to 32 ms
  from the instance and a latency-bound rate is just workers divided by the
  round trip. An eight-process variant of this lane matched k6 within 1 to
  4%, so a gap in this lane is Python-side cost per request, not the wire path.
- Both lanes use the same payload and HTTP/2, with retries and compression
  disabled and a 120-second timeout.
- A 60-second k6 warmup on mTLS estimates capacity from its successful
  requests. Together with measured network RTT, `LoadProfile.for_session`
  sets the k6 concurrency of the active transport to keep about 200 requests
  queued in the instance: in flight = 200 + throughput × RTT, rounded to
  whole connections and capped at `max_concurrency`. Transports run one at a
  time, so this is the whole load the instance sees. If the warmup or the RTT
  cannot be measured the session fails, since without a ceiling there is
  nothing to compare against.
- k6 counts completions inside a 150-second window after 30 seconds of
  warmup. pyvespa measures one batch of documents whole, after an untimed
  warmup batch; the batch APIs report no per-request latency.
- Tests wait for instance CPU to settle between workloads. Documents remain
  until teardown because deleting between tests causes background compaction.

Load settings live in `utils/config.py` and the thresholds in the test files.
`utils/k6_lane.py` and `utils/pyvespa_lane.py` are the two lanes, both
producing the `LaneResult` in `utils/metrics.py`; `utils/asserts.py` holds the
checks the tests fail on; `utils/cpu_probes.py` samples the load generator's
and the instance's CPU.

### Settings

All values live in `utils/config.py`, grouped by who uses them.

**Session** (`conftest.py`)

- `IDLE_CPU_UTIL` 0.30: before each test, wait (up to four minutes) until the
  busiest instance cluster is under 30% CPU, so cleanup or the previous test
  does not leak into the next measurement.
- `CLEANUP_SLICES` 16: parallel slices for `delete_all_docs`. A session
  leaves about 4 million documents; 8 slices took 30 minutes.

**k6 lane** (`test_k6_lane.py`)

- `PROFILE`: the opening and closing runs, a `LoadProfile` with the fields
  listed below, sized per session by `for_session`.
- `WARMUP`: mTLS only, 400 in flight, 15 s ramp and 45 s hold. Warms the
  instance and gives a conservative ceiling estimate from its successful
  requests; 400 on one transport stays under the 429 edge.
- `LATENCY_PROBE`: one request in flight on one connection, 5 s warmup and
  30 s hold, about a thousand samples per transport for the p50.
- `VALIDITY`: the four limits from "Reading results". In-flight is the direct
  check that the client kept the instance's queue full; the client CPU
  fraction is a backstop.

**pyvespa lane** (`test_pyvespa_lane.py`)

- `PYVESPA_METHODS`: the two batch APIs.
- `FEED_ITERABLE_PARAMETERS` and `FEED_ASYNC_ITERABLE_PARAMETERS`: `max_workers` 128
  and 400, `max_connections` 4 for the async API, everything else the library
  default. A CI sweep on 2026-09-28 found one process GIL-bound at about
  3200 rps for both APIs (from 64 and 128 workers respectively) and the
  default queue of 1000 better than both smaller (workers starve) and larger
  (4000 cost a third more CPU per request). The workers are set higher than
  the knee so the process stays CPU-bound on a runner up to 32 ms from
  the instance; a latency-bound process only does requests in flight divided
  by latency. HTTP/2 allows about 128 concurrent streams per connection, so
  the async API's default single connection caps it at 128 in flight whatever
  `max_workers` says; four connections lift that cap.

**Thresholds** (in the test files, next to the asserts)

- `K6_THRESHOLDS` in `test_k6_lane.py`: error rate at most 2%, floors of
  1400 token and 1600 mTLS rps, about 30% below the calibration run of
  2026-09-23 (2019 and 2271 rps). The ratio bounds (token at least 40% of
  mTLS rps, token p95 at most four times mTLS) are loose because the token
  path's extra latency varies with the round trip.
- `MAX_TOKEN_HOP_MS` 50 in `test_k6_lane.py`: bound on token p50 minus mTLS
  p50 at one in flight. Generous until a few runs have shown the usual value.
- `PYVESPA_THRESHOLDS` in `test_pyvespa_lane.py`: floors about 30% below the
  calibration run of 2026-09-28 (`feed_iterable` 2824 token and 3122 mTLS
  rps, `feed_async_iterable` 2215 and 2811); error and ratio bounds as for k6.

**`LoadProfile` fields**

- `server_queue_target` 200: requests kept queued inside the instance, the
  part of the in-flight count that does not depend on the network. 250 sat at
  the 429 edge on a far runner.
- `concurrency` 400: k6 in-flight requests before `for_session` has measured
  the ceiling and round trip; also the warmup load, safely under the 429 edge.
- `max_concurrency` 800: cap on what `for_session` can pick, so a bad round
  trip measurement cannot overload the instance.
- `warmup_s` 30 and `duration_s` 150: ramp-up excluded from counting, then the
  hold window. 150 s keeps drift within a run at 1 to 3%.
- `k6_connections` 8: HTTP/2 connections the k6 streams are spread over. The
  instance does not care between 1 and 16.
- `iterable_docs` 400000: one pyvespa batch per transport, about the hold
  window long at the roughly 3000 rps one process reaches.
- `iterable_warmup_docs` 2000: an untimed first batch so connection and TLS
  setup stay out of the measurement.

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
locally. Each floor comes from one calibration run (see Thresholds under
Settings) while runner speed varies between runs; recalibrate them, or set
them per `runner_cpu`, once several runs exist or when the application or its
hardware changes.

## Metrics reference

`.github/scripts/reports_to_prom.py` writes `metrics.prom` from the reports.
Nothing ships it to a Prometheus yet. `perf_run_info{commit, run_id, runner_cpu} 1`
identifies the run, and every sample carries `runner_cpu` (the CPU model from
`runner.md`), so pyvespa's numbers can be grouped by runner CPU and a point
can be joined to its commit through `perf_run_info`. Commit and run id are
not on every sample, since a label that changes every run makes a new series
each time.

`perf_<field>{source, lane, method, transport, concurrency}`, one sample
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
| `client_cpu_fraction` | 0..1 | load-generator CPU busy share of the whole machine during the window; one GIL-bound process shows about 1 divided by the vCPU count |
| `client_cpu_cores` | cores | pyvespa only: the process's own CPU time over wall time, the client load of one GIL-bound process |
| `server_container_cpu_util`, `server_content_cpu_util` | 0..1 | instance cluster CPU, peak sample covering the window |

`status_counts`, HTTP status to request count, is in the records file for
diagnosing a non-zero error rate but is not exported.

`perf_instance_drift_pct` (label: `runner_cpu`): closing k6 run's token + mTLS rps
relative to the opening one, in percent; the instance's own movement during
the session and therefore the noise floor for that run.

`k6_<metric>_<field>{source}`: every numeric field of the raw k6 summaries,
for diagnosis only; the `perf_*` series carry the same information.

## Graph suggestions

- Token hop: `perf_p50_ms{source="k6_token_hop_records",transport="token"} - ignoring(transport) perf_p50_ms{source="k6_token_hop_records",transport="mtls"}`.
  Independent of load, round trip and runner CPU; the one line that should be
  flat, and the first to move when the token endpoint gets slower.
- One panel per method: `perf_rps` for both lanes and transports, one point
  per run. Reference line: rolling median of the last five runs. Flag a run
  when it is more than 15% below the median; the instance itself moves about
  10% between runs (`perf_instance_drift_pct` shows its movement within one).
- Skip or annotate runs outside the validity limits (`rate_limited_rate` > 0.01,
  `achieved_in_flight / concurrency` < 0.85, `client_cpu_fraction` > 0.90,
  `server_container_cpu_util` < 0.75). Those runs are invalid, not regressions.
- Share of the ceiling per pyvespa method, from the same run:
  `sum(perf_rps{lane="pyvespa",method="feed_iterable"}) / sum(perf_rps{source="k6_token_vs_mtls_records"})`.
  A drop here with a steady k6 line is a pyvespa regression.
- `perf_cpu_ms_per_request` per method, grouped by runner CPU model: the
  client-efficiency trend, the place a pyvespa regression shows first.
- Token path cost per method: the exporter writes no derived series, Grafana
  subtracts the two transports itself:
  `perf_p95_ms{transport="token"} - ignoring(transport) perf_p95_ms{transport="mtls"}`.
