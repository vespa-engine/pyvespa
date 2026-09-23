# Performance Tests

Load tests for regression tracking of pyvespa-relevant serving paths, run
against a persistent Vespa Cloud application dedicated to this purpose:
`vespa-team.pyvespa-performance.default` (prod, `aws-us-east-1c`).

The zone is chosen to sit close to GitHub-hosted runners (US regions), which
generate the load in CI. This keeps network RTT small and stable, which matters
because the workloads use a closed model where throughput is gated by
round-trip time.

## Two lanes

1. **k6 lane** (`test_k6_token_vs_mtls.py` + `k6/token_vs_mtls.js`): raw HTTP
   through k6's Go stack — the endpoint ceiling with no pyvespa involvement.
2. **pyvespa lane** (`test_pyvespa_token_vs_mtls.py` + `utils/loadgen.py`):
   the real pyvespa code paths — `VespaSync.feed_data_point` (thread closed
   loop), `VespaAsync.feed_data_point` (asyncio closed loop), `feed_iterable`,
   and `feed_async_iterable` — each spread over `LoadProfile.processes` worker
   processes per transport.

The point of both lanes is to load the **instance** past what it can absorb,
so the instance is always the bottleneck and never the client library or the
runner. Both lanes reduce to the same `LaneResult` schema and assertion set
(`utils/metrics.py`), run the *same* `LoadProfile.concurrency` in-flight
requests per transport (k6 VUs; pyvespa threads/coroutines summed over its
processes), share warmup/measurement windows (30 s + 150 s, injected into k6
via env vars), use the identical document payload, and load token and mTLS
**concurrently** so both lanes measure under the same total load. Both lanes
count only requests that *complete* inside the 150 s window, so
`rps = count / 150` and latency percentiles exclude ramp-up and shutdown tails
in k6 exactly as in pyvespa. With the instance as the bottleneck, pyvespa rps
should land close to k6 rps; a persistent gap is a client finding. The
iterable methods are batch APIs: they report wall-clock throughput only
(per-request latency is not observable from the callback).

**Why processes.** One Python process (one GIL) tops out around 2000 feed
requests/s on a hosted runner and moves ~1.8x with the CPU model the runner
lands on, which is below the instance ceiling. `utils/loadgen.py` therefore
spawns `processes` (default `min(8, cpu_count)`) workers per transport, each
running `concurrency / processes` workers with its own `Vespa` clients, and
merges the samples. Each result also carries `cpu_ms_per_request`: the client
CPU cost per request summed over processes, the largely runner-independent
number that tracks pyvespa's own efficiency.

**Setting the concurrency.** The instance does not feel the client's
in-flight count, it feels how many requests are queued inside it: with the
ceiling at X rps and a network RTT of r, N in flight means about N - X * r
queued and the rest on the wire. The same N therefore overloads from a 50 ms
GitHub runner (CI run #32: 429s at 400 per transport) and under-loads from
130 ms away. The session fixture measures both, X from the 60 s k6 warmup and
r as the minimum of 20 sequential tiny GETs, and derives N so that
`LoadProfile.server_queue_target` (250) requests sit inside the instance:
about 200 per transport from us-east, about 400 from Europe. Every result
records the concurrency it ran at. `LoadProfile.concurrency` (400) is only
the warmup and fallback value. The `k6_sweep` dispatch input still runs
`test_k6_sweep.py` alone at fixed levels to re-find the knee and the
container CPU at saturation after an instance change.

## Fairness and validity

Both lanes are pinned to the same client behaviour so the only difference is
the HTTP stack, and every result carries evidence that the instance was the
bottleneck:

| Behaviour | k6 | pyvespa |
| --- | --- | --- |
| Retries | none | none: `num_retries_429=0` (sync paths), `docv1_retry_policy=NO_RETRY` (async paths) |
| 429 | counted as a failed, rate-limited request | same; `rate_limited_rate` on both |
| Request compression | none | `compress=False` |
| Response compression | none requested (k6 sends no `Accept-Encoding`) | `Accept-Encoding: identity` pinned via `additional_headers`; pyvespa's default asks for zstd/gzip/deflate/br and the container gzips responses |
| Request timeout | 120 s | 120 s (httpr default in pyvespa) |
| HTTP version | HTTP/2 (`res.proto` verified) | HTTP/2 in both `negotiate` and `h2only` modes (httpr negotiates h2 via ALPN, verified) |
| Ramp / warmup | 30 s VU ramp, dropped | 30 s, dropped |
| Instance warmth | 60 s k6 warmup at target concurrency before the first test (session fixture), results discarded | same |
| Window | completions inside the 150 s hold | closed loops: same; batch methods: completions while every worker process is still feeding |
| Order | first and last test of the session (`perf_last`); the two runs bracket the pyvespa methods and their difference is printed and stored as instance drift (`k6_drift.json`, `perf_instance_drift_pct`) | between the two k6 runs |
| In-flight per transport | `LoadProfile.concurrency` VUs | `LoadProfile.concurrency` workers over `processes` processes |
| Connections per transport | `processes` VUs, each keeping `concurrency / processes` requests in flight over its one HTTP/2 connection (`http.asyncRequest`, `STREAMS_PER_CONNECTION`) | `processes` worker processes, each one httpr client multiplexing `concurrency / processes` requests over one HTTP/2 connection (verified with `lsof`: 50 threads on one `VespaSync` = 1 connection; `pool_maxsize` is ignored by httpr) |
| Corpus | empty at session start, grows within the session; each test starts once every node is idle (`IDLE_CPU_UTIL`) | same |

Two saturation checks, one per side. `utils/metrics.py::assert_measurement_valid`
fails a test (both lanes, same `VALIDITY` limits) when the run cannot be about
the instance: more than 1% of
requests were 429 (overloaded, not saturated: lower the concurrency), the load
generator used more than 70% of the runner's CPU (`client_cpu_fraction`, from
`/proc/stat` over the window; the client was the bottleneck), or, once set
from the sweep, the container node's `cpu_util` right after the window was
below `min_server_container_cpu_util` (instance not saturated: raise the
concurrency). Instance CPU comes from the application's own data-plane
`/prometheus/v1/values` (`utils/saturation.py`) and is recorded per result as
`server_container_cpu_util` / `server_content_cpu_util`. The probe is the
metrics proxy's ~60 s snapshot (peak over the snapshots that cover the load),
so saturated runs read 83-96%; the floor is 75%, against 70% at half the
concurrency. The second check is probe-independent: every pyvespa method must
reach `MIN_PYVESPA_VS_K6_RATIO` (80%) of the opening k6 run's total in the same
session, which is what a client-bound path fails.

Connection count matters to the instance. With 400 single-stream connections
per transport (k6, one per VU) the container delivered ~3400-3750 rps at 94%
CPU; with 8 connections each multiplexing 50 requests (what every pyvespa path
does, since httpr puts all of a client's requests on one HTTP/2 connection) it
delivered ~4400-4500. Fewer connections cost the container less per request.
Both lanes therefore run the same shape, `processes` connections with
`concurrency / processes` streams each, k6 via `http.asyncRequest` inside
`processes` VUs. Set `STREAMS_PER_CONNECTION=1` in the k6 env to measure the
one-connection-per-request shape instead. pyvespa also parses every response
body; that shows up in `cpu_ms_per_request`, not in throughput once the
instance is the limit.

Deleting between tests was tried and dropped (`PERFORMANCE_CLEAN_BETWEEN_TESTS=1`
turns it back on): removing ~600k documents before each test left the content
node compacting and pruning for minutes, the container thread pool filled up
waiting on it, and the closing k6 run of that session fell to half the opening
one. A growing corpus within a session is the smaller state change. Every test
now waits until all nodes are at or below `IDLE_CPU_UTIL` before starting.

## Results with the final shape (2026-09-22, from Europe)

Concurrency 400 per transport over 8 connections, 60 s warmup, idle wait
before each test, retries and compression off on both sides.

| Lane / method | token rps | mTLS rps | total | container CPU | vs k6 opening |
| --- | --- | --- | --- | --- | --- |
| k6 opening | 2149 | 2311 | 4460 | 87% | 1.00 |
| k6 closing (same session) | 2108 | 2222 | 4329 | 87% | 0.97 (drift -2.9%) |
| pyvespa sync_feed_data_point | 2181 | 2267 | 4448 | 87% | 1.00 |
| pyvespa async_feed_data_point | 2159 | 2350 | 4509 | 89% | 1.01 |
| pyvespa feed_iterable | 1702 | 1924 | 3625 | 95% | 0.81 |

The pyvespa rows are from the preceding session (its own k6 opening run was
still in the 400-connection shape at 3750). k6 and the two closed-loop pyvespa
paths agree within 2%, inside the measured drift: the same saturated instance,
measured by two different clients, gives the same number. `feed_iterable` is
the one path with a real cost: ~20% below the others at the highest client CPU
per request (0.52 ms), worth a look in pyvespa. `feed_async_iterable` needs a
clean re-measurement; its last run followed the per-test deletes.

Getting here surfaced four client-side findings, all recorded above: a
`VespaSync` per thread doubles p99 and leaves the instance at 70%; httpr
multiplexes one connection per client and ignores `pool_maxsize`; pyvespa asks
for compressed responses by default and the container gzips them; and fewer
connections give the instance ~20% more throughput.

**Sustained load is a different regime.** A 6-minute batch at ~3800 rps
(400k documents per transport) filled the document/v1 queue on this instance:
3% 429s, other errors, throughput halved, container at 96%. The content node
cannot persist as fast as the container accepts for that long. The suite
measures the 3-minute ceiling on purpose (closed loops 30 s + 150 s, batches
~150 s); a sustained-feed lane would be a separate test with its own floor.

## CI run #34 (2026-09-23, us-east runner): the reference result

Session profile derived by the fixture: ceiling ~4058 rps from the warmup,
RTT 56 ms, so 240 in flight per transport (8 connections x 30 streams) for
~250 requests queued inside the instance.

| Lane / method | token rps | mTLS rps | total | vs k6 opening | container CPU | runner CPU | 429 share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| k6 opening | 2041 | 2346 | 4387 | 1.00 | 91% | 29% | 0.5% |
| pyvespa sync_feed_data_point | 2042 | 2369 | 4411 | 1.01 | 95% | 35% | 0.03% |
| pyvespa async_feed_data_point | 2020 | 2355 | 4375 | 1.00 | 93% | 56% | 0.3% |
| pyvespa feed_iterable | 2143 | 2304 | 4447 | 1.01 | 90% | 45% | 0.0% |
| pyvespa feed_async_iterable | 2050 | 2388 | 4438 | 1.01 | 87% | 51% | 0.01% |
| k6 closing | 2047 | 2271 | 4318 | 0.98 | 94% | 30% | 0.5% |

All six green. Every lane and method lands within 1.5% of the opening k6
run, the whole session spans 3% (4318 to 4447), and the measured drift is
-1.6%: two different clients and five code paths give one number for one
saturated instance, and the number they give is the instance. The token
transport is a consistent 0.86 to 0.93 of mTLS (its extra hop), k6 carries
the only 429s worth mentioning (0.5%, under the 1% cap; its per-VU
scheduling burst-queues slightly more than the pyvespa loops), and
`feed_iterable` is no longer an outlier once its queue is bounded.

## First CI run (#32, 2026-09-23, us-east runner, fixed 400 per transport)

| Lane / method | token rps | mTLS rps | total | container CPU | runner CPU | 429 share |
| --- | --- | --- | --- | --- | --- | --- |
| k6 opening | 1524 | 2549 | 4073 | 95% | 27% | 0.8-1.3% |
| pyvespa sync_feed_data_point | 1358 | 2764 | 4122 | 95% | 32% | 1.0-1.3% |
| pyvespa async_feed_data_point | 1823 | 2362 | 4185 | 95% | 46% | 0.7-1.0% |
| pyvespa feed_iterable | 1943 | 2394 | 4337 | 94% | 99% | 0.5-0.6% |
| pyvespa feed_async_iterable | 1893 | 2554 | 4447 | 94% | 48% | 0.2-0.3% |
| k6 closing | 1487 | 2661 | 4149 | 95% | 28% | 0.7-1.2% |

Runner: Xeon Platinum 8370C, 4 vCPU, Python single-thread score 216k ops/s
(a laptop scored 732k). Totals within 9% with the instance saturated on every
test, closing k6 within 2% of the opening one. Three things differed from the
European runs and were fixed after this run: the 429s (concurrency is now
derived per session from ceiling and RTT, see above), the token/mTLS ratio
(0.5-0.6 here against 0.9 from Europe, because the token path's extra hop
adds ~70 ms per request that the short US network no longer hides; the ratio
thresholds are sanity bounds now), and `feed_iterable` at 99% runner CPU
(its consumer rescans up to 2 x `max_queue_size` futures every 10 ms; the
lane now passes `max_queue_size` equal to the worker count, and this is a
pyvespa improvement to make).

## Layout

| File | Purpose |
| --- | --- |
| `conftest.py` | Session fixture: resolves the endpoints and `Vespa` app objects, deletes leftovers, runs the 60 s k6 warmup; idle wait before each test (opt-in cleanup); `perf_last` ordering; `run_state`. |
| `utils/workloads.py` | Load profile (shared concurrency, processes, windows), document factory, per-method thresholds. |
| `utils/metrics.py` | `LaneResult` schema, records writer, shared `assert_token_vs_mtls`. |
| `utils/loadgen.py` | Multi-process pyvespa load generation: per-method workers, sample merging, CPU per request. |
| `utils/k6_lane.py` | Runs the k6 script and parses its summary into `LaneResult`s, with runner and instance CPU. |
| `utils/saturation.py` | Validity probes: runner CPU from `/proc/stat`, instance `cpu_util` per cluster from `/prometheus/v1/values`. |
| `k6/token_vs_mtls.js` | k6 feed workload with per-endpoint latency/throughput/error metrics, hold-window gated. |
| `test_k6_sweep.py` | Concurrency sweep (only with `PERFORMANCE_K6_SWEEP` set): k6 at each level, table to `k6_sweep.md`. |
| `test_k6_token_vs_mtls.py` | Lane 1: k6 opening and closing runs (`perf_last`), shared asserts, instance drift. |
| `test_pyvespa_token_vs_mtls.py` | Lane 2: the four pyvespa methods per transport. |
| `test_deploy_performance_instance.py` | Manual, one-time deploy; `@unittest.skip`'d. |

## Prerequisites

1. **k6** on `PATH` (k6 lane only)
2. **Environment variables**: `VESPA_TEAM_API_KEY`, `VESPA_CLOUD_SECRET_TOKEN`
   (`VESPA_CLIENT_TOKEN_ID` optional, deploy only)
3. **mTLS data-plane certificate** in
   `~/.vespa/vespa-team.pyvespa-performance.default/`

## Running locally

```bash
uv run pytest tests/performance/ -m performance -v
```

Full suite is roughly 20 minutes and feeds (then deletes) documents on the
shared prod instance — avoid concurrent runs. Local runs from Europe see a
higher RTT than CI, so absolute local numbers are not comparable with CI.

## Running in CI

`.github/workflows/performance-cloud.yml` runs the suite weekly, on pushes to
master, on manual dispatch, and on PRs that touch the workflow file itself.
A manual dispatch with the `k6_sweep` input runs only the concurrency sweep.
Runner image and k6 version are pinned. Required secrets: `VESPA_TEAM_API_KEY`,
`VESPA_CLOUD_SECRET_TOKEN`, `VESPA_PERFORMANCE_MTLS_CERT` /
`VESPA_PERFORMANCE_MTLS_KEY`.

Each run uploads a `performance-report` artifact: JUnit XML, raw k6 summary,
per-method `*records.json`, `runner.json`, and `metrics.prom` (generated by
`.github/scripts/reports_to_prom.py`). Every (lane, method, transport)
combination is one labeled Prometheus series, e.g.
`perf_rps{lane="pyvespa",method="feed_iterable",transport="token"}`, so a
future Grafana dashboard gets one plot per combination.

`runner.json` (from `.github/scripts/runner_info.py`, also printed in the job
summary) records the CPU model, vCPU count and a short single-thread Python
score (`perf_runner_python_cpu_score_ops_per_s`). GitHub-hosted runners land on
different CPU models between runs, and the pyvespa lane is client-CPU-bound
(Python + GIL), so its throughput moved up to ~1.8x between runs in Aug-Sep
2026 while k6 stayed within cloud noise. Read pyvespa numbers against the
runner score, or against the k6 ceiling from the same run, not as absolutes.

Note that JUnit attributes the session fixture's setup to the first test and
its teardown (document cleanup) to the last test, so those two test times
overstate the measurements.

## One-time instance lifecycle

The application is deployed **once** and reused. Redeploy only when the
application package in `test_deploy_performance_instance.py` changes:

```bash
# Remove the @unittest.skip first, then:
uv run pytest tests/performance/test_deploy_performance_instance.py -s -v
```

## Follow-ups / known comparability gaps

- **No HTTP-version axis**: httpr cannot force HTTP/1.1 nor report the
  negotiated protocol; the `http` label records each path's library default.
- **Concurrency comes from the sweep**: the 2026-09-22 sweep (100/200/400/800
  VUs per transport) gave 1121 / 2284 / 3517 / 3383 total rps at 42 / 70 / 94 /
  95 % container CPU with no 429s, so `LoadProfile.concurrency` is 400 and
  `VALIDITY.min_server_container_cpu_util` is 0.85. Re-run the sweep and reset
  both if the instance is resized.
- **Thresholds are floors ~30% under the first server-bound run**
  (`utils/workloads.py`). Tighten to a band under the rolling median once a
  few CI runs exist. A hard gate still wants a fixed load generator
  (self-hosted runner on a pinned instance type in `us-east-1`).
- **Headroom on the 4-vCPU runner is ~2x**, not more: 4 processes give
  roughly 6000-7000 rps of client capacity against a ~3500 rps instance
  ceiling. If the instance grows, the client needs more processes or a
  bigger box.
- **Batch APIs lack per-request latency**; a pyvespa enhancement stamping
  request duration into `VespaResponse` would close this.
- **Storage type / instance generation are not pinned** by the deployed
  package; hardware moves can cause step changes in trends.
