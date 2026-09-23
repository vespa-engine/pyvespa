import http from "k6/http";
import exec from "k6/execution";
import { check } from "k6";
import { Trend, Rate, Counter } from "k6/metrics";

// Load profile is injected by test_k6_token_vs_mtls.py from the shared
// LoadProfile (utils/workloads.py) so both lanes generate identical load.
const maxVus = Number(__ENV.MAX_VUS || 200); // in-flight requests per transport
const rampUp = __ENV.RAMP_UP || "30s"; // warm up to maxVus
const hold = __ENV.HOLD || "2m30s"; // steady-state measurement window
// Connection topology. 1: one connection per in-flight request (one VU each,
// ramping-vus). N > 1: maxVus / N connections (VUs), each keeping N requests
// in flight concurrently over its single HTTP/2 connection via
// http.asyncRequest. The pyvespa lane runs `processes` clients per transport
// with concurrency / processes requests multiplexed over one connection each,
// so N = concurrency / processes gives the instance identical traffic.
const streamsPerConnection = Number(__ENV.STREAMS_PER_CONNECTION || 1);
const connections = Math.max(1, Math.floor(maxVus / streamsPerConnection));

// k6 duration string ("30s", "2m30s") to milliseconds.
function toMs(duration) {
  let ms = 0;
  for (const [, value, unit] of duration.matchAll(/(\d+)(ms|s|m|h)/g)) {
    ms += Number(value) * { ms: 1, s: 1000, m: 60000, h: 3600000 }[unit];
  }
  return ms;
}

// Measurement window, identical to the pyvespa lane: only requests that
// complete inside [rampUp, rampUp + hold] feed the custom metrics, so rps is
// count / hold and latency excludes ramp-up and graceful-stop tails.
const measureStartMs = toMs(rampUp);
const measureEndMs = measureStartMs + toMs(hold);

const schema = "msmarco";

const tokenUrl = __ENV.TOKEN_URL;
const mtlsUrl = __ENV.MTLS_URL;
const tokenAuthHeader = __ENV.TOKEN_AUTH_HEADER;

const tlsAuth = [];
if (__ENV.MTLS_CERT_PATH && __ENV.MTLS_KEY_PATH) {
  tlsAuth.push({
    cert: open(__ENV.MTLS_CERT_PATH),
    key: open(__ENV.MTLS_KEY_PATH),
  });
}

// Closed model: a fixed pool of VUs feeds as fast as the instance responds, so
// throughput is the measured *output* (not a target we try to hit). This avoids
// the "insufficient VUs" warnings and dropped iterations the arrival-rate model
// produced, and gives stable, comparable numbers for regression tracking. Both
// scenarios use the same VU schedule for a fair token-vs-mTLS comparison.
const vuStages = [
  { target: maxVus, duration: rampUp },
  { target: maxVus, duration: hold },
];

const perConnectionScenarios = {
  mtls: {
    executor: "ramping-vus",
    startVUs: 0,
    stages: vuStages,
    gracefulStop: "30s",
    exec: "mtlsScenario",
  },
  token: {
    executor: "ramping-vus",
    startVUs: 0,
    stages: vuStages,
    gracefulStop: "30s",
    exec: "tokenScenario",
  },
};

// One long iteration per VU (= connection) that keeps streamsPerConnection
// requests in flight until the measurement window ends.
const multiplexedDuration = `${Math.ceil(measureEndMs / 1000) + 5}s`;
const multiplexedScenarios = {
  mtls: {
    executor: "constant-vus",
    vus: connections,
    duration: multiplexedDuration,
    gracefulStop: "30s",
    exec: "mtlsMultiplexed",
  },
  token: {
    executor: "constant-vus",
    vus: connections,
    duration: multiplexedDuration,
    gracefulStop: "30s",
    exec: "tokenMultiplexed",
  },
};

export const options = {
  scenarios:
    streamsPerConnection > 1 ? multiplexedScenarios : perConnectionScenarios,
  summaryTrendStats: ["min", "avg", "med", "p(95)", "p(99)", "max"],
  tlsAuth,
};

const mtlsDuration = new Trend("mtls_req_duration");
const tokenDuration = new Trend("token_req_duration");
const mtlsFailRate = new Rate("mtls_fail_rate");
const tokenFailRate = new Rate("token_fail_rate");
const mtlsReqs = new Counter("mtls_reqs");
const tokenReqs = new Counter("token_reqs");
// 429 (backpressure) counted separately: the pyvespa lane runs with retries
// off and counts the same, so both lanes report overload identically.
const mtlsRateLimited = new Counter("mtls_rate_limited");
const tokenRateLimited = new Counter("token_rate_limited");

// Same request timeout as pyvespa's httpr client (120 s).
const requestTimeout = "120s";

function feedDoc(url, authHeader, kindTag) {
  if (!kindTag) {
    throw new Error("kindTag is required for tagging http requests");
  }
  const docId = Math.random().toString(36).slice(2);
  const endpoint = `${url.replace(/\/+$/, "")}/document/v1/${schema}/${schema}/docid/${docId}`;

  const payload = JSON.stringify({
    fields: {
      id: docId,
      title: "performance-doc",
      body: "benchmark run",
    },
  });

  const params = {
    timeout: requestTimeout,
    headers: {
      "Content-Type": "application/json",
      ...(authHeader ? { Authorization: authHeader } : {}),
    },
    tags: {
      kind: kindTag,
      name: `feed_doc_${kindTag}`,
    },
  };

  return http.post(endpoint, payload, params);
}

function feedDocAsync(url, authHeader, kindTag) {
  const docId = Math.random().toString(36).slice(2);
  const endpoint = `${url.replace(/\/+$/, "")}/document/v1/${schema}/${schema}/docid/${docId}`;
  const payload = JSON.stringify({
    fields: { id: docId, title: "performance-doc", body: "benchmark run" },
  });
  const params = {
    timeout: requestTimeout,
    headers: {
      "Content-Type": "application/json",
      ...(authHeader ? { Authorization: authHeader } : {}),
    },
    tags: { kind: kindTag, name: `feed_doc_${kindTag}` },
  };
  return http.asyncRequest("POST", endpoint, payload, params);
}

// Record a completed request into the per-transport metrics if it finished
// inside the measurement window. Built-in http_* metrics still cover every
// request.
function record(res, duration, failRate, reqs, rateLimited) {
  const ok = res.status >= 200 && res.status < 300;
  const completedAtMs = exec.instance.currentTestRunDuration;
  if (completedAtMs >= measureStartMs && completedAtMs <= measureEndMs) {
    duration.add(res.timings.duration);
    failRate.add(!ok);
    reqs.add(1);
    if (res.status === 429) {
      rateLimited.add(1);
    }
  }
  return ok;
}

export function mtlsScenario() {
  const mtlsRes = feedDoc(mtlsUrl, null, "mtls");
  const mtlsOk = record(
    mtlsRes,
    mtlsDuration,
    mtlsFailRate,
    mtlsReqs,
    mtlsRateLimited,
  );
  check(mtlsRes, { "mtls status 2xx": () => mtlsOk });
}

export function tokenScenario() {
  const tokenRes = feedDoc(tokenUrl, tokenAuthHeader, "token");
  const tokenOk = record(
    tokenRes,
    tokenDuration,
    tokenFailRate,
    tokenReqs,
    tokenRateLimited,
  );
  check(tokenRes, { "token status 2xx": () => tokenOk });
}

async function multiplexed(url, authHeader, kindTag, duration, failRate, reqs, rateLimited) {
  async function stream() {
    while (exec.instance.currentTestRunDuration < measureEndMs) {
      const res = await feedDocAsync(url, authHeader, kindTag);
      record(res, duration, failRate, reqs, rateLimited);
    }
  }
  await Promise.all(Array.from({ length: streamsPerConnection }, stream));
}

export async function mtlsMultiplexed() {
  await multiplexed(mtlsUrl, null, "mtls", mtlsDuration, mtlsFailRate, mtlsReqs, mtlsRateLimited);
}

export async function tokenMultiplexed() {
  await multiplexed(tokenUrl, tokenAuthHeader, "token", tokenDuration, tokenFailRate, tokenReqs, tokenRateLimited);
}
