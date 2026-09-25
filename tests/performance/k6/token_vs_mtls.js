import http from "k6/http";
import exec from "k6/execution";
import { Trend, Rate, Counter } from "k6/metrics";

// One transport per run (TRANSPORT=token|mtls), so each gets the whole instance
// and its throughput is an absolute ceiling rather than a share.
const transport = __ENV.TRANSPORT || "mtls";
const url = (transport === "token" ? __ENV.TOKEN_URL : __ENV.MTLS_URL).replace(/\/+$/, "");
const authHeader = transport === "token" ? __ENV.TOKEN_AUTH_HEADER : null;

// Match pyvespa: one connection per worker, with concurrent HTTP/2 streams.
const maxVus = Number(__ENV.MAX_VUS || 400);
const streamsPerConnection = Number(__ENV.STREAMS_PER_CONNECTION || 50);
const connections = Math.max(1, Math.floor(maxVus / streamsPerConnection));

function toMs(duration) {
  let ms = 0;
  for (const [, value, unit] of duration.matchAll(/(\d+)(ms|s|m|h)/g)) {
    ms += Number(value) * { ms: 1, s: 1000, m: 60000, h: 3600000 }[unit];
  }
  return ms;
}

const measureStartMs = toMs(__ENV.RAMP_UP || "30s");
const measureEndMs = measureStartMs + toMs(__ENV.HOLD || "2m30s");
const tlsAuth = [];
if (__ENV.MTLS_CERT_PATH && __ENV.MTLS_KEY_PATH) {
  tlsAuth.push({
    cert: open(__ENV.MTLS_CERT_PATH),
    key: open(__ENV.MTLS_KEY_PATH),
  });
}

export const options = {
  scenarios: {
    feed: {
      executor: "constant-vus",
      vus: connections,
      duration: `${Math.ceil(measureEndMs / 1000) + 5}s`,
      gracefulStop: "30s",
    },
  },
  summaryTrendStats: ["min", "avg", "med", "p(95)", "p(99)", "max"],
  tlsAuth,
};

const measured = {
  duration: new Trend(`${transport}_req_duration`),
  failed: new Rate(`${transport}_fail_rate`),
  requests: new Counter(`${transport}_reqs`),
  limited: new Counter(`${transport}_rate_limited`),
};

async function stream() {
  while (exec.instance.currentTestRunDuration < measureEndMs) {
    const docId = Math.random().toString(36).slice(2);
    const payload = JSON.stringify({
      fields: { id: docId, title: "performance-doc", body: "benchmark run" },
    });
    const res = await http.asyncRequest(
      "POST",
      `${url}/document/v1/msmarco/msmarco/docid/${docId}`,
      payload,
      {
        timeout: "120s",
        headers: {
          "Content-Type": "application/json",
          ...(authHeader ? { Authorization: authHeader } : {}),
        },
        tags: { kind: transport, name: `feed_doc_${transport}` },
      },
    );
    // Count completions only during the hold, excluding warmup and shutdown.
    const completed = exec.instance.currentTestRunDuration;
    if (completed >= measureStartMs && completed <= measureEndMs) {
      measured.duration.add(res.timings.duration);
      measured.failed.add(res.status < 200 || res.status >= 300);
      measured.requests.add(1);
      if (res.status === 429) measured.limited.add(1);
    }
  }
}

export default async function () {
  await Promise.all(Array.from({ length: streamsPerConnection }, stream));
}
