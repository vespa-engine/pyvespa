import http from "k6/http";
import { check, sleep } from "k6";
import { Trend, Rate, Counter } from "k6/metrics";

const maxVus = 200;
const rampUp = "30s"; // warm up to maxVus
const hold = "2m30s"; // steady-state measurement window

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

export const options = {
  scenarios: {
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
  },
  summaryTrendStats: ["min", "avg", "med", "p(95)", "p(99)", "max"],
  tlsAuth,
};

const mtlsDuration = new Trend("mtls_req_duration");
const tokenDuration = new Trend("token_req_duration");
const mtlsFailRate = new Rate("mtls_fail_rate");
const tokenFailRate = new Rate("token_fail_rate");
const mtlsReqs = new Counter("mtls_reqs");
const tokenReqs = new Counter("token_reqs");

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

export function mtlsScenario() {
  const mtlsRes = feedDoc(mtlsUrl, null, "mtls");
  const mtlsOk = mtlsRes.status >= 200 && mtlsRes.status < 300;
  mtlsDuration.add(mtlsRes.timings.duration);
  mtlsFailRate.add(!mtlsOk);
  mtlsReqs.add(1);
  check(mtlsRes, { "mtls status 2xx": () => mtlsOk });
}

export function tokenScenario() {
  const tokenRes = feedDoc(tokenUrl, tokenAuthHeader, "token");
  const tokenOk = tokenRes.status >= 200 && tokenRes.status < 300;
  tokenDuration.add(tokenRes.timings.duration);
  tokenFailRate.add(!tokenOk);
  tokenReqs.add(1);
  check(tokenRes, { "token status 2xx": () => tokenOk });
}

export function handleSummary(data) {
  console.log("=== k6 metrics dump ===");
  console.log(JSON.stringify(data.metrics, null, 2));
  return {};
}
