import http from "k6/http";
import { check, sleep } from "k6";
import { Trend, Rate, Counter } from "k6/metrics";

const vus = Number(__ENV.VUS || 50);
const duration = __ENV.DURATION || "120s";
const sleepDuration = Number(__ENV.K6_SLEEP ?? 0);
const tokenUrl = __ENV.TOKEN_URL;
const mtlsUrl = __ENV.MTLS_URL;
const schema = __ENV.SCHEMA || "msmarco";
const tokenAuthHeader = __ENV.TOKEN_AUTH_HEADER;

function hostnameFromUrl(url) {
  if (!url) {
    throw new Error("Missing URL for TLS auth configuration.");
  }
  if (!url) return undefined;
  const match = url.match(/^https?:\/\/([^/]+)/);
  return match ? match[1] : undefined;
}

const tlsAuth = [];
if (__ENV.MTLS_CERT_PATH && __ENV.MTLS_KEY_PATH && mtlsUrl) {
  const domain = __ENV.MTLS_DOMAIN || hostnameFromUrl(mtlsUrl);
  tlsAuth.push({
    domains: [domain],
    cert: open(__ENV.MTLS_CERT_PATH),
    key: open(__ENV.MTLS_KEY_PATH),
  });
}

export const options = {
  scenarios: {
    mtls: {
      executor: "constant-vus",
      vus,
      duration,
      exec: "mtlsScenario",
    },
    token: {
      executor: "constant-vus",
      vus,
      duration,
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
      title: "perf-doc",
      body: "benchmark run",
    },
  });

  const params = {
    headers: {
      "Content-Type": "application/json",
      ...(authHeader ? { Authorization: authHeader } : {}),
    },
    tags: { kind: kindTag },
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
  sleep(sleepDuration);
}

export function tokenScenario() {
  const tokenRes = feedDoc(tokenUrl, tokenAuthHeader, "token");
  const tokenOk = tokenRes.status >= 200 && tokenRes.status < 300;
  tokenDuration.add(tokenRes.timings.duration);
  tokenFailRate.add(!tokenOk);
  tokenReqs.add(1);
  check(tokenRes, { "token status 2xx": () => tokenOk });
  sleep(sleepDuration);
}

export function handleSummary(data) {
  console.log("=== k6 metrics dump ===");
  console.log(JSON.stringify(data.metrics, null, 2));
  return {};
}
