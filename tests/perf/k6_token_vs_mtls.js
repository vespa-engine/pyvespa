import http from "k6/http";
import { check, sleep } from "k6";
import { Trend, Rate, Counter } from "k6/metrics";

const maxVus = 200;
const duration = "3m";
const startRate = 100;
const rampStep = 200;
const holdTime = "30s";

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

// Build stages: ramp up by RAMP_STEP, then hold, repeat until duration
function buildStages(totalDuration, holdDuration, step, start) {
  const stages = [];
  const holdMs = parseDuration(holdDuration);
  const totalMs = parseDuration(totalDuration);
  const rampTime = "20s"; // Quick ramp between steps

  let elapsed = 0;
  let currentRate = start;

  while (elapsed < totalMs) {
    // Ramp to next level
    currentRate += step;
    stages.push({ target: currentRate, duration: rampTime });
    elapsed += parseDuration(rampTime);

    if (elapsed >= totalMs) break;

    // Hold at this level
    const remainingTime = totalMs - elapsed;
    const actualHold = Math.min(holdMs, remainingTime);
    stages.push({ target: currentRate, duration: `${actualHold}ms` });
    elapsed += actualHold;
  }

  return stages;
}

function parseDuration(d) {
  if (typeof d === "number") return d;
  const match = d.match(/^(\d+)(ms|s|m|h)$/);
  if (!match) return 60000;
  const value = Number(match[1]);
  const unit = match[2];
  const multipliers = { ms: 1, s: 1000, m: 60000, h: 3600000 };
  return value * multipliers[unit];
}

const stages = buildStages(duration, holdTime, rampStep, startRate);

export const options = {
  scenarios: {
    mtls: {
      executor: "ramping-arrival-rate",
      startRate,
      timeUnit: "1s",
      preAllocatedVUs: maxVus,
      maxVUs: maxVus,
      stages,
      startTime: "0s",
      exec: "mtlsScenario",
    },
    token: {
      executor: "ramping-arrival-rate",
      startRate,
      timeUnit: "1s",
      preAllocatedVUs: maxVus,
      maxVUs: maxVus,
      stages,
      startTime: "0s",
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
