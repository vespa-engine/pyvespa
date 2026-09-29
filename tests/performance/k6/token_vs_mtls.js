// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import http from "k6/http";
import exec from "k6/execution";
import { Trend, Rate, Counter } from "k6/metrics";

const transport = __ENV.TRANSPORT || "mtls";
const url = (transport === "token" ? __ENV.TOKEN_URL : __ENV.MTLS_URL).replace(/\/+$/, "");
const authHeader = transport === "token" ? __ENV.TOKEN_AUTH_HEADER : null;

const maxVus = Number(__ENV.MAX_VUS || 400);
const connections = Number(__ENV.CONNECTIONS || 8);
const streamsPerConnection = Math.max(1, Math.round(maxVus / connections));

// Same document text as the pyvespa lane.
const bodyBytes = Number(__ENV.BODY_BYTES || 0);
const body = bodyBytes
  ? open("../documents/body_4k.txt").slice(0, bodyBytes)
  : open("../documents/body_small.txt");
const compression = __ENV.COMPRESSION || "";

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
      // Ends with the hold; gracefulStop lets requests in flight complete.
      duration: `${Math.ceil(measureEndMs / 1000)}s`,
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
      fields: { id: docId, title: "performance-doc", body },
    });
    const res = await http.asyncRequest(
      "POST",
      `${url}/document/v1/msmarco/msmarco/docid/${docId}`,
      payload,
      {
        timeout: "120s",
        compression,
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
