// Ramp 1 -> 50 VUs over 60 s against POST /score with test-split rows.
// Payloads cycle; each iteration gets a unique TransactionID; every request uses ONE constant
// TransactionDT (the API 409s on repeats and earlier DTs, and concurrent VUs arrive out of order).
// CAVEAT: a constant DT means every request sees prior_n=0, so the latency excludes real
// neighbour-count work; the README must say so when quoting it. Override: BASE_URL, DURATION (seconds), MAX_VUS.
import http from 'k6/http';
import { check } from 'k6';
import { SharedArray } from 'k6/data';
import exec from 'k6/execution';

const payloads = new SharedArray('payloads', () => JSON.parse(open('./payloads.json')));
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const DURATION = parseInt(__ENV.DURATION || '60');
const MAX_VUS = parseInt(__ENV.MAX_VUS || '50');
const ID_BASE = 1000000000;
// One shared DT past any dataset DT. VUs run concurrently, so requests can reach the server out
// of order; the API only rejects an EARLIER DT, so a constant is the only value that cannot 409.
const DT = 1000000000;

export const options = {
  stages: [{ duration: `${DURATION}s`, target: MAX_VUS }],
};

export default function () {
  const i = exec.scenario.iterationInTest;
  const body = Object.assign({}, payloads[i % payloads.length], {
    TransactionID: ID_BASE + i,
    TransactionDT: DT,
  });
  const res = http.post(`${BASE_URL}/score`, JSON.stringify(body),
    { headers: { 'Content-Type': 'application/json' } });
  check(res, { 'status 200': (r) => r.status === 200 });
}
