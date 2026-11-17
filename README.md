# Edge QC Classifier Inference Service

A FastAPI + NumPy inference service for an edge quality-control machine with a
hard constraint: **the machine can host at most two model processes**. Models
are replaced online — without dropping accepted inference and without ever
running a third model process — and the active model **heals itself after a
process crash**: the saved weights are reloaded in a fresh process under the
same version id, no re-upload needed.

```
POST /infer                 classify a feature vector (with optional client timeout)
POST /admin/models          upload weights/bias/labels and hot-swap the model in
GET  /admin/versions        version registry
GET  /admin/status          active version, in-flight counts, per-process state
POST /admin/demo/kill-active  demo: SIGKILL the active model to watch recovery
GET  /healthz               liveness
GET  /docs                  Swagger UI (interactive; shows every field)
```

## Running it

```bash
docker compose up --build
# API + Swagger UI:  http://localhost:8000/docs
# quick demo:        python examples/demo.py http://localhost:8000
```

Or without Docker:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
DATA_DIR=./data uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Tests (they start real API and model processes):

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Model artifact

Each version is a linear classifier. 120 draws a class per row: `score =
weights @ features + bias`, and the highest-scoring label wins.

```json
{
  "version": "v2",
  "weights": [[1.2, -0.4, 0.0, 0.3],
              [-0.2, 0.9, 0.5, -0.1],
              [0.1, -0.8, 1.1, 0.2]],
  "bias": [-0.10, 0.05, -0.20],
  "labels": ["pass", "scratch", "dent"]
}
```

`weights` must be `n_classes x n_features`, `bias` must have `n_classes`
entries, and `labels` must be `n_classes` unique non-empty strings. Anything
else is rejected with 422 before a process is started.

`POST /admin/models` also accepts these optional fields:

| field | purpose |
| --- | --- |
| `test_features` | vector for the pre-activation smoke test (defaults to zeros) |
| `expected_active_version` | optimistic guard; 409 if the active version changed |
| `slow_load_ms` | demo: delay worker readiness to watch the old version serve |
| `simulate_load_failure` | demo: make the new worker report a load failure |

## The guarantees and how they are enforced

Each version runs in **its own OS process** (`python -m app.worker_launcher`,
its own process group, private loopback HTTP port). The API process is the only
owner of lifecycle decisions and keeps the registry in a
`multiprocessing.Manager`, so HTTP concurrency cannot race the accounting.
Two fixed slots, `slot-0` and `slot-1`, are the two-process budget.

```
(vacant) --reserve--> loading --commit--> active --retire--> draining --exit--> idle
                           └── load/test failed ──────────────> failed (slot freed)

active --crash confirmed--> recovering --recovery load+test ok--> active (same version)
                                 ├── recovery load/test failed --> failed (no retry)
                                 └── manual upload won ---------> retired (superseded)
```

* **At most two model processes.** A slot is reserved only when it is `idle`.
  It becomes `idle` only after the worker's OS process has exited *and* the
  slot's in-flight counter is zero. A `loading` candidate and the `active`
  version both occupy slots, so a third load gets 429 while a swap is in
  progress. The API refuses to start twice against the same data directory
  (file lock), so there is only ever one budget owner.
* **The old version serves during load.** Promotion is one atomic registry
  operation (`commit_swap`) performed only after the candidate answers a
  smoke-test inference. Until then traffic goes to the old process. A failed
  load or smoke test stops the candidate, frees its slot, and returns 502 with
  `fallback_version` = the still-active old version.
* **Accepted inference is pinned to its version.** `accept_work` reads the
  active version *and* increments that slot's in-flight counter in one lock.
  The returned version is used for the whole request. A swap commits
  afterwards, the request still executes on the process it was accepted for.
* **Timeouts do not end the work.** `timeout_ms` (or its seconds alias
  `timeout_seconds`, or the API's default) bounds only how long the caller
  waits. On expiry the client gets 504, but the worker call runs in a task
  shielded from the request timeout/cancellation, and the in-flight counter is
  decremented when the worker actually returns. A disconnected client behaves
  the same way.
* **The old process exits only after real work ends.** The drain loop stops and
  reaps a `draining` process only when its in-flight count is zero. Just after
  a 504 you will see the old version `draining` with `in_flight: 1`; the slot
  is not reusable yet.
* **A full machine refuses the third version, without touching the current
  one.** While both slots are occupied, uploads get 429 (after a short bounded
  wait for a drain that is already finishing). The active version is never
  stopped to make room.
* **Concurrent replacement of the same old version: at most one wins.** A swap
  claims its old version in the registry before starting a process. A second
  swap from the same old version gets 409 immediately and never spawns a
  process. A stale `expected_active_version` also yields 409.
* **A crashed active model recovers itself, once.** If the active worker
  process dies, the API confirms the OS process exited, takes the version out
  of service (`recovering`; `/infer` returns 503), and starts exactly one
  recovery for that failure: the version's saved artifact
  (`models/<version>.json`) is loaded in a new process, smoke-tested, and
  promoted under the **same version id**. A failed recovery is final — no
  retry loop — and with no usable model the service keeps returning 503 until
  a manual upload. A later, *new* crash of the recovered process is a new
  failure and gets its own single recovery.
* **Unfinished requests fail; they are never replayed.** Requests in flight
  when the process dies get 502 and are settled as `failed` on the old slot.
  `accept_work` tags each request with the process `generation`; a late
  completion callback from the dead process carries a stale generation and is
  ignored, so the recovered process's counters stay exact.
* **A manual upload outranks a recovery.** While a recovery is loading, a
  manual upload may load in parallel in the free slot and takes over: the
  late recovery commits only if nothing else became active, otherwise it
  stops its own candidate (`superseded`) and never touches the new version.
  If the recovery commits first, the manual swap simply retires it.
* **Draining versions do not self-heal.** Only the *active* version is
  recovered; a retired process that dies is just reaped and its slot freed.
  Loading + active + draining processes together still never exceed two, and
  a slot is reused only after the previous process's exit is confirmed.
* **Status distinguishes the old and new process of a version.** Each version
  record carries a `generation` (incremented per process) and a `recovery`
  object: attempt counter, result (`in_progress` / `succeeded` / `failed` /
  `superseded`), and the crashed vs recovered pid/generation. Slots report
  `failed` (requests interrupted by a crash) alongside `accepted` /
  `completed`.

## Demo scenarios

1. **Normal swap.** Upload `examples/models/v1.json`, infer, upload `v2`,
   infer. `/admin/status` shows the previous version `draining` then `idle`.
2. **Load failure.** Upload with `"simulate_load_failure": true` (or send an
   inconsistent artifact). You get 502/422; the old version keeps answering,
   and the candidate's slot is immediately free again.
3. **Slow inference + timeout + capacity.** `POST /infer` with
   `{"features": [...], "slow_ms": 6000, "timeout_ms": 150}`. You get 504.
   `/admin/status` still shows `in_flight: 1`. Upload a new version — it
   becomes active and the slow version goes `draining`. A third upload returns
   429 while the slow computation still runs. After it finishes the slot is
   released and the upload succeeds.
4. **Crash recovery.** `POST /admin/demo/kill-active` (optionally with
   `{"recovery_delay_ms": 3000}`) SIGKILLs the active model process. The
   version goes `recovering`, `/infer` returns 503, and the saved weights are
   loaded in a new process that takes over under the same version id — watch
   the old vs new pid/generation in `/admin/status`. Upload a new version
   during the delay and it wins; the late recovery steps aside (`superseded`).

`python examples/demo.py <base-url>` runs all of them end to end.

## Observations from the smoke runs

- Single request latency on loopback: ~5–7 ms end to end (worker computation
  itself is microseconds; this includes HTTP round trips).
- After a 504, the drain gate held the slot for the full remaining computation
  (verified: 429 until the slow request finished, then `idle`).
- SIGTERM with an in-flight request: the request completed with 200, the API
  exited 0/-15 cleanly, and no worker processes remained.
- SIGKILL on the active worker: the version went `recovering` within ~100 ms,
  `/infer` returned 503, and the same version id was back active on a new pid
  ~1 s later (plus the demo delay when configured). A slow request in flight
  at the kill got 502 and the recovered process's counters showed only its
  own work.

## Layout

```
app/
  main.py             API, swap orchestration, drain loop, status
  process_manager.py  shared registry/state machine + worker process handles
  worker.py           per-versionHTTP server (one process per version)
  worker_launcher.py  process entry point: load artifact, bind, serve, handshake
  models.py           artifact validation, NumPy scoring
tests/
  test_models.py          artifact validation + scoring
  test_registry.py        state machine, claims, drain gate, crash recovery
  test_api_integration.py real API + real worker processes, crash-recovery runs
examples/
  demo.py, models/*.json
```

## Configuration

| env var | default | meaning |
| --- | --- | --- |
| `DATA_DIR` | `/data` | models, staging, worker control dirs, instance lock |
| `DEFAULT_INFER_TIMEOUT_MS` | `5000` | client wait when `timeout_ms` is omitted |
| `SWAP_WAIT_SECONDS` | `3` | how long an upload waits for a draining slot before 429 |
| `SHUTDOWN_GRACE_SECONDS` | `20` | how long shutdown waits for in-flight work |
| `RECOVERY_DELAY_MS` | `0` | demo: delay every automatic recovery's readiness |
| `BOOTSTRAP_VERSION` / `BOOTSTRAP_MODEL` | unset | optionally load one artifact at startup |
| `LOG_LEVEL` | `INFO` | logging |
