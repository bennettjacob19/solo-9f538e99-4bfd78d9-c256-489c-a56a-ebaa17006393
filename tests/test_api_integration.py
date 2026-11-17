"""Integration tests against a real API process and real worker processes.

These exercise the guarantees that only show up with real OS processes:
version pinning, serving during candidate load, load-failure fallback, the
two-process cap, and the fact that a client timeout does *not* free the slot
until the worker has actually finished.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ApiServer:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.port = _free_port()
        self.proc: subprocess.Popen | None = None
        self.log = open(data_dir / "api.log", "w", encoding="utf-8")

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "ApiServer":
        env = {**os.environ, "DATA_DIR": str(self.data_dir), "LOG_LEVEL": "WARNING"}
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--log-level",
                "warning",
            ],
            cwd=str(ROOT),
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.time() + 30
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("API exited during startup:\n" + self._tail())
            try:
                if httpx.get(f"{self.base}/healthz", timeout=1.0).status_code == 200:
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
        raise TimeoutError("API did not start:\n" + self._tail())

    def __exit__(self, *exc) -> None:
        if self.proc is not None and self.proc.poll() is None:
            os.killpg(os.getpgid(self.proc.pid), 15)
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), 9)
                self.proc.wait(10)
        # Any model workers belong to their own process groups; the API stops
        # them on shutdown, but reap stragglers that could poison later tests.
        subprocess.run(
            ["pkill", "-f", f"{self.data_dir}/workers"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.log.close()

    def _tail(self) -> str:
        self.log.flush()
        return self.log.name


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("qcdata")
    with ApiServer(data_dir) as server:
        yield server
    shutil.rmtree(data_dir, ignore_errors=True)


def artifact(version: str, weights, bias, labels, **extra):
    return {"version": version, "weights": weights, "bias": bias, "labels": labels, **extra}


W_A = [[1.0, 0.0], [0.0, 1.0]]
W_B = [[0.0, 1.0], [1.0, 0.0]]
LABELS = ["defect-A", "defect-B"]


def upload(api: ApiServer, payload, timeout=40.0):
    return httpx.post(f"{api.base}/admin/models", json=payload, timeout=timeout)


def infer(api: ApiServer, features, timeout_ms=None, slow_ms=0, timeout=40.0):
    body = {"features": features, "slow_ms": slow_ms}
    if timeout_ms is not None:
        body["timeout_ms"] = timeout_ms
    return httpx.post(f"{api.base}/infer", json=body, timeout=timeout)


def status(api: ApiServer):
    return httpx.get(f"{api.base}/admin/status", timeout=5.0).json()


def wait_for(predicate, timeout=10.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError("condition not met in time")


def slot_states(api: ApiServer):
    return [(s["slot_id"], s["version"], s["process_state"]) for s in status(api)["slots"]]


def test_01_initial_load_and_inference(api):
    resp = upload(api, artifact("v1", W_A, [0.0, 0.0], LABELS))
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["previous_version"] is None
    assert body["state"] == "active"

    result = infer(api, [0.9, 0.1])
    assert result.status_code == 200, result.text
    payload = result.json()
    assert payload["version"] == "v1"
    assert payload["label"] == "defect-A"
    assert payload["score"] == pytest.approx(0.9)

    st = status(api)
    assert st["active_version"] == "v1"
    assert st["slots_in_use"] == 1
    assert st["total_in_flight"] == 0
    assert st["total_completed"] >= 1


def test_02_feature_width_mismatch(api):
    resp = infer(api, [1.0, 2.0, 3.0])
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "feature_dimension_mismatch"
    assert status(api)["total_in_flight"] == 0


def test_03_load_failure_keeps_old_version(api):
    before = status(api)["slots_in_use"]
    resp = upload(
        api,
        artifact("bad-load", W_B, [0.0, 0.0], LABELS, simulate_load_failure=True),
    )
    assert resp.status_code == 502, resp.text
    detail = resp.json()["detail"]
    assert detail["error"] == "candidate_load_failed"
    assert detail["fallback_version"] == "v1"

    st = status(api)
    assert st["active_version"] == "v1"
    assert st["slots_in_use"] == before  # candidate slot released
    failed = next(v for v in st["versions"] if v["version"] == "bad-load")
    assert failed["state"] == "failed"
    assert "simulated" in failed["error"]
    # Old version still answers with the old weights.
    assert infer(api, [0.9, 0.1]).json()["label"] == "defect-A"


def test_04_invalid_artifact_rejected(api):
    resp = upload(api, artifact("broken", [[1.0, 2.0]], [0.0, 0.0], ["a"]))
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_artifact"


def test_05_old_version_serves_during_candidate_load(api):
    observations = []
    resp = {}

    def do_upload():
        resp["r"] = upload(
            api, artifact("v2", W_B, [0.0, 0.0], LABELS, slow_load_ms=1200)
        )

    import threading

    t = threading.Thread(target=do_upload)
    t.start()
    deadline = time.time() + 1.0
    while time.time() < deadline:
        r = infer(api, [0.9, 0.1])
        if r.status_code == 200:
            observations.append(r.json()["version"])
        time.sleep(0.1)
    t.join()
    assert resp["r"].status_code == 201, resp["r"].text
    assert observations, "old version did not answer during candidate load"
    assert set(observations) <= {"v1", "v2"}
    assert "v1" in observations

    st = status(api)
    assert st["active_version"] == "v2"
    v1 = next(v for v in st["versions"] if v["version"] == "v1")
    assert v1["state"] == "retired"


def test_06_timeout_does_not_release_slot(api):
    # Let the drain of v1 settle first so both slots are known.
    wait_for(lambda: status(api)["slots_in_use"] == 1)

    # Long enough that the upload's bounded drain-wait (3s) expires.
    resp = infer(api, [0.9, 0.1], timeout_ms=150, slow_ms=6000)
    assert resp.status_code == 504, resp.text
    detail = resp.json()["detail"]
    assert detail["error"] == "inference_timeout"
    assert detail["version"] == "v2"

    # Immediately after the client gave up the work is still running.
    st = status(api)
    assert st["total_in_flight"] == 1
    assert st["active_version"] == "v2"

    # Swap commits while the old request is still computing.
    resp2 = upload(api, artifact("v3", W_A, [0.0, 0.0], LABELS))
    assert resp2.status_code == 201, resp2.text
    st = status(api)
    assert st["active_version"] == "v3"
    states = dict((s["version"], s["process_state"]) for s in st["slots"] if s["version"])
    assert states.get("v2") == "draining"
    assert st["slots_in_use"] == 2

    # Third load must be refused while the drain is still held by real work.
    resp3 = upload(api, artifact("v4", W_B, [0.0, 0.0], LABELS))
    assert resp3.status_code == 429, resp3.text
    assert resp3.json()["detail"]["error"] == "capacity_exhausted"
    assert status(api)["active_version"] == "v3"  # never stopped the current version

    # New traffic already goes to the new version.
    assert infer(api, [0.9, 0.1]).json()["version"] == "v3"

    # Once the slow work actually finishes, the drained slot is released.
    wait_for(lambda: status(api)["slots_in_use"] == 1, timeout=15)
    st = status(api)
    v2 = next(v for v in st["versions"] if v["version"] == "v2")
    assert v2["state"] == "retired"
    assert v2["process_state"] == "idle"
    assert st["total_in_flight"] == 0


def test_07_third_version_loads_after_release(api):
    resp = upload(api, artifact("v4", W_B, [0.0, 0.0], LABELS))
    assert resp.status_code == 201, resp.text
    st = status(api)
    assert st["active_version"] == "v4"
    in_use = [s for s in st["slots"] if s["process_state"] != "idle"]
    assert len(in_use) == 2  # v4 active + v3 draining
    assert infer(api, [0.9, 0.1]).json()["version"] == "v4"


def test_08_concurrent_swaps_one_winner(api):
    wait_for(lambda: status(api)["slots_in_use"] == 1, timeout=10)
    import threading

    results = []

    def go(version, weights, slow):
        results.append(
            (
                version,
                upload(
                    api,
                    artifact(version, weights, [0.0, 0.0], LABELS, slow_load_ms=slow),
                ).status_code,
            )
        )

    t1 = threading.Thread(target=go, args=("c1", W_A, 1200))
    t2 = threading.Thread(target=go, args=("c2", W_B, 0))
    t1.start()
    time.sleep(0.25)
    t2.start()
    t1.join()
    t2.join()
    winners = [v for v, code in results if code == 201]
    losers = [v for v, code in results if code == 409]
    assert len(winners) == 1, results
    assert len(losers) == 1, results
    assert status(api)["active_version"] == winners[0]


def test_08b_duplicate_id_concurrent_uploads(api):
    """Two simultaneous uploads of the same id: exactly one wins, one 409."""
    wait_for(lambda: status(api)["slots_in_use"] == 1, timeout=10)
    import threading

    codes = []

    def go():
        codes.append(upload(api, artifact("dup", W_A, [0.0, 0.0], LABELS)).status_code)

    threads = [threading.Thread(target=go) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(codes) == [201, 409], codes
    assert status(api)["active_version"] == "dup"
    # The loser must not leave a stale swap claim behind: another swap works.
    resp = upload(api, artifact("dup-next", W_B, [0.0, 0.0], LABELS))
    assert resp.status_code == 201, resp.text


def test_09_status_reports_processes_and_counts(api):
    st = status(api)
    assert st["capacity"] == 2
    assert 1 <= st["slots_in_use"] <= 2
    for slot in st["slots"]:
        assert slot["process_state"] in {"loading", "active", "draining", "idle"}
        if slot["version"] is not None:
            assert slot["alive"] is True
            assert slot["pid"] is not None
    active_slots = [s for s in st["slots"] if s["process_state"] == "active"]
    assert len(active_slots) == 1
    assert active_slots[0]["version"] == st["active_version"]


def test_10_duplicate_version_is_conflict(api):
    active = status(api)["active_version"]
    resp = upload(api, artifact(active, W_A, [0.0, 0.0], LABELS))
    assert resp.status_code == 409
    assert resp.json()["detail"]["error"] == "version_exists"


def test_11_stale_expected_version(api):
    resp = upload(
        api,
        artifact(
            f"stale-{uuid.uuid4().hex[:6]}",
            W_A,
            [0.0, 0.0],
            LABELS,
            expected_active_version="does-not-exist",
        ),
    )
    assert resp.status_code == 409
    assert resp.json()["detail"]["error"] == "swap_conflict"


def test_12_no_model_after_shutdown(api):
    # Sanity: the service still answers health and exposes /docs.
    assert httpx.get(f"{api.base}/healthz", timeout=5).status_code == 200
    assert httpx.get(f"{api.base}/docs", timeout=5).status_code == 200
    assert httpx.get(f"{api.base}/openapi.json", timeout=5).status_code == 200


# ---------------------------------------------------------------------------
# Crash recovery of the active model process
# ---------------------------------------------------------------------------


def kill_active(api: ApiServer, recovery_delay_ms=0):
    return httpx.post(
        f"{api.base}/admin/demo/kill-active",
        json={"recovery_delay_ms": recovery_delay_ms},
        timeout=15.0,
    )


def version_view(api: ApiServer, version: str):
    return next(v for v in status(api)["versions"] if v["version"] == version)


def test_13_active_crash_recovers_same_version(api):
    active = status(api)["active_version"]
    assert active is not None
    before = version_view(api, active)
    assert before["generation"] == 1
    crashed_pid = before["pid"]
    expected_label = infer(api, [0.9, 0.1]).json()["label"]

    resp = kill_active(api, recovery_delay_ms=2500)
    assert resp.status_code == 200, resp.text
    assert resp.json()["killed"]["version"] == active
    assert resp.json()["killed"]["pid"] == crashed_pid

    # The version leaves service; inference gets 503 while it recovers.
    wait_for(lambda: status(api)["active_version"] is None)
    r = infer(api, [0.9, 0.1])
    assert r.status_code == 503
    assert r.json()["detail"]["error"] == "no_active_model"
    # Status stays queryable during the recovery and shows it in progress.
    st = status(api)
    view = next(v for v in st["versions"] if v["version"] == active)
    assert view["state"] == "recovering"
    assert view["recovery"]["state"] == "in_progress"
    assert view["recovery"]["attempt"] == 1
    assert view["recovery"]["crashed_pid"] == crashed_pid

    # The recovery commits under the same version id, on a new process.
    wait_for(lambda: status(api)["active_version"] == active, timeout=30)
    st = status(api)
    view = next(v for v in st["versions"] if v["version"] == active)
    assert view["state"] == "active"
    assert view["generation"] == 2
    assert view["pid"] != crashed_pid
    rec = view["recovery"]
    assert rec["state"] == "succeeded"
    assert rec["crashed_generation"] == 1
    assert rec["recovered_generation"] == 2
    assert rec["recovered_pid"] == view["pid"]

    result = infer(api, [0.9, 0.1])
    assert result.status_code == 200
    assert result.json()["version"] == active
    assert result.json()["label"] == expected_label


def test_14_in_flight_request_fails_at_crash_and_is_not_replayed(api):
    active = status(api)["active_version"]
    crashed_pid = version_view(api, active)["pid"]
    import threading

    responses = []

    def slow_call():
        responses.append(infer(api, [0.9, 0.1], slow_ms=5000, timeout_ms=30000))

    t = threading.Thread(target=slow_call)
    t.start()
    wait_for(lambda: status(api)["total_in_flight"] == 1)

    assert kill_active(api).status_code == 200
    t.join(timeout=20)
    # The unfinished request fails; it is never replayed on the new process.
    assert len(responses) == 1
    assert responses[0].status_code == 502, responses[0].text
    assert responses[0].json()["detail"]["error"] == "worker_unreachable"

    wait_for(lambda: status(api)["active_version"] == active, timeout=30)
    view = version_view(api, active)
    assert view["recovery"]["state"] == "succeeded"
    assert view["recovery"]["crashed_pid"] == crashed_pid

    # Exactly one new inference: the recovered process's counters reflect
    # only its own work (no pollution from the crashed process's callback).
    assert infer(api, [0.9, 0.1]).status_code == 200
    st = status(api)
    slot = next(s for s in st["slots"] if s["version"] == active)
    assert slot["accepted"] == 1
    assert slot["completed"] == 1
    assert slot["in_flight"] == 0
    assert st["total_in_flight"] == 0


def test_15_failed_recovery_stops_and_serves_503(api):
    active = status(api)["active_version"]
    # Corrupt the saved artifact so the recovery load fails.
    artifact_path = api.data_dir / "models" / f"{active}.json"
    assert artifact_path.is_file()
    artifact_path.write_text("{ not json", encoding="utf-8")

    assert kill_active(api).status_code == 200
    wait_for(lambda: version_view(api, active)["state"] == "failed", timeout=30)
    view = version_view(api, active)
    assert view["recovery"]["state"] == "failed"
    assert "recovery failed" in view["error"]
    assert status(api)["active_version"] is None
    assert infer(api, [0.9, 0.1]).status_code == 503

    # No retry loop: the failure is final and no process is spawned again.
    time.sleep(2.0)
    st = status(api)
    assert version_view(api, active)["state"] == "failed"
    assert st["slots_in_use"] == 0
    # The demo kill now has nothing to kill.
    assert kill_active(api).status_code == 409

    # Manual re-upload of the same (failed) version id restores service.
    resp = upload(api, artifact(active, W_A, [0.0, 0.0], LABELS))
    assert resp.status_code == 201, resp.text
    assert status(api)["active_version"] == active
    assert infer(api, [0.9, 0.1]).json()["label"] == "defect-A"


def test_16_manual_upload_takes_priority_over_delayed_recovery(api):
    active = status(api)["active_version"]
    assert kill_active(api, recovery_delay_ms=4000).status_code == 200
    wait_for(lambda: version_view(api, active)["state"] == "recovering")

    # A manual upload loads in parallel in the free slot and takes over.
    resp = upload(api, artifact("manual-takeover", W_B, [0.0, 0.0], LABELS))
    assert resp.status_code == 201, resp.text
    assert status(api)["active_version"] == "manual-takeover"
    takeover_pid = version_view(api, "manual-takeover")["pid"]

    # The late recovery neither takes over nor stops the new version.
    wait_for(
        lambda: version_view(api, active)["recovery"]["state"] == "superseded",
        timeout=30,
    )
    view = version_view(api, active)
    assert view["state"] == "retired"
    st = status(api)
    assert st["active_version"] == "manual-takeover"
    assert version_view(api, "manual-takeover")["pid"] == takeover_pid
    wait_for(lambda: status(api)["slots_in_use"] == 1, timeout=15)
    assert infer(api, [0.9, 0.1]).json()["version"] == "manual-takeover"


def test_17_draining_version_does_not_self_heal(api):
    import signal

    assert status(api)["active_version"] == "manual-takeover"
    # Park a slow request on the active version, then swap it out so it drains.
    r = infer(api, [0.9, 0.1], slow_ms=3000, timeout_ms=150)
    assert r.status_code == 504
    resp = upload(api, artifact("post-takeover", W_A, [0.0, 0.0], LABELS))
    assert resp.status_code == 201, resp.text
    st = status(api)
    assert st["active_version"] == "post-takeover"
    draining = next(s for s in st["slots"] if s["version"] == "manual-takeover")
    assert draining["process_state"] == "draining"
    assert draining["in_flight"] == 1

    # Kill the draining process: no recovery is started for it.
    os.kill(draining["pid"], signal.SIGKILL)
    wait_for(
        lambda: version_view(api, "manual-takeover")["process_state"] == "idle",
        timeout=15,
    )
    view = version_view(api, "manual-takeover")
    assert view["state"] == "retired"
    assert view["recovery"] is None
    st = status(api)
    assert st["active_version"] == "post-takeover"
    assert st["slots_in_use"] == 1
    assert infer(api, [0.9, 0.1]).json()["version"] == "post-takeover"
