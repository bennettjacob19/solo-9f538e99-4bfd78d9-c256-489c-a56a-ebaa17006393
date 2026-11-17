"""Unit tests for the shared registry state machine, including the drain gate."""

from __future__ import annotations

import multiprocessing

import pytest

from app.process_manager import (
    VERSION_ACTIVE,
    VERSION_FAILED,
    VERSION_RETIRED,
    FeatureMismatch,
    NoActiveModel,
    NoCapacity,
    Registry,
    SwapConflict,
)


@pytest.fixture(scope="module")
def manager():
    ctx = multiprocessing.get_context("fork")
    mgr = ctx.Manager()
    yield mgr
    mgr.shutdown()


@pytest.fixture()
def registry(manager):
    reg = Registry(manager)
    yield reg
    # Clear state between tests so the shared manager stays pristine.
    for slot in list(reg._slots.keys()):
        reg._slots[slot] = {
            "slot_id": slot,
            "version": None,
            "process_state": "idle",
            "pid": None,
            "in_flight": 0,
            "accepted": 0,
            "completed": 0,
            "failed": 0,
            "started_at": None,
            "idle_since": 0.0,
            "generation": reg._slots[slot]["generation"],
        }
    for version in list(reg._versions.keys()):
        del reg._versions[version]
    for key in list(reg._claims.keys()):
        del reg._claims[key]
    reg._state["active_version"] = None


def _load(reg: Registry, version: str) -> str:
    slot = reg.reserve_any_slot(version)
    reg.set_pid(version, 1000)
    return slot


def _activate(reg: Registry, version: str, old: str | None = None, token: str = "t") -> None:
    reg.begin_swap(old, token)
    reg.mark_candidate_tested(version)
    reg.commit_swap(version, old, token)


def test_first_load_claims_none_and_activates(registry):
    slot = _load(registry, "v1")
    assert slot == "slot-0"
    registry.mark_candidate_tested("v1")
    old = registry.begin_swap(None, "tok")
    registry.commit_swap("v1", old, "tok")
    assert registry.active_version() == "v1"
    assert registry.version_info("v1")["state"] == VERSION_ACTIVE
    assert registry.snapshot()["slots_in_use"] == 1


def test_second_swap_retires_old_but_keeps_slot_until_drained(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    _load(registry, "v2")
    _activate(registry, "v2", old="v1")

    assert registry.active_version() == "v2"
    assert registry.version_info("v1")["state"] == VERSION_RETIRED
    assert registry.snapshot()["slots_in_use"] == 2  # v1 still draining
    assert not registry.has_capacity()

    slot = registry.version_info("v1")["slot_id"]
    # In-flight work blocks release.
    version, _, _gen = registry.accept_work()
    assert version == "v2"
    assert registry.release_slot(slot, "v1", process_exited=True) is True  # no in-flight on v1
    registry.complete_work("v2")


def test_no_capacity_when_both_slots_held(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    _load(registry, "v2")
    _activate(registry, "v2", old="v1")
    with pytest.raises(NoCapacity):
        registry.reserve_any_slot("v3")
    # The current version is untouched.
    assert registry.active_version() == "v2"


def test_in_flight_request_blocks_slot_release(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    _load(registry, "v2")
    _activate(registry, "v2", old="v1")

    # A request accepted on v1 *before* the swap is counted on v1.
    # (Simulated here by accepting on v1 while it was active is not possible
    # after commit, so drive the counter the way the API does.)
    registry._slots["slot-0"] = {**registry._slots["slot-0"], "in_flight": 1}
    assert registry.release_slot("slot-0", "v1", process_exited=True) is False
    assert registry.release_slot("slot-0", "v1", process_exited=False) is False
    registry.complete_work("v1")
    assert registry.release_slot("slot-0", "v1", process_exited=True) is True
    assert registry.has_capacity()


def test_concurrent_swap_claim_allows_one_winner(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    assert registry.begin_swap("v1", "a") == "v1"
    with pytest.raises(SwapConflict, match="already in progress"):
        registry.begin_swap("v1", "b")
    registry.end_swap("v1", "a")
    assert registry.begin_swap("v1", "b") == "v1"


def test_stale_expected_version_is_conflict(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    with pytest.raises(SwapConflict, match="precondition"):
        registry.begin_swap("v0", "token")


def test_commit_requires_smoke_test(registry):
    _load(registry, "v1")
    registry.begin_swap(None, "t")
    with pytest.raises(SwapConflict, match="smoke test"):
        registry.commit_swap("v1", None, "t")


def test_failed_candidate_frees_slot_and_can_be_retried(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    registry.begin_swap("v1", "t")
    _load(registry, "bad")
    registry.fail_candidate("bad", "boom")
    registry.end_swap("v1", "t")

    info = registry.version_info("bad")
    assert info["state"] == VERSION_FAILED
    assert info["error"] == "boom"
    assert registry.has_capacity()
    assert registry.active_version() == "v1"

    registry.begin_swap("v1", "t2")
    _load(registry, "bad")  # retry allowed, keeps previous error
    assert registry.version_info("bad")["attempt"] == 2
    registry.end_swap("v1", "t2")


def test_duplicate_live_version_rejected(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    with pytest.raises(SwapConflict, match="already exists"):
        registry.reserve_any_slot("v1")


def test_accepted_work_is_pinned_to_active_version(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    version, slot, _gen = registry.accept_work()
    assert (version, slot) == ("v1", "slot-0")

    _load(registry, "v2")
    _activate(registry, "v2", old="v1")
    # The pre-existing in-flight count belongs to v1, not the new active.
    assert registry.version_in_flight("v1") == 1
    assert registry.version_in_flight("v2") == 0
    registry.complete_work("v1")
    assert registry.version_in_flight("v1") == 0


def test_feature_mismatch_does_not_count_in_flight(registry):
    _load(registry, "v1")
    registry.set_model_shape("v1", 3, 2)
    _activate(registry, "v1")
    with pytest.raises(FeatureMismatch):
        registry.accept_work(2)
    assert registry.version_in_flight("v1") == 0
    version, _, _gen = registry.accept_work(3)
    assert version == "v1"


def test_inference_without_model_is_rejected(registry):
    with pytest.raises(NoActiveModel):
        registry.accept_work()


def test_dead_active_worker_releases_slot(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    registry.accept_work()
    registry.settle_dead_worker("v1")
    assert registry.fail_active("v1", "crashed") is True
    assert registry.active_version() is None
    assert registry.has_capacity()
    assert registry.version_info("v1")["state"] == VERSION_FAILED


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------


def test_recovery_recovers_same_version_and_ignores_late_callback(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    version, slot, gen = registry.accept_work()
    assert (version, slot, gen) == ("v1", "slot-0", 1)

    ctx = registry.begin_recovery("v1", "crashed")
    assert ctx is not None
    assert ctx["generation"] == 2
    assert ctx["attempt"] == 1
    # One recovery per failure: a second begin for the same crash is a no-op.
    assert registry.begin_recovery("v1", "crashed again") is None
    # No usable model while recovering.
    assert registry.active_version() is None
    with pytest.raises(NoActiveModel):
        registry.accept_work()
    info = registry.version_info("v1")
    assert info["state"] == "recovering"
    assert info["recovery"]["state"] == "in_progress"
    assert info["recovery"]["crashed_pid"] == 1000
    assert info["recovery"]["crashed_generation"] == 1
    # The crashed slot was freed and its interrupted request counted failed.
    slot0 = next(s for s in registry.snapshot()["slots"] if s["slot_id"] == "slot-0")
    assert slot0["process_state"] == "idle"
    assert slot0["failed"] == 1
    assert slot0["in_flight"] == 0

    # A late completion from the dead process must not touch anything.
    registry.complete_work("v1", gen)
    slot0 = next(s for s in registry.snapshot()["slots"] if s["slot_id"] == "slot-0")
    assert slot0["completed"] == 0

    # Recovery loads in a new process and commits under the same version id.
    slot_id = registry.reserve_recovery_slot("v1")
    registry.set_pid("v1", 2000)
    registry.mark_recovery_tested("v1", 2)
    assert registry.commit_recovery("v1", 2) == "committed"
    assert registry.active_version() == "v1"
    info = registry.version_info("v1")
    assert info["state"] == VERSION_ACTIVE
    assert info["generation"] == 2
    assert info["pid"] == 2000
    assert info["recovery"]["state"] == "succeeded"
    assert info["recovery"]["recovered_pid"] == 2000
    assert info["recovery"]["recovered_generation"] == 2

    # New work is booked against the new process; stale completions are not.
    version, slot, gen2 = registry.accept_work()
    assert (version, gen2) == ("v1", 2)
    registry.complete_work("v1", 1)  # late callback from the crashed process
    assert registry.version_in_flight("v1") == 1
    registry.complete_work("v1", gen2)
    assert registry.version_in_flight("v1") == 0
    slot_view = next(s for s in registry.snapshot()["slots"] if s["slot_id"] == slot)
    assert slot_view["completed"] == 1  # only the genuine completion counted


def test_recovery_failure_is_final_and_frees_capacity(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    registry.begin_recovery("v1", "crashed")
    slot_id = registry.reserve_recovery_slot("v1")
    assert not any(
        s["process_state"] == "idle" and s["slot_id"] == slot_id
        for s in registry.snapshot()["slots"]
    )
    assert registry.fail_recovery("v1", "load blew up") is True

    info = registry.version_info("v1")
    assert info["state"] == VERSION_FAILED
    assert info["recovery"]["state"] == "failed"
    assert info["recovery"]["error"] == "load blew up"
    assert registry.active_version() is None
    assert registry.has_capacity()  # slot freed
    # No automatic retry for the same failure.
    assert registry.begin_recovery("v1", "again") is None
    # A failed version id may still be re-uploaded manually.
    registry.begin_swap(None, "t")
    _load(registry, "v1")
    assert registry.version_info("v1")["attempt"] == 2
    registry.end_swap(None, "t")


def test_manual_upload_supersedes_late_recovery(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    registry.begin_recovery("v1", "crashed")
    registry.reserve_recovery_slot("v1")

    # A manual upload loads in parallel in the other slot and wins.
    registry.begin_swap(None, "tok")
    _load(registry, "v2")
    registry.mark_candidate_tested("v2")
    registry.commit_swap("v2", None, "tok")
    assert registry.active_version() == "v2"

    # The late recovery must not take over.
    registry.mark_recovery_tested("v1", 2)
    assert registry.commit_recovery("v1", 2) == "superseded"
    assert registry.recovery_superseded("v1") is True
    info = registry.version_info("v1")
    assert info["state"] == VERSION_RETIRED
    assert info["recovery"]["state"] == "superseded"
    # The new version is untouched and its slot is still active.
    assert registry.active_version() == "v2"
    assert registry.version_info("v2")["state"] == VERSION_ACTIVE
    assert registry.has_capacity()  # recovery candidate's slot was freed


def test_manual_swap_commits_after_recovery_and_retires_it(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    registry.begin_recovery("v1", "crashed")
    registry.reserve_recovery_slot("v1")
    registry.mark_recovery_tested("v1", 2)

    # Manual swap begins while nothing is active; recovery commits first.
    registry.begin_swap(None, "tok")
    _load(registry, "v2")
    registry.mark_candidate_tested("v2")
    assert registry.commit_recovery("v1", 2) == "committed"
    assert registry.active_version() == "v1"
    # The manual upload still takes over: the recovered version drains.
    registry.commit_swap("v2", None, "tok")
    assert registry.active_version() == "v2"
    assert registry.version_info("v1")["state"] == VERSION_RETIRED


def test_manual_swap_survives_old_version_crashing_mid_swap(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    registry.begin_swap("v1", "tok")
    _load(registry, "v2")
    registry.mark_candidate_tested("v2")
    # The old version crashes while the manual swap is in flight.
    assert registry.begin_recovery("v1", "crashed") is not None
    # The swap still commits (nothing left to retire) and wins the takeover.
    registry.commit_swap("v2", "v1", "tok")
    assert registry.active_version() == "v2"
    # The recovery of the crashed version is then superseded.
    registry.reserve_recovery_slot("v1")
    registry.mark_recovery_tested("v1", 2)
    assert registry.commit_recovery("v1", 2) == "superseded"


def test_retired_version_does_not_self_heal(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    _load(registry, "v2")
    _activate(registry, "v2", old="v1")
    # A retired (draining) version never triggers a recovery.
    assert registry.begin_recovery("v1", "crashed") is None
    assert registry.version_info("v1")["state"] == VERSION_RETIRED
    assert registry.active_version() == "v2"
