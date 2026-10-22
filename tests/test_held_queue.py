from datetime import datetime, timedelta, timezone

import pytest

from video_prompter.held_queue import HeldQueue


def payload():
    return {"prompt": {"1": {"class_type": "SaveVideo", "inputs": {}}}, "prompt_id": "abc"}


def test_held_job_is_persistent_and_not_due(tmp_path):
    path = tmp_path / "held.json"
    created = HeldQueue(path).add(payload(), {"shot_name": "Wide shot"})

    reloaded = HeldQueue(path).list()
    assert reloaded[0]["id"] == created["id"]
    assert reloaded[0]["status"] == "held"
    assert HeldQueue(path).due() == []


def test_scheduled_job_becomes_due(tmp_path):
    store = HeldQueue(tmp_path / "held.json")
    now = datetime.now(timezone.utc)
    past = (now - timedelta(minutes=1)).isoformat()
    future = (now + timedelta(hours=1)).isoformat()
    due = store.add(payload(), {}, past)
    store.add(payload(), {}, future)

    assert [job["id"] for job in store.due(now)] == [due["id"]]


def test_schedule_can_be_changed_or_cleared(tmp_path):
    store = HeldQueue(tmp_path / "held.json")
    job = store.add(payload(), {})
    release = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()

    updated = store.update_schedule(job["id"], release)
    assert updated["status"] == "scheduled"
    assert updated["release_at"]
    assert store.update_schedule(job["id"], None)["status"] == "held"


def test_dispatching_job_is_not_retried_automatically(tmp_path):
    store = HeldQueue(tmp_path / "held.json")
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    job = store.add(payload(), {}, past)

    dispatching = store.mark_dispatching(job["id"])
    assert dispatching["attempts"] == 1
    assert store.due() == []


def test_dispatch_claim_cannot_be_reopened_by_a_second_release_or_schedule(tmp_path):
    store = HeldQueue(tmp_path / "held.json")
    job = store.add(payload(), {})
    store.mark_dispatching(job["id"])
    with pytest.raises(ValueError, match="already being dispatched"):
        store.mark_dispatching(job["id"])
    with pytest.raises(ValueError, match="already being dispatched"):
        store.update_schedule(job["id"], None)
    assert store.get(job["id"])["attempts"] == 1


def test_release_time_requires_timezone(tmp_path):
    store = HeldQueue(tmp_path / "held.json")
    with pytest.raises(ValueError, match="timezone"):
        store.add(payload(), {}, "2026-08-18T01:00:00")


def test_whole_shelf_takes_one_run_time(tmp_path):
    store = HeldQueue(tmp_path / "held.json")
    for _ in range(3):
        store.add(payload(), {})
    release = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()

    changed = store.update_schedule_all(release)

    assert len(changed) == 3
    assert {job["status"] for job in store.list()} == {"scheduled"}
    assert all(job["release_at"] for job in store.list())
    assert {job["status"] for job in store.update_schedule_all(None)} == {"held"}


def test_bulk_schedule_leaves_a_job_already_dispatching_alone(tmp_path):
    store = HeldQueue(tmp_path / "held.json")
    waiting = store.add(payload(), {})
    flying = store.add(payload(), {})
    store.mark_dispatching(flying["id"])
    release = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()

    changed = store.update_schedule_all(release)

    assert [job["id"] for job in changed] == [waiting["id"]]
    assert store.get(flying["id"])["status"] == "dispatching"


def test_bulk_run_time_requires_timezone(tmp_path):
    store = HeldQueue(tmp_path / "held.json")
    store.add(payload(), {})
    with pytest.raises(ValueError, match="timezone"):
        store.update_schedule_all("2026-09-01T01:00:00")


def test_a_job_orphaned_mid_dispatch_is_unstuck_as_an_error(tmp_path):
    # A server that dies during the POST leaves the job "dispatching", and
    # nothing would ever move it on. It must not be resubmitted blindly.
    path = tmp_path / "held.json"
    queue = HeldQueue(path)
    job = queue.add(payload(), {})
    queue.mark_dispatching(job["id"])

    recovered = HeldQueue(path).recover_interrupted()

    assert [j["id"] for j in recovered] == [job["id"]]
    stored = HeldQueue(path).get(job["id"])
    assert stored["status"] == "error"
    assert "render twice" in stored["last_error"]
    assert HeldQueue(path).mark_dispatching(job["id"])["status"] == "dispatching"


def test_an_unreadable_queue_file_is_kept_aside_not_overwritten(tmp_path):
    path = tmp_path / "held.json"
    path.write_text('{"jobs": [ {"id": "precious"', encoding="utf-8")

    queue = HeldQueue(path)
    assert queue.list() == []
    queue.add(payload(), {})

    aside = list(tmp_path.glob("held.corrupt-*.json"))
    assert len(aside) == 1
    assert "precious" in aside[0].read_text(encoding="utf-8")
