# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""The state that makes a killed container resumable."""

from __future__ import annotations

from pathlib import Path

import pytest

from app import db
from app.chunk import Window


@pytest.fixture
def conn():
    connection = db.connect()
    yield connection
    connection.close()


@pytest.fixture
def source_id(conn) -> int:
    return db.upsert_source(
        conn, "lec4", "Lecture 4", Path("/in/lecture-04.mp4"), None, "signal processing"
    )


def _windows(count: int = 2) -> list[Window]:
    return [
        Window(
            ord=order,
            start_s=float(order * 100),
            end_s=float(order * 100 + 99),
            labels=(f"p{order:04d}", f"p{order + 1:04d}"),
            text_path=Path(f"/work/lec4/windows/{order:03d}.md"),
        )
        for order in range(1, count + 1)
    ]


def test_upsert_source_is_idempotent_on_the_slug(conn) -> None:
    first = db.upsert_source(conn, "lec4", "Lecture 4", Path("/in/a.mp4"))
    second = db.upsert_source(conn, "lec4", "Lecture 4, again", Path("/in/b.mp4"))
    assert first == second
    row = db.source_row(conn, first)
    assert row["title"] == "Lecture 4, again"
    assert row["media_path"] == "/in/b.mp4"


def test_source_id_for_an_unknown_slug_is_none(conn) -> None:
    assert db.source_id_for(conn, "nothing") is None


def test_replace_chunks_stores_the_labels_of_each_window(conn, source_id) -> None:
    db.replace_chunks(conn, source_id, _windows())
    rows = db.list_chunks(conn, source_id)
    assert [row["ord"] for row in rows] == [1, 2]
    assert rows[0]["labels"] == "p0001 p0002"
    assert rows[1]["start_s"] == 200.0


def test_replacing_a_plan_drops_windows_that_are_no_longer_in_it(
    conn, source_id
) -> None:
    db.replace_chunks(conn, source_id, _windows(3))
    db.replace_chunks(conn, source_id, _windows(1))
    assert [row["ord"] for row in db.list_chunks(conn, source_id)] == [1]


def test_re_planning_an_unchanged_transcript_keeps_the_finished_windows(
    conn, source_id
) -> None:
    """The chunk stage is a pure function of the transcript; a rerun is a no-op."""
    db.replace_chunks(conn, source_id, _windows(2))
    for row in db.list_chunks(conn, source_id):
        db.stage_start(conn, source_id, row["id"], "generate", "m")
        db.stage_finish(
            conn, source_id, row["id"], "generate", "m", db.DONE, Path("/out/n.md")
        )

    db.replace_chunks(conn, source_id, _windows(2))
    rows = db.list_chunks(conn, source_id)
    assert [row["ord"] for row in rows] == [1, 2]
    assert all(
        db.stage_done(conn, source_id, row["id"], "generate", "m") for row in rows
    )


def test_a_window_whose_content_changed_is_generated_again(conn, source_id) -> None:
    """A note made from different speech is not the right note any more."""
    db.replace_chunks(conn, source_id, _windows(1))
    first = db.list_chunks(conn, source_id)[0]["id"]
    db.stage_start(conn, source_id, first, "generate", "m")
    db.stage_finish(conn, source_id, first, "generate", "m", db.DONE, Path("/out/n.md"))

    moved = [
        Window(
            ord=1,
            start_s=100.0,
            end_s=199.0,
            labels=("p0001", "p0002", "p0003"),
            text_path=Path("/work/lec4/windows/001.md"),
        )
    ]
    db.replace_chunks(conn, source_id, moved)
    row = db.list_chunks(conn, source_id)[0]
    assert row["labels"] == "p0001 p0002 p0003"
    assert not db.stage_done(conn, source_id, row["id"], "generate", "m")
    assert db.stage_outputs(conn, source_id, "generate") == {}


def test_stage_round_trip_for_one_window(conn, source_id) -> None:
    db.replace_chunks(conn, source_id, _windows())
    chunk_id = db.list_chunks(conn, source_id)[0]["id"]
    assert not db.stage_done(conn, source_id, chunk_id, "generate", "qwen2.5:32b")
    db.stage_start(conn, source_id, chunk_id, "generate", "qwen2.5:32b")
    db.stage_finish(
        conn, source_id, chunk_id, "generate", "qwen2.5:32b", db.DONE, Path("/out/x.md")
    )
    assert db.stage_done(conn, source_id, chunk_id, "generate", "qwen2.5:32b")


def test_a_second_asr_model_does_not_collide_with_the_first(conn, source_id) -> None:
    """CLAUDE.md: the transcribe stage's model name is the ASR model name."""
    db.stage_start(conn, source_id, None, "transcribe", "large-v3")
    db.stage_finish(conn, source_id, None, "transcribe", "large-v3", db.DONE)
    assert db.stage_done(conn, source_id, None, "transcribe", "large-v3")
    assert not db.stage_done(conn, source_id, None, "transcribe", "large-v3-turbo")


def test_stage_finish_records_the_rule_a_response_broke(conn, source_id) -> None:
    db.stage_start(conn, source_id, None, "generate", "m")
    db.stage_finish(
        conn, source_id, None, "generate", "m", db.FAILED, error="rule broken: x"
    )
    assert db.stage_rows(conn, source_id, "generate")[0]["error"] == "rule broken: x"


def test_stage_outputs_gives_the_newest_path_per_window(conn, source_id) -> None:
    db.replace_chunks(conn, source_id, _windows())
    chunk_id = db.list_chunks(conn, source_id)[0]["id"]
    for model, path in (("old", "/out/lec4/01 Old.md"), ("new", "/out/lec4/01 New.md")):
        db.stage_start(conn, source_id, chunk_id, "generate", model)
        db.stage_finish(
            conn, source_id, chunk_id, "generate", model, db.DONE, Path(path)
        )
    assert db.stage_outputs(conn, source_id, "generate") == {
        chunk_id: Path("/out/lec4/01 New.md")
    }


def test_stage_outputs_ignores_a_failed_window(conn, source_id) -> None:
    db.replace_chunks(conn, source_id, _windows())
    chunk_id = db.list_chunks(conn, source_id)[0]["id"]
    db.stage_start(conn, source_id, chunk_id, "generate", "m")
    db.stage_finish(conn, source_id, chunk_id, "generate", "m", db.FAILED, error="no")
    assert db.stage_outputs(conn, source_id, "generate") == {}


def test_clear_stage_forgets_only_that_stage(conn, source_id) -> None:
    for stage in ("transcribe", "generate"):
        db.stage_start(conn, source_id, None, stage, "m")
        db.stage_finish(conn, source_id, None, stage, "m", db.DONE)
    assert db.clear_stage(conn, source_id, "generate") == 1
    assert db.stage_done(conn, source_id, None, "transcribe", "m")
    assert not db.stage_done(conn, source_id, None, "generate", "m")


def test_job_queue_claim_and_finish(conn, source_id) -> None:
    job_id = db.enqueue(conn, source_id, "generate", "qwen2.5:32b")
    claimed = db.claim_job(conn)
    assert claimed["id"] == job_id
    assert claimed["status"] == db.RUNNING
    assert db.claim_job(conn) is None
    db.finish_job(conn, job_id, db.DONE)
    assert db.recent_jobs(conn)[0]["status"] == db.DONE


def test_stop_all_asks_every_unfinished_job(conn, source_id) -> None:
    first = db.enqueue(conn, source_id, "transcribe", "large-v3")
    second = db.enqueue(conn, source_id, "generate", "m")
    db.claim_job(conn)
    assert db.stop_all(conn) == [first, second]
    assert db.stop_requested(conn, first)
    # The queued one is settled outright; the running one is only asked.
    rows = {row["id"]: row for row in db.recent_jobs(conn)}
    assert rows[second]["status"] == db.STOPPED
    assert rows[first]["status"] == db.RUNNING


def test_reset_running_jobs_requeues_one_that_was_not_asked_to_stop(
    conn, source_id
) -> None:
    job_id = db.enqueue(conn, source_id, "generate", "m")
    db.claim_job(conn)
    db.reset_running_jobs(conn)
    assert db.recent_jobs(conn)[0]["status"] == db.QUEUED
    assert db.claim_job(conn)["id"] == job_id


def test_reset_running_jobs_settles_one_that_was_killed_to_stop_it(
    conn, source_id
) -> None:
    """A kill is how a stop gets carried out; re-queueing would undo it."""
    job_id = db.enqueue(conn, source_id, "generate", "m")
    db.claim_job(conn)
    db.request_stop(conn, job_id)
    db.reset_running_jobs(conn)
    assert db.recent_jobs(conn)[0]["status"] == db.STOPPED


def test_clear_finished_keeps_what_each_stage_completed(conn, source_id) -> None:
    db.stage_start(conn, source_id, None, "generate", "m")
    db.stage_finish(conn, source_id, None, "generate", "m", db.DONE)
    job_id = db.enqueue(conn, source_id, "generate", "m")
    db.finish_job(conn, job_id, db.DONE)
    assert db.clear_finished(conn) == 1
    assert db.recent_jobs(conn) == []
    assert db.stage_done(conn, source_id, None, "generate", "m")
