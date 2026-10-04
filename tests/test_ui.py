# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""The interface only queues jobs and reads state.

These tests exist for the reason glosator's do: the UI is the one module no
other test exercises, and a renamed gradio argument or a signature that no
longer matches only shows up when the container starts.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import gradio as gr
import pytest

from app import config, generate, transcribe, ui


def test_launch_options_are_still_accepted_by_the_installed_gradio() -> None:
    accepted = inspect.signature(gr.Blocks.launch).parameters
    assert set(ui.LAUNCH_OPTIONS) <= set(accepted)


def test_it_listens_on_every_interface_inside_the_container() -> None:
    """Published on the host at 7861, because glosator holds 7860."""
    assert ui.LAUNCH_OPTIONS["server_name"] == "0.0.0.0"
    assert ui.LAUNCH_OPTIONS["server_port"] == 7860
    assert ui.LAUNCH_OPTIONS["share"] is False


def test_build_returns_blocks_without_touching_the_network() -> None:
    assert isinstance(ui.build(), gr.Blocks)


def test_media_choices_lists_only_what_ffmpeg_decodes() -> None:
    for name in ("lecture-04.mp4", "talk.mkv", "notes.pdf", "audio.mp3"):
        (config.IN_DIR / name).write_bytes(b"x")
    (config.IN_DIR / "a-folder.mp4").mkdir()
    assert ui.media_choices(config.IN_DIR) == [
        "audio.mp3",
        "lecture-04.mp4",
        "talk.mkv",
    ]


def test_media_choices_of_a_missing_folder_is_empty() -> None:
    assert ui.media_choices("/nowhere-at-all") == []


def test_resolve_media_needs_a_choice() -> None:
    with pytest.raises(gr.Error, match="pick a recording"):
        ui.resolve_media(config.IN_DIR, None)


def test_resolve_media_reports_a_recording_that_is_gone() -> None:
    with pytest.raises(gr.Error, match="no such recording"):
        ui.resolve_media(config.IN_DIR, "deleted.mp4")


def test_resolve_out_dir_refuses_a_folder_that_does_not_exist() -> None:
    """Creating it blindly invites a typo becoming a new folder."""
    with pytest.raises(gr.Error, match="does not exist"):
        ui.resolve_out_dir("/not/a/real/staging/folder")


def test_resolve_out_dir_accepts_one_that_does(tmp_path: Path) -> None:
    assert ui.resolve_out_dir(str(tmp_path)) == tmp_path.resolve()


def test_the_device_note_says_cpu_when_the_card_is_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Said before the job: hours on the CPU is not a surprise to spring."""
    monkeypatch.setattr(transcribe, "gpu_free_mb", lambda: 200)
    note = ui.device_note("large-v3-turbo")
    assert "CPU" in note
    assert "longer than the recording" in note


def test_the_device_note_says_gpu_when_the_card_is_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(transcribe, "gpu_free_mb", lambda: 3800)
    assert "GPU" in ui.device_note("large-v3-turbo")


def test_the_device_note_names_the_model_it_is_talking_about(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The answer differs per model on a 4 GB card, so the note must say which."""
    monkeypatch.setattr(transcribe, "gpu_free_mb", lambda: 4004)
    assert "GPU" in ui.device_note("large-v3-turbo")
    assert "CPU" in ui.device_note("large-v3")


def test_last_note_skips_the_transcript_and_foreign_files() -> None:
    folder = config.OUT_DIR / "lec4"
    folder.mkdir(parents=True)
    (folder / "someone-elses.md").write_text("not ours\n", encoding="utf-8")
    header = "---\ngenerator: tachygraf\n---\n"
    (folder / transcribe.TRANSCRIPT_NOTE).write_text(
        header + "# Transcript\n", encoding="utf-8"
    )
    (folder / "01 A note.md").write_text(header + "# 01 A note\n", encoding="utf-8")

    shown = ui.last_note(config.OUT_DIR)
    assert "# 01 A note" in shown
    assert "Transcript" not in shown


def test_last_note_says_so_when_there_is_nothing_yet() -> None:
    assert ui.last_note(config.OUT_DIR) == "_no notes yet_"
    assert ui.last_note("/nowhere-at-all") == "_output folder not found_"


def test_stop_with_no_job_running_says_so() -> None:
    from app import db

    conn = db.connect()
    assert "Nothing to stop" in ui.stop_jobs(conn, None)


def test_stop_everything_asks_every_pending_job() -> None:
    from app import db

    conn = db.connect()
    source_id = db.upsert_source(conn, "lec4", "L4", Path("/in/a.mp4"))
    db.enqueue(conn, source_id, "transcribe", "large-v3")
    db.enqueue(conn, source_id, "generate", config.TEXT_MODEL)

    message = ui.stop_jobs(conn, None)
    assert "2 job(s)" in message
    assert all(job["stop"] for job in db.recent_jobs(conn))


def test_stop_one_job_by_id() -> None:
    from app import db

    conn = db.connect()
    source_id = db.upsert_source(conn, "lec4", "L4", Path("/in/a.mp4"))
    first = db.enqueue(conn, source_id, "generate", config.TEXT_MODEL)
    second = db.enqueue(conn, source_id, "index", "none")

    ui.stop_jobs(conn, first)
    rows = {row["id"]: row for row in db.recent_jobs(conn)}
    assert rows[first]["stop"]
    assert not rows[second]["stop"]


def test_kill_worker_without_one_running_is_false() -> None:
    assert ui.kill_worker() is False


def test_the_job_table_has_the_columns_it_declares() -> None:
    from app import db

    conn = db.connect()
    source_id = db.upsert_source(conn, "lec4", "L4", Path("/in/a.mp4"))
    db.enqueue(conn, source_id, "generate", config.TEXT_MODEL)
    rows = ui.job_table(conn)
    assert len(rows[0]) == len(ui.JOB_COLUMNS)


def test_a_job_asked_to_stop_reads_as_stopping_while_it_finishes() -> None:
    from app import db

    conn = db.connect()
    source_id = db.upsert_source(conn, "lec4", "L4", Path("/in/a.mp4"))
    job_id = db.enqueue(conn, source_id, "generate", config.TEXT_MODEL)
    db.claim_job(conn)
    db.request_stop(conn, job_id)
    assert ui.job_table(conn)[0][2] == "stopping"


def test_the_generator_marker_is_the_one_the_ui_looks_for() -> None:
    """is_ours is what keeps the preview and the index off the user's files."""
    assert generate.GENERATOR == "tachygraf"
