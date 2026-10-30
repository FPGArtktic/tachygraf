# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""The command line queues work and nothing else; the worker does the work."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import cli, config, db


def _conn():
    return db.connect()


def test_add_then_queue_every_stage(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["add", "/in/lecture-04.mp4", "--slug", "lec4"]) == 0
    for stage in cli.STAGES:
        assert cli.main([stage, "--source", "lec4"]) == 0

    jobs = db.recent_jobs(_conn())
    assert [job["stage"] for job in reversed(jobs)] == list(cli.STAGES)
    assert "job 1 queued" in capsys.readouterr().out


def test_add_defaults_the_title_and_the_slug_to_the_file_name() -> None:
    cli.main(["add", "/in/Wykład 04.mkv"])
    row = db.list_sources(_conn())[0]
    assert row["slug"] == "wyklad-04"
    assert row["title"] == "Wykład 04"


def test_add_stores_the_resolved_media_path(tmp_path: Path) -> None:
    recording = tmp_path / "lecture-04.mp4"
    recording.write_bytes(b"x")
    cli.main(["add", str(recording), "--slug", "lec4"])
    assert db.list_sources(_conn())[0]["media_path"] == str(recording.resolve())


def test_add_records_the_output_folder_and_the_domain(tmp_path: Path) -> None:
    out = tmp_path / "staging"
    out.mkdir()
    cli.main(
        [
            "add",
            "/in/a.mp4",
            "--slug",
            "lec4",
            "--domain",
            "signal processing",
            "--out",
            str(out),
        ]
    )
    row = db.list_sources(_conn())[0]
    assert row["out_dir"] == str(out.resolve())
    assert row["domain"] == "signal processing"


def test_add_without_an_output_folder_leaves_it_unset() -> None:
    """An unset folder falls back to the mount, decided by the worker."""
    cli.main(["add", "/in/a.mp4", "--slug", "lec4"])
    assert db.list_sources(_conn())[0]["out_dir"] is None


def test_an_unknown_recording_is_refused_with_what_to_do() -> None:
    with pytest.raises(SystemExit, match="tachygraf add"):
        cli.main(["generate", "--source", "nothing"])


def test_the_generate_stage_defaults_to_the_text_model() -> None:
    cli.main(["add", "/in/a.mp4", "--slug", "lec4"])
    cli.main(["generate", "--source", "lec4"])
    assert db.recent_jobs(_conn())[0]["model"] == config.TEXT_MODEL


def test_the_transcribe_stage_defaults_to_the_asr_model() -> None:
    """The stage's model name is the ASR model, so two runs do not collide."""
    cli.main(["add", "/in/a.mp4", "--slug", "lec4"])
    cli.main(["transcribe", "--source", "lec4"])
    job = db.recent_jobs(_conn())[0]
    assert job["model"] == config.ASR_MODEL
    assert json.loads(job["params"])["language"] == config.LANGUAGE


def test_a_stage_that_needs_no_model_says_none() -> None:
    cli.main(["add", "/in/a.mp4", "--slug", "lec4"])
    cli.main(["index", "--source", "lec4"])
    assert db.recent_jobs(_conn())[0]["model"] == "none"


def test_the_language_reaches_the_job_parameters() -> None:
    cli.main(["add", "/in/a.mp4", "--slug", "lec4"])
    cli.main(["transcribe", "--source", "lec4", "--language", "pl"])
    assert json.loads(db.recent_jobs(_conn())[0]["params"])["language"] == "pl"


def test_the_vocabulary_reaches_the_job_parameters() -> None:
    cli.main(["add", "/in/a.mp4", "--slug", "lec4"])
    cli.main(["transcribe", "--source", "lec4", "--vocabulary", "Claude Code"])
    params = json.loads(db.recent_jobs(_conn())[0]["params"])
    assert params["vocabulary"] == "Claude Code"


def test_force_clears_the_previous_stage_rows() -> None:
    cli.main(["add", "/in/a.mp4", "--slug", "lec4"])
    conn = _conn()
    source_id = db.source_id_for(conn, "lec4")
    db.stage_start(conn, source_id, None, "generate", config.TEXT_MODEL)
    db.stage_finish(conn, source_id, None, "generate", config.TEXT_MODEL, db.DONE)

    cli.main(["generate", "--source", "lec4", "--force"])
    assert not db.stage_done(conn, source_id, None, "generate", config.TEXT_MODEL)
    assert json.loads(db.recent_jobs(conn)[0]["params"])["force"] is True


def test_without_force_the_previous_stage_rows_stay() -> None:
    cli.main(["add", "/in/a.mp4", "--slug", "lec4"])
    conn = _conn()
    source_id = db.source_id_for(conn, "lec4")
    db.stage_start(conn, source_id, None, "generate", config.TEXT_MODEL)
    db.stage_finish(conn, source_id, None, "generate", config.TEXT_MODEL, db.DONE)

    cli.main(["generate", "--source", "lec4"])
    assert db.stage_done(conn, source_id, None, "generate", config.TEXT_MODEL)


def test_queue_and_sources_print_without_a_database(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["queue"]) == 0
    assert cli.main(["sources"]) == 0
    assert capsys.readouterr().out == ""


def test_sources_lists_what_was_added(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["add", "/in/a.mp4", "--slug", "lec4", "--domain", "optics"])
    capsys.readouterr()
    cli.main(["sources"])
    out = capsys.readouterr().out
    assert "lec4" in out
    assert "optics" in out


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit):
        cli.main([])
