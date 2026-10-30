# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""The queue consumer: resuming, failing one window, stopping, giving the card back."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import chunk as chunk_module
from app import config, db, generate, index, media, ollama, transcribe, worker
from app.transcribe import Segment


@pytest.fixture
def conn():
    connection = db.connect()
    yield connection
    connection.close()


@pytest.fixture
def source_id(conn) -> int:
    return db.upsert_source(
        conn,
        "lec4",
        "Lecture 4",
        Path("/in/lecture-04.mp4"),
        config.OUT_DIR,
        "signal processing",
    )


@pytest.fixture
def planned(conn, source_id, monkeypatch: pytest.MonkeyPatch) -> list:
    """A recording with a transcript and three windows, ready to generate."""
    monkeypatch.setattr(config, "CHUNK_MIN_TOKENS", 5)
    monkeypatch.setattr(config, "CHUNK_MAX_TOKENS", 40)
    transcribe.write_segments(
        "lec4",
        [
            Segment(0.0, 10.0, "Pierwszy fragment o próbkowaniu sygnalu.", -0.2),
            Segment(14.0, 24.0, "Drugi fragment o aliasingu i filtrach.", -0.2),
            Segment(28.0, 38.0, "Trzeci fragment po dluzszej przerwie.", -0.2),
            Segment(42.0, 52.0, "Czwarty fragment o twierdzeniu.", -0.2),
        ],
        {"language": "pl", "duration_s": 3600.0},
    )
    windows = chunk_module.plan("lec4")
    db.replace_chunks(conn, source_id, windows)
    return windows


def _window_labels(prompt: str) -> list[str]:
    """The labels of the window, not the ones in the prompt's own examples.

    note.md illustrates the form with [p0042]; a model that copied it would be
    rejected by the parser, which is the point of that check but not of these
    tests.
    """
    source = prompt.split("--- SOURCE TEXT ---", 1)[1]
    return generate.labels_in(source)


def _answer(labels) -> str:
    label = next(iter(labels))
    return (
        "## In short\nWhat this window of the lecture is about.\n\n"
        "## Explanation\nThe speaker developed the point in this order.\n\n"
        f"## Key claims\n- A claim the speaker made. [{label}]\n\n"
        "## Formulas\nNone in this window.\n\n"
        "## Glossary\n- aliasing — aliasing\n"
    )


@pytest.fixture
def model_answers(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A model that answers every window correctly, and counts the calls."""
    asked: list[str] = []

    def fake_generate(model: str, prompt: str) -> str:
        asked.append(prompt)
        return _answer(_window_labels(prompt))

    monkeypatch.setattr(ollama, "generate", fake_generate)
    monkeypatch.setattr(ollama, "unload", lambda model: None)
    return asked


def _job(conn, source_id, stage, model="none", **params) -> int:
    job_id = db.enqueue(conn, source_id, stage, model, json.dumps(params))
    return job_id


def _run_next(conn) -> None:
    worker._run_job(conn, db.claim_job(conn))


# --- the stages -------------------------------------------------------------


def test_an_unknown_stage_fails_the_job_rather_than_the_worker(conn, source_id) -> None:
    _job(conn, source_id, "polish")
    _run_next(conn)
    job = db.recent_jobs(conn)[0]
    assert job["status"] == db.FAILED
    assert "unknown stage polish" in job["error"]


def test_the_transcribe_stage_hands_on_the_vocabulary(
    conn, source_id, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def run(slug, media_path, **kwargs):
        seen.update(kwargs)
        return config.OUT_DIR / slug / transcribe.TRANSCRIPT_NOTE

    monkeypatch.setattr(transcribe, "run", run)
    _job(conn, source_id, "transcribe", "large-v3", vocabulary="Claude Code")
    _run_next(conn)
    assert db.recent_jobs(conn)[0]["status"] == db.DONE
    assert seen["vocabulary"] == "Claude Code"


def test_the_extract_stage_records_the_audio_it_wrote(
    conn, source_id, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        media,
        "run",
        lambda slug, path, force: media.MediaInfo(path, 60.0, 1, "pol", "aac"),
    )
    media.audio_path("lec4").parent.mkdir(parents=True, exist_ok=True)
    media.audio_path("lec4").write_bytes(b"RIFF")

    _job(conn, source_id, "extract")
    _run_next(conn)
    assert db.recent_jobs(conn)[0]["status"] == db.DONE
    assert db.stage_rows(conn, source_id, "extract")[0]["output_path"].endswith(
        "audio.wav"
    )


def test_the_chunk_stage_stores_the_plan_in_the_database(
    conn, source_id, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "CHUNK_MIN_TOKENS", 5)
    transcribe.write_segments(
        "lec4",
        [
            Segment(0.0, 10.0, "Jeden dwa trzy.", -0.2),
            Segment(20.0, 30.0, "Cztery.", -0.2),
        ],
        {"language": "pl"},
    )
    _job(conn, source_id, "chunk")
    _run_next(conn)
    assert db.recent_jobs(conn)[0]["status"] == db.DONE
    assert db.list_chunks(conn, source_id)


def test_the_generate_stage_writes_one_note_per_window(
    conn, source_id, planned, model_answers
) -> None:
    _job(conn, source_id, "generate", config.TEXT_MODEL)
    _run_next(conn)

    job = db.recent_jobs(conn)[0]
    assert job["status"] == db.DONE
    assert job["done"] == len(planned)
    assert len(model_answers) == len(planned)
    notes = sorted((config.OUT_DIR / "lec4").glob("[0-9]*.md"))
    assert len(notes) == len(planned)


def test_a_finished_window_is_not_generated_again(
    conn, source_id, planned, model_answers
) -> None:
    """This is what makes a killed container resume instead of starting again."""
    _job(conn, source_id, "generate", config.TEXT_MODEL)
    _run_next(conn)
    first_pass = len(model_answers)

    _job(conn, source_id, "generate", config.TEXT_MODEL)
    _run_next(conn)
    assert len(model_answers) == first_pass, "the model was asked a second time"
    assert db.recent_jobs(conn)[0]["done"] == len(planned)


def test_force_generates_a_finished_window_again(
    conn, source_id, planned, model_answers
) -> None:
    _job(conn, source_id, "generate", config.TEXT_MODEL)
    _run_next(conn)
    first_pass = len(model_answers)

    _job(conn, source_id, "generate", config.TEXT_MODEL, force=True)
    _run_next(conn)
    assert len(model_answers) == first_pass * 2


def test_one_rejected_response_does_not_stop_the_job(
    conn, source_id, planned, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed window is recorded with the rule it broke and the job goes on."""
    calls: list[int] = []

    def sometimes_bad(model: str, prompt: str) -> str:
        calls.append(1)
        if len(calls) == 1:
            return _answer(_window_labels(prompt)).replace(
                "## Explanation\nThe speaker developed the point in this order.", ""
            )
        return _answer(_window_labels(prompt))

    monkeypatch.setattr(ollama, "generate", sometimes_bad)
    monkeypatch.setattr(ollama, "unload", lambda model: None)

    _job(conn, source_id, "generate", config.TEXT_MODEL)
    _run_next(conn)

    assert db.recent_jobs(conn)[0]["status"] == db.DONE
    rows = db.stage_rows(conn, source_id, "generate")
    failed = [row for row in rows if row["status"] == db.FAILED]
    assert len(failed) == 1
    assert failed[0]["error"].startswith("rule broken: missing section")
    assert len(calls) == len(planned), "the remaining windows were still generated"


def test_a_rejected_window_is_retried_on_the_next_run(
    conn, source_id, planned, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[int] = []
    bad = {"first": True}

    def answer(model: str, prompt: str) -> str:
        attempts.append(1)
        labels = _window_labels(prompt)
        text = _answer(labels)
        if bad["first"]:
            return text.replace(f"[{labels[0]}]", "[p9999]")
        return text

    monkeypatch.setattr(ollama, "generate", answer)
    monkeypatch.setattr(ollama, "unload", lambda model: None)

    _job(conn, source_id, "generate", config.TEXT_MODEL)
    _run_next(conn)
    assert any(
        row["status"] == db.FAILED for row in db.stage_rows(conn, source_id, "generate")
    )

    bad["first"] = False
    _job(conn, source_id, "generate", config.TEXT_MODEL)
    _run_next(conn)
    assert all(
        row["status"] == db.DONE for row in db.stage_rows(conn, source_id, "generate")
    )


def test_a_stop_request_ends_the_job_without_the_remaining_windows(
    conn, source_id, planned, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = _job(conn, source_id, "generate", config.TEXT_MODEL)
    calls: list[int] = []

    def answer_then_stop(model: str, prompt: str) -> str:
        calls.append(1)
        db.request_stop(conn, job_id)
        return _answer(_window_labels(prompt))

    monkeypatch.setattr(ollama, "generate", answer_then_stop)
    monkeypatch.setattr(ollama, "unload", lambda model: None)

    _run_next(conn)
    assert db.recent_jobs(conn)[0]["status"] == db.STOPPED
    assert len(calls) == 1
    assert len(planned) > 1


def test_the_index_stage_maps_the_notes_the_generate_stage_wrote(
    conn, source_id, planned, model_answers
) -> None:
    """A note is named after a title the model chose, so the path is read back."""
    _job(conn, source_id, "generate", config.TEXT_MODEL)
    _run_next(conn)
    _job(conn, source_id, "index")
    _run_next(conn)

    assert db.recent_jobs(conn)[0]["status"] == db.DONE
    map_text = (config.OUT_DIR / "lec4" / index.MAP_NOTE).read_text(encoding="utf-8")
    for window in planned:
        assert f"[[{window.ord:02d} " in map_text
    # The duration comes from segments.json, not from a second ffprobe.
    assert 'duration: "01:00:00"' in map_text


def test_the_index_stage_leaves_a_window_that_produced_no_note_out_of_the_map(
    conn, source_id, planned, monkeypatch: pytest.MonkeyPatch
) -> None:
    only_the_first = {"done": False}

    def answer(model: str, prompt: str) -> str:
        if only_the_first["done"]:
            raise ollama.OllamaUnavailable("the card went away")
        only_the_first["done"] = True
        return _answer(_window_labels(prompt))

    monkeypatch.setattr(ollama, "generate", answer)
    monkeypatch.setattr(ollama, "unload", lambda model: None)

    _job(conn, source_id, "generate", config.TEXT_MODEL)
    _run_next(conn)
    _job(conn, source_id, "index")
    _run_next(conn)

    map_text = (config.OUT_DIR / "lec4" / index.MAP_NOTE).read_text(encoding="utf-8")
    assert map_text.count("[[0") == 1


# --- the card ---------------------------------------------------------------


def test_the_text_model_is_unloaded_when_a_generate_job_ends(
    conn, source_id, planned, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The next job's ASR stage needs the card; 4 GB does not hold both."""
    unloaded: list[str] = []
    monkeypatch.setattr(ollama, "unload", unloaded.append)
    monkeypatch.setattr(
        ollama, "generate", lambda model, prompt: _answer(_window_labels(prompt))
    )

    _job(conn, source_id, "generate", config.TEXT_MODEL)
    _run_next(conn)
    assert unloaded == [config.TEXT_MODEL]


def test_the_text_model_is_unloaded_even_when_the_job_failed(
    conn, source_id, monkeypatch: pytest.MonkeyPatch
) -> None:
    unloaded: list[str] = []
    monkeypatch.setattr(ollama, "unload", unloaded.append)
    # No plan on disk, so the stage raises before any model call.
    _job(conn, source_id, "generate", config.TEXT_MODEL)
    _run_next(conn)
    assert db.recent_jobs(conn)[0]["status"] == db.FAILED
    assert unloaded == [config.TEXT_MODEL]


def test_a_stage_that_never_touched_ollama_does_not_unload_anything(
    conn, source_id, monkeypatch: pytest.MonkeyPatch
) -> None:
    unloaded: list[str] = []
    monkeypatch.setattr(ollama, "unload", unloaded.append)
    _job(conn, source_id, "chunk")
    _run_next(conn)
    assert unloaded == []


def test_the_index_stage_runs_without_the_original_recording(
    conn, source_id, planned, model_answers
) -> None:
    """It is the stage that runs again after a stop, a failure or a --force."""
    _job(conn, source_id, "generate", config.TEXT_MODEL)
    _run_next(conn)

    def gone(_path):
        raise media.MediaError("the recording has been moved away")

    original = media.probe
    media.probe = gone
    try:
        _job(conn, source_id, "index")
        _run_next(conn)
    finally:
        media.probe = original

    assert db.recent_jobs(conn)[0]["status"] == db.DONE
    assert (config.OUT_DIR / "lec4" / index.MAP_NOTE).exists()
