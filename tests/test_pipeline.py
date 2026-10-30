# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""One recording through all five stages, with a fake ASR model and a fake Ollama.

This is the test that would have caught the two things unit tests cannot: a
label in a note that does not match the transcript's block IDs, and a stage
whose output the next stage cannot find.
"""

from __future__ import annotations

import json
import re
import wave
from pathlib import Path

import pytest

from app import cli, config, db, generate, index, media, ollama, transcribe, worker

PARAGRAPHS = [
    "Dzisiaj zajmiemy sie probkowaniem sygnalu.",
    "Pytanie, ktore zawsze pada, brzmi: dlaczego nie mozemy filtrowac po probkowaniu.",
    "Odpowiedz jest taka, ze alias zajmuje te same czestotliwosci co sygnal.",
    "Zapisze warunek na tablicy.",
    "To jest twierdzenie o probkowaniu.",
    "Na koniec jeszcze jedna uwaga o filtrach antyaliasingowych.",
]


class FakeSegment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text
        self.avg_logprob = -0.2


class FakeInfo:
    language = "pl"
    language_probability = 0.99
    duration = 420.0


class FakeAsr:
    def transcribe(self, audio, **kwargs):
        # Ten seconds of speech, then a four-second pause, per paragraph: long
        # enough that every paragraph is its own block and every gap is a cut.
        segments = [
            FakeSegment(index * 14.0, index * 14.0 + 10.0, text)
            for index, text in enumerate(PARAGRAPHS)
        ]
        return iter(segments), FakeInfo()


def _model_answer(prompt: str) -> str:
    """A well-behaved model: it cites only labels it was given."""
    source = prompt.split("--- SOURCE TEXT ---", 1)[1]
    labels = generate.labels_in(source)
    claims = "\n".join(f"- A claim the speaker made. [{label}]" for label in labels)
    return (
        "## In short\nWhy filtering after sampling cannot undo aliasing. "
        "A second sentence that is not the title.\n\n"
        "## Explanation\nThe speaker developed the point in this order.\n\n"
        f"## Key claims\n{claims}\n\n"
        f"## Formulas\n$$f_s > 2 f_{{max}}$$\nThe condition. [{labels[0]}]\n\n"
        "## Glossary\n- aliasing — aliasing\n"
        "- sampling rate — częstotliwość próbkowania\n\n"
        f"## Pitfalls\n- A filter afterwards does not help. [{labels[0]}]\n"
    )


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, asr_in_process) -> dict:
    recording = config.IN_DIR / "lecture-04.mp4"
    recording.write_bytes(b"not really an mp4")

    monkeypatch.setattr(
        media, "probe", lambda path: media.MediaInfo(path, 420.0, 1, "pol", "aac")
    )
    monkeypatch.setattr(
        media, "extract_audio", lambda slug, path, force=False: _fake_audio(slug)
    )
    monkeypatch.setattr(transcribe, "gpu_free_mb", lambda: 15000)
    monkeypatch.setattr(
        transcribe, "load_model", lambda name, device, compute: FakeAsr()
    )
    monkeypatch.setattr(ollama, "generate", lambda model, prompt: _model_answer(prompt))
    monkeypatch.setattr(ollama, "unload", lambda model: None)
    monkeypatch.setattr(config, "CHUNK_MIN_TOKENS", 12)
    monkeypatch.setattr(config, "CHUNK_MAX_TOKENS", 60)
    return {"recording": recording}


def _fake_audio(slug: str) -> Path:
    """A real, short WAV in the format the extract stage produces.

    The fake ASR model ignores the samples, but the stage reads them, so this
    has to be a WAV the standard library can open. Written here with the wave
    module rather than committed as a fixture.
    """
    path = media.audio_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(config.AUDIO_CHANNELS)
        handle.setsampwidth(2)
        handle.setframerate(config.AUDIO_RATE)
        handle.writeframes(b"\x00\x00" * config.AUDIO_RATE)
    return path


def _run_everything() -> None:
    cli.main(
        [
            "add",
            str(config.IN_DIR / "lecture-04.mp4"),
            "--slug",
            "lec4",
            "--title",
            "Wykład 4",
            "--domain",
            "signal processing",
            "--out",
            str(config.OUT_DIR),
        ]
    )
    for stage in cli.STAGES:
        cli.main([stage, "--source", "lec4"])

    conn = db.connect()
    while True:
        job = db.claim_job(conn)
        if job is None:
            break
        worker._run_job(conn, job)


def test_every_stage_finishes(pipeline) -> None:
    _run_everything()
    jobs = db.recent_jobs(db.connect())
    assert [job["status"] for job in jobs] == [db.DONE] * len(cli.STAGES), [
        (job["stage"], job["error"]) for job in jobs
    ]


def test_the_output_folder_holds_the_contract(pipeline) -> None:
    _run_everything()
    folder = config.OUT_DIR / "lec4"
    names = sorted(path.name for path in folder.glob("*.md"))
    assert transcribe.TRANSCRIPT_NOTE in names
    assert index.MAP_NOTE in names
    assert index.GLOSSARY_NOTE in names
    assert any(re.fullmatch(r"\d\d .+\.md", name) for name in names)
    # Nothing was written anywhere else under the output folder.
    assert sorted(path.name for path in config.OUT_DIR.iterdir()) == ["lec4"]


def test_every_cited_label_exists_in_the_transcript(pipeline) -> None:
    """The one thing the model must get right, checked end to end."""
    _run_everything()
    folder = config.OUT_DIR / "lec4"
    transcript = (folder / transcribe.TRANSCRIPT_NOTE).read_text(encoding="utf-8")
    block_ids = set(re.findall(r"\^(p\d{4})", transcript))
    assert block_ids

    cited = set()
    for note in folder.glob("[0-9]*.md"):
        cited |= set(
            re.findall(r"\[\[_Transcript#\^(p\d{4})\]\]", note.read_text("utf-8"))
        )
    assert cited
    assert cited <= block_ids


def test_no_note_carries_a_timestamp(pipeline) -> None:
    _run_everything()
    folder = config.OUT_DIR / "lec4"
    for note in folder.glob("[0-9]*.md"):
        body = note.read_text(encoding="utf-8").split("---", 2)[2]
        assert not re.search(r"\d\d:\d\d", body), note.name


def test_the_transcript_is_the_only_place_with_times(pipeline) -> None:
    _run_everything()
    transcript = (config.OUT_DIR / "lec4" / transcribe.TRANSCRIPT_NOTE).read_text(
        encoding="utf-8"
    )
    assert "**[00:00:00]**" in transcript
    assert "**[00:00:14]**" in transcript


def test_the_notes_are_linked_into_a_chain(pipeline) -> None:
    _run_everything()
    notes = sorted((config.OUT_DIR / "lec4").glob("[0-9]*.md"))
    assert len(notes) > 1
    first = notes[0].read_text(encoding="utf-8")
    last = notes[-1].read_text(encoding="utf-8")
    assert f"[[{notes[1].stem}]]" in first
    assert f"[[{notes[-2].stem}]]" in last


def test_the_glossary_merges_the_notes_terms(pipeline) -> None:
    _run_everything()
    glossary = (config.OUT_DIR / "lec4" / index.GLOSSARY_NOTE).read_text(
        encoding="utf-8"
    )
    assert "- **aliasing** — aliasing" in glossary
    assert "częstotliwość próbkowania" in glossary
    # One line per term however many notes defined it.
    assert glossary.count("**aliasing**") == 1


def test_the_map_lists_every_note(pipeline) -> None:
    _run_everything()
    folder = config.OUT_DIR / "lec4"
    map_text = (folder / index.MAP_NOTE).read_text(encoding="utf-8")
    for note in folder.glob("[0-9]*.md"):
        assert f"[[{note.stem}]]" in map_text


def test_running_everything_twice_changes_nothing(pipeline) -> None:
    """The container may be killed at any point; the next run resumes."""
    _run_everything()
    folder = config.OUT_DIR / "lec4"
    before = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }

    _run_everything()
    after = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }
    assert after == before


def test_a_second_run_does_not_ask_the_model_again(
    pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_everything()
    asked: list[str] = []

    def count(model: str, prompt: str) -> str:
        asked.append(prompt)
        return _model_answer(prompt)

    monkeypatch.setattr(ollama, "generate", count)
    _run_everything()
    assert asked == []


def test_every_generated_file_carries_the_marker(pipeline) -> None:
    _run_everything()
    for path in (config.OUT_DIR / "lec4").glob("*.md"):
        assert generate.is_ours(path), path.name


def test_the_note_frontmatter_records_the_prompt_and_the_model(pipeline) -> None:
    _run_everything()
    note = sorted((config.OUT_DIR / "lec4").glob("[0-9]*.md"))[0]
    text = note.read_text(encoding="utf-8")
    assert f"prompt: {config.PROMPT_VERSION}" in text
    assert f"model: {config.TEXT_MODEL}" in text
    assert 'source: "lecture-04.mp4"' in text
    assert "tags: [signal-processing, lec4]" in text


def test_the_transcript_records_the_asr_model_not_the_text_model(pipeline) -> None:
    _run_everything()
    text = (config.OUT_DIR / "lec4" / transcribe.TRANSCRIPT_NOTE).read_text(
        encoding="utf-8"
    )
    assert f"model: {config.ASR_MODEL}" in text
    assert 'duration: "00:07:00"' in text
    assert "language: pl" in text


def test_the_segments_stay_in_the_work_folder(pipeline) -> None:
    """The per-segment times the transcript does not keep."""
    _run_everything()
    payload = json.loads(transcribe.segments_path("lec4").read_text(encoding="utf-8"))
    assert len(payload["segments"]) == len(PARAGRAPHS)
    assert payload["asr"]["device"] == "cuda"


def test_a_window_the_model_fails_leaves_the_rest_of_the_run_intact(
    pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def fail_the_second(model: str, prompt: str) -> str:
        calls.append(1)
        if len(calls) == 2:
            return "## In short\nNo explanation section follows.\n"
        return _model_answer(prompt)

    monkeypatch.setattr(ollama, "generate", fail_the_second)
    _run_everything()

    folder = config.OUT_DIR / "lec4"
    assert (folder / index.MAP_NOTE).exists()
    conn = db.connect()
    source_id = db.source_id_for(conn, "lec4")
    rows = db.stage_rows(conn, source_id, "generate")
    assert sum(row["status"] == db.FAILED for row in rows) == 1
    assert sum(row["status"] == db.DONE for row in rows) == len(rows) - 1
    # The failed window is simply absent from the map; the chain skips it.
    map_text = (folder / index.MAP_NOTE).read_text(encoding="utf-8")
    assert map_text.count("- **") == len(rows) - 1
