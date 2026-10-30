# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""Shared fixtures: every test runs against a throwaway /work, /state, /out.

Copied from glosator's conftest. The autouse fixture is what keeps a test from
reaching the real mounts: config is the only module that reads the environment,
so redirecting it redirects everything.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app import config, transcribe

_NO_FFMPEG = shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip the tests marked ``ffmpeg`` when the binaries are not installed.

    Declaring the marker in one place keeps the skip reason out of every test
    that needs it, and keeps ``pytest -m ffmpeg`` meaning what it says.
    """
    if "ffmpeg" in item.keywords and _NO_FFMPEG:
        pytest.skip("ffmpeg and ffprobe are not installed")


@pytest.fixture(autouse=True)
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name, relative in (
        ("IN_DIR", "in"),
        ("WORK_DIR", "work"),
        ("STATE_DIR", "state"),
        ("OUT_DIR", "out"),
        ("LOG_DIR", "state/logs"),
    ):
        path = tmp_path / relative
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(config, name, path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "state" / "tachygraf.db")
    return tmp_path


@pytest.fixture
def asr_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the ASR stage in this process instead of spawning a child.

    The child exists to hand the card back (see docs/DECISIONS.md), which a
    test has no card to hand back. Tests of the spawn itself replace
    subprocess.Popen instead; everything else replaces the hop.
    """

    def inline(slug, audio, model_name, language, vocabulary=""):
        segments, reported = transcribe.transcribe_audio(
            audio, model_name, language, vocabulary
        )
        transcribe.write_segments(slug, segments, reported)
        return segments, reported

    monkeypatch.setattr(transcribe, "produce_segments", inline)
