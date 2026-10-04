# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""log.py is copied from glosator; these are its tests, for the file name."""

from __future__ import annotations

import json

from app import log


def test_event_writes_one_json_line_per_call() -> None:
    log.event("transcribe", "device chosen", device="cuda")
    log.event("generate", "note written", window=3)

    lines = log.log_path().read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["stage"] == "transcribe"
    assert first["device"] == "cuda"


def test_log_file_is_named_for_tachygraf() -> None:
    log.event("worker", "started")
    assert log.log_path().name.startswith("tachygraf-")


def test_tail_is_human_readable() -> None:
    log.event("chunk", "planned", windows=7)
    assert "[chunk] planned" in log.tail()
    assert '"windows": 7' in log.tail()


def test_tail_without_a_log_file() -> None:
    assert log.tail() == ""
