# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""Turn a transcript into an ordered plan of windows.

A window is one note. glosator cuts a book at its section boundaries, which a
recording does not have: a lecture is one unbroken stretch of speech. What it
has instead is pauses, so a window ends where the speaker stopped talking and
the window is long enough to be worth a note.

Two consequences follow, and both are deliberate:

- A window is never cut mid-sentence, because the only places it may be cut
  are pauses of at least ``PAUSE_SPLIT_S``.
- Each window repeats the last paragraph of the one before it. A claim the
  speaker began before the break and finished after it is then whole in the
  second window, and one paragraph of overlap costs a few hundred tokens.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app import config, generate, log, transcribe
from app.transcribe import Paragraph

CHUNKS_JSON = "chunks.json"
WINDOW_DIR = "windows"


@dataclass(frozen=True)
class Window:
    """One note's worth of the recording."""

    ord: int
    start_s: float
    end_s: float
    labels: tuple[str, ...]
    text_path: Path


def work_dir(slug: str) -> Path:
    return config.WORK_DIR / slug


def window_text(group: Sequence[Paragraph]) -> str:
    """The text handed to the model: one labelled paragraph per line.

    The label goes in front, where the prompt says to look for it, and the
    paragraph's time does not go in at all. The model cannot write a timestamp
    it was never shown, which is the cheapest way to enforce that rule.
    """
    return "\n".join(f"[{paragraph.label}] {paragraph.text}" for paragraph in group)


def group_paragraphs(built: Sequence[Paragraph]) -> list[list[Paragraph]]:
    """Cut the paragraphs into windows at pauses, within the token budget.

    A window closes when it has reached ``CHUNK_MIN_TOKENS`` and the speaker
    pauses for at least ``PAUSE_SPLIT_S``, or unconditionally at
    ``CHUNK_MAX_TOKENS``. Without the second condition a lecture delivered
    without a real break would be one window the model cannot read.
    """
    groups: list[list[Paragraph]] = []
    current: list[Paragraph] = []
    tokens = 0
    carried = False

    for position, paragraph in enumerate(built):
        current.append(paragraph)
        tokens += generate.estimate_tokens(paragraph.text)
        following = built[position + 1] if position + 1 < len(built) else None
        if following is None:
            break
        gap = following.start - paragraph.end
        at_pause = tokens >= config.CHUNK_MIN_TOKENS and gap >= config.PAUSE_SPLIT_S
        if at_pause or tokens >= config.CHUNK_MAX_TOKENS:
            groups.append(current)
            # One paragraph of overlap: the window that closed ends with this
            # paragraph and the next one begins with it.
            current = [paragraph]
            tokens = generate.estimate_tokens(paragraph.text)
            carried = True

    if current and not (carried and len(current) == 1 and groups):
        # A trailing window holding nothing but the carried paragraph would be
        # a note about text the previous note already covers.
        groups.append(current)
    return groups


def plan(slug: str) -> list[Window]:
    """Build the window plan and write one text file per window."""
    segments, _reported = transcribe.load_segments(slug)
    built = transcribe.paragraphs(segments)
    if not built:
        raise ValueError(f"{slug}: the transcript has no paragraphs to window")

    groups = group_paragraphs(built)
    window_dir = work_dir(slug) / WINDOW_DIR
    window_dir.mkdir(parents=True, exist_ok=True)

    windows: list[Window] = []
    for order, group in enumerate(groups, start=1):
        text_path = window_dir / f"{order:03d}.md"
        text_path.write_text(window_text(group) + "\n", encoding="utf-8")
        windows.append(
            Window(
                ord=order,
                start_s=group[0].start,
                end_s=group[-1].end,
                labels=tuple(paragraph.label for paragraph in group),
                text_path=text_path,
            )
        )

    _write_plan(slug, windows)
    log.event(
        "chunk",
        "planned",
        slug=slug,
        windows=len(windows),
        paragraphs=len(built),
        minutes=round((built[-1].end - built[0].start) / 60, 1),
    )
    for window in oversized(windows):
        log.event(
            "chunk",
            "window over budget; the generate stage will refuse it",
            slug=slug,
            window=window.ord,
            tokens=generate.estimate_tokens(
                window.text_path.read_text(encoding="utf-8")
            ),
            budget=config.CHUNK_MAX_TOKENS,
        )
    return windows


def oversized(windows: Sequence[Window]) -> list[Window]:
    """Windows whose text does not fit the model's context window.

    Only reachable through a single paragraph longer than the whole budget,
    which means ``PARAGRAPH_MAX_S`` was raised past what the model can read.
    """
    return [
        window
        for window in windows
        if generate.estimate_tokens(window.text_path.read_text(encoding="utf-8"))
        > config.CHUNK_MAX_TOKENS
    ]


def _write_plan(slug: str, windows: Sequence[Window]) -> None:
    path = work_dir(slug) / CHUNKS_JSON
    records = [
        {
            "ord": window.ord,
            "start_s": window.start_s,
            "end_s": window.end_s,
            "labels": list(window.labels),
            "text_path": str(window.text_path),
        }
        for window in windows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    generate.write_atomic(
        path, json.dumps(records, ensure_ascii=False, indent=2) + "\n"
    )


def load_plan(slug: str) -> list[Window]:
    path = work_dir(slug) / CHUNKS_JSON
    if not path.exists():
        raise FileNotFoundError(
            f"no window plan at {path}; run the chunk stage for {slug} first"
        )
    records = json.loads(path.read_text(encoding="utf-8"))
    return [
        Window(
            ord=record["ord"],
            start_s=float(record["start_s"]),
            end_s=float(record["end_s"]),
            labels=tuple(record["labels"]),
            text_path=Path(record["text_path"]),
        )
        for record in records
    ]
