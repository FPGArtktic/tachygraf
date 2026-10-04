# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""Windows: where a note begins and ends, and why there.

The budget numbers are shrunk in these tests so that a window boundary can be
reached with a handful of sentences instead of three thousand tokens of them.
"""

from __future__ import annotations

import itertools

import pytest

from app import chunk, config, transcribe
from app.transcribe import Paragraph, Segment


@pytest.fixture(autouse=True)
def small_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "CHUNK_MIN_TOKENS", 5)
    monkeypatch.setattr(config, "CHUNK_MAX_TOKENS", 20)
    monkeypatch.setattr(config, "PAUSE_SPLIT_S", 2.5)


def _paragraph(index: int, start: float, end: float, words: int = 4) -> Paragraph:
    return Paragraph(
        label=transcribe.label_for(index),
        start=start,
        end=end,
        text=" ".join(f"word{index}" for _ in range(words)),
    )


def test_a_window_is_cut_at_a_pause_once_it_is_long_enough() -> None:
    built = [
        _paragraph(1, 0.0, 10.0),
        _paragraph(2, 10.5, 20.0),
        # A four-second pause here: long enough to end a window.
        _paragraph(3, 24.0, 34.0),
    ]
    groups = chunk.group_paragraphs(built)
    assert [[p.label for p in group] for group in groups] == [
        ["p0001", "p0002"],
        ["p0002", "p0003"],
    ]


def test_a_short_pause_does_not_cut_a_window() -> None:
    """A note must not begin mid-sentence, so only a real pause closes one."""
    built = [
        _paragraph(1, 0.0, 10.0),
        _paragraph(2, 10.5, 20.0),
        _paragraph(3, 20.3, 30.0),
    ]
    assert len(chunk.group_paragraphs(built)) == 1


def test_a_window_closes_at_the_budget_even_without_a_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lecture delivered without a break would be one unreadable window."""
    monkeypatch.setattr(config, "CHUNK_MAX_TOKENS", 10)
    built = [
        _paragraph(index, float(index) * 10, float(index) * 10 + 9.8, words=8)
        for index in range(1, 7)
    ]
    groups = chunk.group_paragraphs(built)
    assert len(groups) > 1


def test_each_window_repeats_the_last_paragraph_of_the_one_before() -> None:
    """A claim begun before the break and finished after it stays whole."""
    built = [
        _paragraph(1, 0.0, 10.0),
        _paragraph(2, 10.5, 20.0),
        _paragraph(3, 24.0, 34.0),
        _paragraph(4, 34.5, 44.0),
        _paragraph(5, 48.0, 58.0),
    ]
    groups = chunk.group_paragraphs(built)
    for earlier, later in itertools.pairwise(groups):
        assert earlier[-1] == later[0]


def test_a_trailing_window_of_nothing_but_the_overlap_is_not_made() -> None:
    """It would be a note about text the previous note already covers."""
    built = [
        _paragraph(1, 0.0, 10.0),
        _paragraph(2, 10.5, 20.0),
        _paragraph(3, 24.0, 34.0),
    ]
    groups = chunk.group_paragraphs(built)
    assert all(len(group) > 1 for group in groups)


def test_one_paragraph_is_one_window() -> None:
    groups = chunk.group_paragraphs([_paragraph(1, 0.0, 10.0)])
    assert [[p.label for p in group] for group in groups] == [["p0001"]]


def test_no_paragraphs_means_no_windows() -> None:
    assert chunk.group_paragraphs([]) == []


def test_the_window_text_carries_labels_and_no_times() -> None:
    text = chunk.window_text([_paragraph(1, 760.0, 800.0), _paragraph(2, 801.0, 900.0)])
    assert text.splitlines()[0].startswith("[p0001] ")
    assert len(text.splitlines()) == 2
    assert "00:12:40" not in text
    assert "760" not in text


# --- the stage --------------------------------------------------------------


def _segments_on_disk(slug: str = "lec4") -> None:
    segments = [
        Segment(0.0, 10.0, "Pierwszy fragment o próbkowaniu.", -0.2),
        Segment(10.5, 20.0, "Drugi fragment o aliasingu.", -0.2),
        Segment(24.0, 34.0, "Trzeci fragment po przerwie.", -0.2),
        Segment(34.5, 44.0, "Czwarty fragment.", -0.2),
    ]
    transcribe.write_segments(slug, segments, {"language": "pl"})


def test_plan_writes_a_text_file_per_window_and_the_plan_json() -> None:
    _segments_on_disk()
    windows = chunk.plan("lec4")

    assert windows
    for window in windows:
        assert window.text_path.exists()
        assert window.labels
        assert window.text_path.read_text(encoding="utf-8").startswith("[p")
    assert (chunk.work_dir("lec4") / chunk.CHUNKS_JSON).exists()


def test_plan_round_trips_through_load_plan() -> None:
    _segments_on_disk()
    planned = chunk.plan("lec4")
    assert chunk.load_plan("lec4") == planned


def test_a_windows_span_covers_its_paragraphs() -> None:
    _segments_on_disk()
    window = chunk.plan("lec4")[0]
    assert window.start_s == 0.0
    assert window.end_s >= 20.0


def test_load_plan_without_a_plan_says_which_stage_is_missing() -> None:
    with pytest.raises(FileNotFoundError, match="chunk stage"):
        chunk.load_plan("lec4")


def test_plan_without_a_transcript_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    transcribe.write_segments("lec4", [], {})
    with pytest.raises(ValueError, match="no paragraphs"):
        chunk.plan("lec4")


def test_an_oversized_window_is_reported_rather_than_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One paragraph over the whole budget means PARAGRAPH_MAX_S is too large."""
    monkeypatch.setattr(config, "CHUNK_MAX_TOKENS", 2)
    transcribe.write_segments(
        "lec4", [Segment(0.0, 60.0, "word " * 200, -0.2)], {"language": "pl"}
    )
    windows = chunk.plan("lec4")
    assert chunk.oversized(windows) == windows
