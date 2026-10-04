# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""The map, the glossary and the footers. No model is involved in any of it."""

from __future__ import annotations

from pathlib import Path

from app import config, generate, index
from app.chunk import Window


def _window(order: int) -> Window:
    return Window(
        ord=order,
        start_s=float(order) * 500,
        end_s=float(order) * 500 + 499,
        labels=(f"p{order:04d}",),
        text_path=Path(f"/work/lec4/windows/{order:03d}.md"),
    )


def _note_text(order: int, glossary: str = "", short: str = "") -> str:
    return "\n".join(
        [
            "---",
            "generator: tachygraf",
            "prompt: 1",
            'source: "lecture-04.mp4"',
            "---",
            f"# {order:02d} A window",
            "",
            "## In short",
            short or f"The {order}th part of the lecture. A second sentence.",
            "",
            "## Glossary",
            glossary or "- aliasing — aliasing",
            "",
        ]
    )


def _written(count: int = 3, **kwargs) -> list[tuple[Window, Path]]:
    folder = generate.source_dir("lec4")
    generate.claim_dir(folder)
    notes = []
    for order in range(1, count + 1):
        path = folder / f"{order:02d} A window.md"
        path.write_text(_note_text(order, **kwargs), encoding="utf-8")
        notes.append((_window(order), path))
    return notes


# --- the glossary -----------------------------------------------------------


def test_parse_glossary_reads_the_pairs() -> None:
    text = _note_text(1, glossary="- emitter follower — wtórnik emiterowy\n- a — b")
    assert index.parse_glossary(text) == [
        ("emitter follower", "wtórnik emiterowy"),
        ("a", "b"),
    ]


def test_merge_glossary_keeps_sources_and_drops_duplicates() -> None:
    merged = index.merge_glossary(
        [
            ("01 One", ("aliasing", "aliasing")),
            ("02 Two", ("aliasing", "aliasing")),
            ("02 Two", ("sampling rate", "częstotliwość próbkowania")),
        ]
    )
    assert [entry[0] for entry in merged] == ["aliasing", "sampling rate"]
    assert merged[0][2] == ["01 One", "02 Two"]


def test_the_glossary_note_is_ours_and_links_back() -> None:
    text = index.glossary_note(
        index.merge_glossary([("01 One", ("aliasing", "aliasing"))]),
        source_name="lecture-04.mp4",
        slug="lec4",
        domain="signal processing",
    )
    assert generate.is_ours_text(text)
    assert "tags: [signal-processing, lec4, glossary]" in text
    assert "- **aliasing** — aliasing  [[01 One]]" in text


# --- the footers ------------------------------------------------------------


def test_apply_nav_adds_a_footer() -> None:
    text = index.apply_nav("# A note\n\nbody\n", "02 Previous", "04 Next")
    assert text.endswith("---\n[[02 Previous]] · [[04 Next]]\n")


def test_apply_nav_is_idempotent() -> None:
    once = index.apply_nav("# A note\n\nbody\n", "02 Previous", "04 Next")
    assert index.apply_nav(once, "02 Previous", "04 Next") == once


def test_apply_nav_replaces_a_stale_footer() -> None:
    once = index.apply_nav("# A note\n\nbody\n", "02 Old", None)
    twice = index.apply_nav(once, "02 New", "04 Next")
    assert "02 Old" not in twice
    assert "[[02 New]] · [[04 Next]]" in twice


def test_the_first_and_last_note_get_one_link_each() -> None:
    first = index.apply_nav("body\n", None, "02 Next")
    assert first.rstrip().endswith("[[02 Next]]")
    assert "·" not in first


# --- the map ----------------------------------------------------------------


def test_the_map_lists_every_note_with_its_span_and_first_sentence() -> None:
    notes = _written()
    text = index.map_note(
        notes,
        source_name="lecture-04.mp4",
        slug="lec4",
        title="Lecture 4",
        duration_s=1800.0,
        domain="signal processing",
    )
    assert generate.is_ours_text(text)
    assert "# Map — Lecture 4" in text
    assert 'duration: "00:30:00"' in text
    assert "- **00:08:20 - 00:16:39** [[01 A window]] — The 1th part" in text
    assert "A second sentence" not in text, "one sentence per line, not the whole note"
    assert "[[_Transcript]]" in text


def test_the_map_template_takes_exactly_the_fields_we_pass() -> None:
    text = index.map_note([], source_name="a.mp4", slug="s", title="T", duration_s=0.0)
    assert "{{" not in text


# --- the stage --------------------------------------------------------------


def _run(notes, **kwargs):
    return index.run(
        notes, source_name="lecture-04.mp4", slug="lec4", title="Lecture 4", **kwargs
    )


def test_run_links_the_notes_and_writes_both_index_files() -> None:
    notes = _written()
    written = _run(notes)

    folder = generate.source_dir("lec4")
    assert (folder / index.MAP_NOTE) in written
    assert (folder / index.GLOSSARY_NOTE) in written
    middle = notes[1][1].read_text(encoding="utf-8")
    assert "[[01 A window]] · [[03 A window]]" in middle


def test_run_is_rerunnable_without_changing_anything() -> None:
    notes = _written()
    _run(notes)
    before = {path: path.read_text(encoding="utf-8") for _window, path in notes}
    _run(notes)
    assert {path: path.read_text(encoding="utf-8") for _window, path in notes} == before


def test_run_leaves_a_foreign_file_at_a_note_path_alone() -> None:
    notes = _written(2)
    foreign = notes[1][1]
    foreign.write_text("someone else wrote this\n", encoding="utf-8")

    _run(notes)
    assert foreign.read_text(encoding="utf-8") == "someone else wrote this\n"
    # It takes no part in the map either.
    map_text = (generate.source_dir("lec4") / index.MAP_NOTE).read_text(
        encoding="utf-8"
    )
    assert "02 A window" not in map_text
    assert "01 A window" in map_text


def test_run_skips_a_window_whose_note_was_never_written() -> None:
    notes = _written(2)
    missing = (_window(3), generate.source_dir("lec4") / "03 Never written.md")
    written = _run([*notes, missing])
    assert missing[1] not in written


def test_run_writes_into_the_given_output_folder(tmp_path: Path) -> None:
    elsewhere = tmp_path / "another-staging-folder"
    elsewhere.mkdir()
    folder = generate.source_dir("lec4", elsewhere)
    generate.claim_dir(folder, elsewhere)
    path = folder / "01 A window.md"
    path.write_text(_note_text(1), encoding="utf-8")

    _run([(_window(1), path)], out_dir=elsewhere)
    assert (folder / index.MAP_NOTE).exists()
    assert not (config.OUT_DIR / "lec4").exists()
