# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""A folder of recordings: what gets processed, what gets skipped, and why.

The hard part of a batch is not the loop. It is that a downloaded course holds
the same lecture twice, in two containers, beside files that are not
recordings at all. These tests are mostly about that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app import batch, config, db, media


class FakeProbe:
    """Stands in for ffprobe: durations and stream counts, by file name."""

    def __init__(self, table: dict[str, tuple[float, int]]):
        self.table = table

    def __call__(self, path: Path) -> media.MediaInfo:
        for name, (duration, videos) in self.table.items():
            if path.name == name:
                return media.MediaInfo(
                    path, duration, 1, "eng", "aac", video_streams=videos
                )
        raise media.MediaError(f"{path} has no audio stream")


def _touch(path: Path, size: int = 1000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


# --- ordering ---------------------------------------------------------------


def test_natural_order_puts_nine_before_ten() -> None:
    """A course is numbered, and a string sort gets it wrong."""
    names = ["lekcja-10.mp4", "lekcja-9.mp4", "lekcja-2.mp4"]
    assert sorted(names, key=batch.natural_key) == [
        "lekcja-2.mp4",
        "lekcja-9.mp4",
        "lekcja-10.mp4",
    ]


def test_natural_order_mixes_numbered_and_unnumbered_names() -> None:
    """The real course folder, which crashed the first version with TypeError.

    "120V ..." starts with a number and "AC Basics ..." does not, and a key
    that puts a bare int beside a bare str cannot compare the two.
    """
    names = [
        "AC Basics; Learn All About Alternating Current.m4a",
        "120V 240V Electricity explained.m4a",
        "3 Way Switches Explained.m4a",
    ]
    assert sorted(names, key=batch.natural_key) == [
        "3 Way Switches Explained.m4a",
        "120V 240V Electricity explained.m4a",
        "AC Basics; Learn All About Alternating Current.m4a",
    ]


def test_find_media_walks_subfolders_in_reading_order(tmp_path: Path) -> None:
    """A downloaded playlist is often one folder per recording."""
    for name in ("10 - Ten/a.mp4", "2 - Two/a.mp4", "1 - One/a.m4a"):
        _touch(tmp_path / name)
    found = batch.find_media(tmp_path)
    assert [path.parent.name for path in found] == ["1 - One", "2 - Two", "10 - Ten"]


def test_find_media_ignores_what_is_not_a_recording(tmp_path: Path) -> None:
    for name in ("a.mp4", "a.srt", "a.jpg", "a.txt", "a.nfo", "a.vtt"):
        _touch(tmp_path / name)
    assert [path.name for path in batch.find_media(tmp_path)] == ["a.mp4"]


def test_find_media_leaves_hidden_directories_alone(tmp_path: Path) -> None:
    _touch(tmp_path / "keep.mp4")
    _touch(tmp_path / ".trash" / "deleted.mp4")
    assert [path.name for path in batch.find_media(tmp_path)] == ["keep.mp4"]


def test_no_recurse_looks_only_at_the_folder_itself(tmp_path: Path) -> None:
    _touch(tmp_path / "top.mp4")
    _touch(tmp_path / "sub" / "deep.mp4")
    assert [p.name for p in batch.find_media(tmp_path, recurse=False)] == ["top.mp4"]


def test_a_missing_folder_is_reported(tmp_path: Path) -> None:
    with pytest.raises(media.MediaError, match="not a folder"):
        batch.find_media(tmp_path / "nowhere")


# --- titles -----------------------------------------------------------------


def test_a_downloaders_format_tag_is_stripped() -> None:
    assert (
        batch.clean_title("Diodes Explained (2160p_25fps_AV1-128kbit_AAC-English)")
        == "Diodes Explained"
    )
    assert (
        batch.clean_title("Transistors Explained (1080p_30fps_H264-128kbit_AAC)")
        == "Transistors Explained"
    )


def test_underscores_inside_a_format_tag_do_not_hide_it() -> None:
    """An underscore is a word character, so \\b never matched inside one."""
    assert batch.clean_title("Lecture (720p_AAC)") == "Lecture"


def test_a_bracket_that_is_part_of_the_title_is_kept() -> None:
    assert batch.clean_title("Rectifiers (part 2) (720p_AAC-English)") == (
        "Rectifiers (part 2)"
    )
    assert batch.clean_title("Diodes (continued)") == "Diodes (continued)"


def test_a_title_that_is_only_a_format_tag_keeps_the_stem() -> None:
    assert batch.clean_title("(1080p_AAC)") == "(1080p_AAC)"


def test_a_long_title_gives_a_slug_cut_at_a_hyphen() -> None:
    """A slug is a folder name, a tag, and part of every wikilink."""
    slug = batch.short_slug(
        "Diodes Explained - The basics how diodes work working principle pn junction"
    )
    assert len(slug) <= batch.SLUG_MAX_CHARS
    assert not slug.endswith("-")
    assert slug.startswith("diodes-explained")


# --- the plan ---------------------------------------------------------------


def test_the_same_lecture_in_two_containers_counts_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure this whole module exists for: a course done twice."""
    video = _touch(tmp_path / "L1" / "Lecture One (2160p_AV1-AAC).mp4", 160_000_000)
    audio = _touch(tmp_path / "L1" / "Lecture One (128kbit_AAC).m4a", 11_000_000)
    monkeypatch.setattr(
        media,
        "probe",
        FakeProbe({video.name: (692.0, 1), audio.name: (692.4, 0)}),
    )

    built = batch.plan(tmp_path)
    assert len(built.recordings) == 1
    chosen = built.recordings[0]
    assert chosen.path.name == audio.name, "the audio-only file is preferred"
    assert chosen.duplicates == (video,)
    assert any("the same" in item.reason for item in built.skipped)


def test_two_different_lectures_of_the_same_length_stay_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A course cut to a slot has many recordings of the same length."""
    first = _touch(tmp_path / "L1" / "One.m4a")
    second = _touch(tmp_path / "L2" / "Two.m4a")
    monkeypatch.setattr(
        media, "probe", FakeProbe({first.name: (600.0, 0), second.name: (600.0, 0)})
    )
    assert len(batch.plan(tmp_path).recordings) == 2


def test_a_flat_course_of_equal_length_episodes_is_not_merged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first version of the duplicate test collapsed exactly this folder.

    Same directory, same duration, different lectures: a course cut to a slot
    and downloaded into one folder.
    """
    names = ["1 - Diodes.m4a", "2 - Transistors.m4a", "3 - Rectifiers.m4a"]
    for name in names:
        _touch(tmp_path / name)
    monkeypatch.setattr(media, "probe", FakeProbe({name: (600.0, 0) for name in names}))
    assert len(batch.plan(tmp_path).recordings) == 3


def test_a_lecture_and_its_trailer_are_not_merged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same title, same folder, very different length."""
    full = _touch(tmp_path / "Diodes (1080p_AAC).mp4")
    trailer = _touch(tmp_path / "Diodes (720p_AAC).m4a")
    monkeypatch.setattr(
        media, "probe", FakeProbe({full.name: (692.0, 1), trailer.name: (31.0, 0)})
    )
    assert len(batch.plan(tmp_path).recordings) == 2


def test_a_file_with_no_audio_is_skipped_with_its_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One bad file in a course of twenty must not stop the other nineteen."""
    good = _touch(tmp_path / "good.m4a")
    _touch(tmp_path / "thumbnail.mp4")
    monkeypatch.setattr(media, "probe", FakeProbe({good.name: (600.0, 0)}))

    built = batch.plan(tmp_path)
    assert [item.path.name for item in built.recordings] == ["good.m4a"]
    assert "no audio stream" in built.skipped[0].reason


def test_numbering_keeps_a_course_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("1 - Diodes.m4a", "2 - Transistors.m4a", "10 - Rectifiers.m4a"):
        _touch(tmp_path / name)
    monkeypatch.setattr(
        media,
        "probe",
        FakeProbe(
            {
                "1 - Diodes.m4a": (600.0, 0),
                "2 - Transistors.m4a": (600.0, 0),
                "10 - Rectifiers.m4a": (600.0, 0),
            }
        ),
    )
    slugs = [item.slug for item in batch.plan(tmp_path, number=True).recordings]
    assert slugs == ["01-1-diodes", "02-2-transistors", "03-10-rectifiers"]


def test_a_prefix_namespaces_the_course(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _touch(tmp_path / "Diodes.m4a")
    monkeypatch.setattr(media, "probe", FakeProbe({"Diodes.m4a": (600.0, 0)}))
    built = batch.plan(tmp_path, slug_prefix="kurs elektroniki")
    assert built.recordings[0].slug == "kurs-elektroniki-diodes"


def test_two_recordings_with_the_same_title_get_distinct_slugs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _touch(tmp_path / "a" / "Diodes.m4a")
    second = _touch(tmp_path / "b" / "Diodes.m4a")
    monkeypatch.setattr(media, "probe", FakeProbe({"Diodes.m4a": (600.0, 0)}))
    slugs = [item.slug for item in batch.plan(tmp_path).recordings]
    assert slugs == ["diodes", "diodes-2"]
    assert first.parent != second.parent


def test_an_empty_folder_plans_nothing(tmp_path: Path) -> None:
    built = batch.plan(tmp_path)
    assert built.recordings == ()
    assert batch._estimate(built) == ""


# --- queueing ---------------------------------------------------------------


def test_queue_registers_each_recording_and_all_five_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("1 - One.m4a", "2 - Two.m4a"):
        _touch(tmp_path / name)
    monkeypatch.setattr(
        media,
        "probe",
        FakeProbe({"1 - One.m4a": (600.0, 0), "2 - Two.m4a": (600.0, 0)}),
    )
    conn = db.connect()
    built = batch.plan(tmp_path)
    jobs = batch.queue(conn, built, domain="electronics", out_dir=config.OUT_DIR)

    assert len(jobs) == 10
    assert len(db.list_sources(conn)) == 2
    assert {row["domain"] for row in db.list_sources(conn)} == {"electronics"}

    queued = list(reversed(db.recent_jobs(conn)))
    stages = [job["stage"] for job in queued]
    # Grouped per recording: the first lecture's notes before the second
    # lecture's transcription.
    assert stages == [
        "extract",
        "transcribe",
        "chunk",
        "generate",
        "index",
        "extract",
        "transcribe",
        "chunk",
        "generate",
        "index",
    ]


def test_queue_passes_the_models_and_the_language_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    _touch(tmp_path / "One.m4a")
    monkeypatch.setattr(media, "probe", FakeProbe({"One.m4a": (600.0, 0)}))
    conn = db.connect()
    batch.queue(
        conn,
        batch.plan(tmp_path),
        language="pl",
        vocabulary="Claude Code, Anthropic",
        asr_model="large-v3-turbo",
        text_model="gemma3:12b-it-qat",
    )
    jobs = {job["stage"]: job for job in db.recent_jobs(conn)}
    assert jobs["transcribe"]["model"] == "large-v3-turbo"
    params = json.loads(jobs["transcribe"]["params"])
    assert params["language"] == "pl"
    assert params["vocabulary"] == "Claude Code, Anthropic"
    assert jobs["generate"]["model"] == "gemma3:12b-it-qat"


def test_describe_names_the_file_it_chose_and_the_one_it_did_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A choice made on someone's behalf has to be visible to them."""
    video = _touch(tmp_path / "L1" / "One (2160p_AV1-AAC).mp4", 160_000_000)
    audio = _touch(tmp_path / "L1" / "One (128kbit_AAC).m4a", 11_000_000)
    monkeypatch.setattr(
        media, "probe", FakeProbe({video.name: (692.0, 1), audio.name: (692.0, 0)})
    )
    text = batch.describe(batch.plan(tmp_path))
    assert "1 recording(s)" in text
    assert "One (128kbit_AAC).m4a" in text
    assert "1 file(s) skipped" in text
    assert "the same 00:11:32 recording as" in text
    assert "10 min of generation" in text
