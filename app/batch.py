# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""A folder of recordings in, one set of notes per recording out.

A course is not one long recording, it is twenty of them, and the thing that
makes a folder hard is not the loop. It is what a downloader actually leaves
on disk: the same lecture as a 4K video and as an audio-only track, beside a
subtitle file, a thumbnail and a description. Adding every media file it finds
would transcribe every lecture twice and generate two sets of notes from the
same words, which for a twenty-part course is a day of the card spent on
duplicates.

So this module does three things before anything is queued:

- Finds the recordings, in the order a person would read them, which is not
  the order a string sort gives: "lekcja-9" comes before "lekcja-10".
- Collapses each recording to one file. Two files in the same folder with the
  same duration are the same lecture, and the audio-only one is preferred:
  identical audio, a tenth of the bytes, and no video stream for ffmpeg to
  walk past.
- Names each one. A downloader's title carries the format in brackets, and
  "(2160p_25fps_AV1-128kbit_AAC-English)" is not part of what the lecture is
  called.

Nothing here touches a model or writes to the output folder. It reads
durations with ffprobe, which costs milliseconds per file, and returns a plan
that the caller queues or prints.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from app import config, db, log, media

# Containers a lecture arrives in. Deliberately the same list the interface
# offers, and deliberately not "whatever ffmpeg decodes": a folder of
# recordings may also hold a thumbnail and a description, and guessing from
# ffprobe alone would mean probing every file in the tree.
MEDIA_SUFFIXES = frozenset(
    {
        ".mp4",
        ".mkv",
        ".webm",
        ".mov",
        ".avi",
        ".m4a",
        ".m4b",
        ".mp3",
        ".wav",
        ".flac",
        ".ogg",
        ".oga",
        ".opus",
        ".aac",
        ".wma",
    }
)

# Trailing bracketed text a downloader appends: resolution, frame rate, codec,
# bit rate, the audio language. Stripped from the title, because it describes
# the file and not the lecture. A bracket that says something else -- "(part
# 2)", "(continued)" -- is kept, which is why this looks for format tokens
# rather than for brackets.
#
# The contents are split on underscores and hyphens before being matched:
# "1080p_30fps_H264-128kbit_AAC" is one word to a word-boundary pattern,
# because an underscore is a word character, and the first version of this
# matched none of it.
FORMAT_TOKEN_RE = re.compile(
    r"\b(?:\d{3,4}p|\d+fps|\d+kbit|\d+kbps|av01|av1|avc1?|h\.?26[45]|hevc|vp0?9"
    r"|aac|opus|mp4a|flac|ac3|eac3|dash|webm|hdr|sdr|stereo|mono"
    r"|english|polish|audio|video|only)\b",
    re.IGNORECASE,
)
TRAILING_BRACKET_RE = re.compile(r"\s*[\(\[]([^()\[\]]*)[\)\]]\s*$")

# A slug becomes a folder name under the staging folder, a tag in every note
# and part of every wikilink. A downloader's title is routinely seventy
# characters of search-engine prose, so it is cut at a hyphen.
SLUG_MAX_CHARS = 48

# Two files of one recording differ in duration by a frame or two of container
# padding, never by a second. Used together with the title, never alone: see
# _group_by_recording.
SAME_RECORDING_S = 1.5


@dataclass(frozen=True)
class Planned:
    """One recording the batch will process, and what it stood in for."""

    path: Path
    slug: str
    title: str
    duration_s: float
    #: Files that are the same recording in another container, not processed.
    duplicates: tuple[Path, ...] = ()

    @property
    def duration_hms(self) -> str:
        return media.hms(self.duration_s)


@dataclass(frozen=True)
class Skipped:
    """A file the batch will not process, and the reason a person needs."""

    path: Path
    reason: str


@dataclass(frozen=True)
class Plan:
    """What a batch would do. Printed by --dry-run, queued otherwise."""

    folder: Path
    recordings: tuple[Planned, ...]
    skipped: tuple[Skipped, ...]

    @property
    def total_s(self) -> float:
        return sum(item.duration_s for item in self.recordings)


def natural_key(text: str) -> tuple:
    """Sort key that orders "9" before "10".

    A course is numbered, and a string sort puts lekcja-10 before lekcja-9,
    which then becomes the order the notes are generated in and the order they
    appear in every listing.
    """
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", text)
        if part != ""
    )


def find_media(folder: Path, recurse: bool = True) -> list[Path]:
    """Media files under ``folder``, in reading order.

    Recursive by default, because the layout a downloader produces for a
    playlist is often one folder per video rather than one folder of videos.
    Hidden directories are left alone.
    """
    if not folder.is_dir():
        raise media.MediaError(f"not a folder: {folder}")
    walk = folder.rglob("*") if recurse else folder.iterdir()
    found = [
        path
        for path in walk
        if path.is_file()
        and path.suffix.lower() in MEDIA_SUFFIXES
        and not any(part.startswith(".") for part in path.relative_to(folder).parts)
    ]
    return sorted(found, key=lambda path: natural_key(str(path.relative_to(folder))))


def clean_title(stem: str) -> str:
    """A file stem as a lecture title, without the downloader's format tags.

    Only a trailing bracket whose contents look like format metadata is
    removed, so "(part 2)" survives and "(1080p_AAC-English)" does not.
    Repeated, because a downloader may have appended more than one.
    """
    title = stem.strip()
    while True:
        match = TRAILING_BRACKET_RE.search(title)
        if not match:
            break
        inside = re.sub(r"[_\-+.]+", " ", match.group(1))
        if not FORMAT_TOKEN_RE.search(inside):
            break
        title = title[: match.start()].rstrip()
    # Separators a downloader uses where a person would use a space.
    title = re.sub(r"[_\s]+", " ", title).strip(" -–—_")  # noqa: RUF001
    return title or stem


def short_slug(text: str) -> str:
    """A slug for a folder name, cut at a hyphen rather than mid-word."""
    slug = media.slugify(text)
    if len(slug) <= SLUG_MAX_CHARS:
        return slug
    head = slug[:SLUG_MAX_CHARS].rsplit("-", 1)[0]
    return (head or slug[:SLUG_MAX_CHARS]).strip("-")


def _describe(path: Path, folder: Path) -> str:
    try:
        return str(path.relative_to(folder))
    except ValueError:
        return str(path)


def plan(
    folder: Path,
    *,
    recurse: bool = True,
    slug_prefix: str = "",
    number: bool = False,
) -> Plan:
    """Decide what a batch over ``folder`` would process, and what it would not.

    Probes every candidate, which is the only way to know a file has audio and
    the only way to tell two containers of one lecture apart. Reads nothing
    else and writes nothing.
    """
    folder = folder.expanduser().resolve()
    candidates = find_media(folder, recurse)
    skipped: list[Skipped] = []

    # (path, info) for everything that is usable at all.
    usable: list[tuple[Path, media.MediaInfo]] = []
    for path in candidates:
        try:
            usable.append((path, media.probe(path)))
        except media.MediaError as error:
            # A thumbnail with a video extension, a truncated download, a file
            # with no audio track. Named and skipped, never fatal: one bad
            # file in a course of twenty must not stop the other nineteen.
            skipped.append(Skipped(path, str(error)))

    groups = _group_by_recording(usable)
    recordings: list[Planned] = []
    used: set[str] = set()
    for position, group in enumerate(groups, start=1):
        chosen, info = _preferred(group)
        others = tuple(path for path, _ in group if path != chosen)
        for other in others:
            skipped.append(
                Skipped(
                    other,
                    f"the same {info.duration_hms} recording as "
                    f"{_describe(chosen, folder)}",
                )
            )
        title = clean_title(chosen.stem)
        stem = short_slug(title)
        if number:
            stem = f"{position:02d}-{stem}"
        if slug_prefix:
            stem = f"{media.slugify(slug_prefix)}-{stem}"
        recordings.append(
            Planned(
                path=chosen,
                slug=_unique(stem, used),
                title=title,
                duration_s=info.duration_s,
                duplicates=others,
            )
        )

    built = Plan(folder=folder, recordings=tuple(recordings), skipped=tuple(skipped))
    log.event(
        "batch",
        "planned",
        folder=str(folder),
        recordings=len(built.recordings),
        skipped=len(built.skipped),
        duration=media.hms(built.total_s),
    )
    return built


def _group_by_recording(
    usable: list[tuple[Path, media.MediaInfo]],
) -> list[list[tuple[Path, media.MediaInfo]]]:
    """Collapse the same lecture in several containers into one group.

    Three things have to agree: the parent directory, the title once the
    format tag is stripped, and the duration. All three, because any two of
    them give a false positive on real data:

    - Duration and directory alone merge two different lectures, which is not
      an edge case. A flat course folder holds every episode side by side, and
      a course cut to a slot holds many of exactly the same length. The first
      version of this did that and merged a whole folder into one recording.
    - Title and directory alone would merge a lecture with its own trailer.

    Being wrong in this direction costs a duplicated recording, which the
    plan shows and the user can see. Being wrong in the other direction drops
    a lecture silently, which is worse, so the test is deliberately strict.
    """
    groups: list[list[tuple[Path, media.MediaInfo]]] = []
    for path, info in usable:
        title = clean_title(path.stem).casefold()
        for group in groups:
            head_path, head_info = group[0]
            if (
                head_path.parent == path.parent
                and clean_title(head_path.stem).casefold() == title
                and abs(head_info.duration_s - info.duration_s) <= SAME_RECORDING_S
            ):
                group.append((path, info))
                break
        else:
            groups.append([(path, info)])
    return groups


def _preferred(
    group: list[tuple[Path, media.MediaInfo]],
) -> tuple[Path, media.MediaInfo]:
    """The file to transcribe out of several of the same recording.

    Audio-only first: the audio is identical, the file is a tenth of the size,
    and the extract stage does not have to decode past a video stream to reach
    it. Smaller file next, as a tie-break that prefers the plainer encoding.
    """
    return min(
        group,
        key=lambda item: (
            not item[1].audio_only,
            item[0].stat().st_size,
            str(item[0]),
        ),
    )


def _unique(slug: str, used: set[str]) -> str:
    """Copied in shape from glosator's chunk._unique."""
    candidate = slug
    suffix = 2
    while candidate in used:
        candidate = f"{slug}-{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def describe(built: Plan) -> str:
    """The plan as a person reads it, for --dry-run and the interface.

    Printed before anything is queued because the batch chooses files on the
    user's behalf, and a choice made on someone's behalf has to be visible.
    """
    lines = [
        f"{len(built.recordings)} recording(s) in {built.folder}, "
        f"{media.hms(built.total_s)} in total:",
        "",
    ]
    for item in built.recordings:
        lines.append(f"  {item.duration_hms}  {item.slug}")
        lines.append(f"             {_describe(item.path, built.folder)}")
    prefix = f"{built.folder}/"
    if built.skipped:
        lines += ["", f"{len(built.skipped)} file(s) skipped:", ""]
        for item in built.skipped:
            lines.append(f"  {_describe(item.path, built.folder)}")
            # ffprobe's own message carries the absolute path; the listing is
            # already relative to the folder, so say it once.
            lines.append(f"             {item.reason.replace(prefix, '')}")
    estimate = _estimate(built)
    if estimate:
        lines += ["", estimate]
    return "\n".join(lines)


def _estimate(built: Plan) -> str:
    """A rough idea of how long the queue will take, from PILOT.md.

    An estimate, and said to be one. The measured figures are ten minutes per
    window of generation and roughly one window per twelve minutes of speech,
    so this is those two multiplied and nothing more.
    """
    if not built.recordings:
        return ""
    windows = max(len(built.recordings), round(built.total_s / 60 / 12))
    minutes = windows * 10
    span = f"{minutes / 60:.1f} h" if minutes >= 60 else f"{minutes} min"
    return (
        f"Roughly {windows} window(s), so on the order of {span} of generation "
        f"at the measured ten minutes each (docs/PILOT.md). Transcription is "
        "minutes; the notes are the hours."
    )


def queue(
    conn: sqlite3.Connection,
    built: Plan,
    *,
    domain: str = config.DEFAULT_DOMAIN,
    out_dir: Path | None = None,
    language: str = config.LANGUAGE,
    asr_model: str = config.ASR_MODEL,
    text_model: str = config.TEXT_MODEL,
) -> list[int]:
    """Register every recording and queue all five stages for each.

    Grouped by recording rather than by stage: the first lecture's notes are
    finished before the second lecture's transcription starts. A complete note
    is the product and a transcript is an intermediate, so a batch that is
    stopped halfway should leave finished lectures behind rather than twenty
    transcripts and nothing to read.

    The cost of that order is that the text model is unloaded and reloaded
    once per recording, because the card is handed back when a job ends. See
    docs/DECISIONS.md.
    """
    jobs: list[int] = []
    for item in built.recordings:
        source_id = db.upsert_source(
            conn, item.slug, item.title, item.path, out_dir, domain
        )
        jobs += [
            db.enqueue(conn, source_id, "extract", "none", json.dumps({})),
            db.enqueue(
                conn,
                source_id,
                "transcribe",
                asr_model,
                json.dumps({"language": language}),
            ),
            db.enqueue(conn, source_id, "chunk", "none", json.dumps({})),
            db.enqueue(conn, source_id, "generate", text_model, json.dumps({})),
            db.enqueue(conn, source_id, "index", "none", json.dumps({})),
        ]
    log.event(
        "batch",
        "queued",
        folder=str(built.folder),
        recordings=len(built.recordings),
        jobs=len(jobs),
        domain=domain,
    )
    return jobs
