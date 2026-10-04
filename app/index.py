# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""The map of content, the merged glossary and previous/next links.

Copied from glosator's index.py and adapted: a recording has no chapters, so
there is one map per recording rather than one per chapter, and the map lines
carry a span of the recording instead of a page range.

Deterministic, and no model is involved. That is what makes this stage safe to
re-run after any regeneration: it reads the notes that exist, rewrites their
footers and rebuilds two files, and it changes nothing else in the output
folder. A file at a note's path that carries no generator marker takes no part
and is counted, not rewritten.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from app import config, generate, log, media
from app.chunk import Window

GLOSSARY_LINE_RE = generate.GLOSSARY_LINE_RE
# The footer is the trailing "---" line plus the one link line under it.
NAV_RE = re.compile(r"\n---\n[^\n]*\[\[[^\n]*\n?\s*$")
GLOSSARY_NOTE = "_Glossary.md"
MAP_NOTE = "_Map.md"


def parse_glossary(note_text: str) -> list[tuple[str, str]]:
    """``- emitter follower — wtórnik emiterowy`` pairs from one note."""
    sections = generate.split_sections(note_text)
    entries: list[tuple[str, str]] = []
    for line in sections.get("Glossary", "").splitlines():
        match = GLOSSARY_LINE_RE.match(line)
        if match:
            entries.append((match.group(1).strip(), match.group(2).strip()))
    return entries


def merge_glossary(
    per_note: Sequence[tuple[str, tuple[str, str]]],
) -> list[tuple[str, str, list[str]]]:
    """Collapse duplicates across notes, keeping where each term was defined."""
    merged: dict[str, tuple[str, str, list[str]]] = {}
    for source, (term, translation) in per_note:
        key = term.casefold()
        if key not in merged:
            merged[key] = (term, translation, [source])
        elif source not in merged[key][2]:
            merged[key][2].append(source)
    return sorted(merged.values(), key=lambda entry: entry[0].casefold())


def glossary_note(
    entries: Sequence[tuple[str, str, list[str]]],
    *,
    source_name: str,
    slug: str,
    domain: str = config.DEFAULT_DOMAIN,
) -> str:
    lines = [
        "---",
        f"generator: {generate.GENERATOR}",
        f'source: "{source_name}"',
        "type: glossary",
        f"tags: [{media.slugify(domain)}, {media.slugify(slug)}, glossary]",
        "---",
        "# Glossary",
        "",
    ]
    for term, translation, sources in entries:
        links = " ".join(f"[[{source}]]" for source in sources)
        lines.append(f"- **{term}** — {translation}  {links}")
    return "\n".join(lines) + "\n"


def nav_footer(previous: str | None, following: str | None) -> str:
    links = [
        f"[[{previous}]]" if previous else "",
        f"[[{following}]]" if following else "",
    ]
    return "\n---\n" + " · ".join(link for link in links if link) + "\n"


def apply_nav(note_text: str, previous: str | None, following: str | None) -> str:
    """Replace the note's footer, or add one. Idempotent."""
    body = NAV_RE.sub("", note_text.rstrip()) + "\n"
    if not previous and not following:
        return body
    return body.rstrip() + "\n" + nav_footer(previous, following)


def first_sentence(note_text: str) -> str:
    """The note's own opening sentence, for its line in the map.

    Taken from the note rather than written here: the map must not say anything
    about the lecture that a note does not already say.
    """
    short = generate.split_sections(note_text).get("In short", "").strip()
    flat = " ".join(re.sub(r"[*_`]", "", short).split())
    return re.split(r"(?<=[.!?])\s", flat, maxsplit=1)[0].strip() if flat else ""


def map_entries(notes: Sequence[tuple[Window, Path]]) -> str:
    lines = []
    for window, path in notes:
        opening = first_sentence(path.read_text(encoding="utf-8"))
        tail = f" — {opening}" if opening else ""
        lines.append(f"- **{generate.span_text(window)}** [[{path.stem}]]{tail}")
    return "\n".join(lines)


def map_note(
    notes: Sequence[tuple[Window, Path]],
    *,
    source_name: str,
    slug: str,
    title: str,
    duration_s: float,
    domain: str = config.DEFAULT_DOMAIN,
) -> str:
    return generate.render(
        generate.load_prompt("moc.md"),
        source=source_name,
        title=title or Path(source_name).stem,
        slug=media.slugify(slug),
        domain=media.slugify(domain),
        duration=media.hms(duration_s),
        count=len(notes),
        entries=map_entries(notes),
    )


def run(
    notes: Sequence[tuple[Window, Path]],
    *,
    source_name: str,
    slug: str,
    title: str = "",
    duration_s: float = 0.0,
    domain: str = config.DEFAULT_DOMAIN,
    out_dir: Path | str | None = None,
) -> list[Path]:
    """Link every note to its neighbours, then write the map and the glossary."""
    written: list[Path] = []
    present = [
        (window, path)
        for window, path in notes
        if path.exists() and generate.is_ours(path)
    ]
    foreign = sum(
        1 for _window, path in notes if path.exists() and not generate.is_ours(path)
    )

    for position, (_window, path) in enumerate(present):
        previous = present[position - 1][1].stem if position else None
        following = (
            present[position + 1][1].stem if position + 1 < len(present) else None
        )
        text = path.read_text(encoding="utf-8")
        updated = apply_nav(text, previous, following)
        if updated != text:
            generate.write_note(path, updated, out_dir)
        written.append(path)

    folder = generate.source_dir(slug, out_dir)
    generate.claim_dir(folder, out_dir)

    map_path = folder / MAP_NOTE
    generate.write_note(
        map_path,
        map_note(
            present,
            source_name=source_name,
            slug=slug,
            title=title,
            duration_s=duration_s,
            domain=domain,
        ),
        out_dir,
    )
    written.append(map_path)

    pairs = [
        (path.stem, entry)
        for _window, path in present
        for entry in parse_glossary(path.read_text(encoding="utf-8"))
    ]
    glossary_path = folder / GLOSSARY_NOTE
    generate.write_note(
        glossary_path,
        glossary_note(
            merge_glossary(pairs),
            source_name=source_name,
            slug=slug,
            domain=domain,
        ),
        out_dir,
    )
    written.append(glossary_path)

    log.event(
        "index",
        "index written",
        slug=slug,
        notes=len(present),
        files=len(written),
        foreign=foreign,
    )
    return written
