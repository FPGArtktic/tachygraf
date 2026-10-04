# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""Text pass: one window in, one note out.

The model writes prose only. Frontmatter, the heading, the section order and
the footer are assembled here, so a note's machine-readable parts never depend
on the model behaving. That much is glosator's design, and the path and
ownership guards below are copied from it verbatim.

What is new here is ``parse``. A note's claims point back into the transcript
by label, and a reference that looks right and is not would be worse than no
reference at all, so every rule in CLAUDE.md's table that can be checked is
checked here: an unknown label, a written timestamp and a missing section each
reject the whole response, and a claim without a label is dropped and counted.
A response that reaches the output folder has passed all of them.

This module imports nothing from ``chunk`` or ``transcribe`` at run time, even
though it sits after them in the pipeline, because they both need the write
guards that live here.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from app import config, log, media, ollama

if TYPE_CHECKING:  # the window type, for the signatures only
    from app.chunk import Window

PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")

# Models trained on LaTeX emit \[ ... \] and \( ... \) however loudly the
# prompt asks for dollars, and Obsidian renders neither.
MATH_DISPLAY_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
MATH_INLINE_RE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
SECTION_RE = re.compile(r"^##\s+(.*\S)\s*$", re.MULTILINE)

# A paragraph label as the window text presents it and as the prompt asks for
# it back: four digits, square brackets, nothing else.
LABEL_RE = re.compile(r"\[(p\d{4})\]")

# Any clock time at all. The model is never shown one, so a time in its answer
# was invented, and a study note that cites a time that does not exist is worse
# than one that cites nothing.
TIME_RE = re.compile(r"\d\d:\d\d")

BULLET_RE = re.compile(r"^\s*[-*+]\s+")

# What the prompt asks for when a window holds no formula. Matched loosely,
# because a model that is told to write one sentence writes it with its own
# punctuation.
NO_FORMULAS_RE = re.compile(r"none in this window", re.IGNORECASE)

# A glossary line is "term — polski odpowiednik". The separator is an em dash;
# a hyphen is accepted because the term itself may be hyphenated and the model
# reaches for whichever is on its keyboard.
GLOSSARY_LINE_RE = re.compile(r"^\s*[-*+]\s+(.+?)\s+[—–-]\s+(.+?)\s*$")  # noqa: RUF001

# Written into every file's frontmatter and checked before any overwrite: a
# file without it was not produced here and is never touched.
GENERATOR = "tachygraf"

# Dropped into every directory tachygraf creates inside the output folder, so a
# later run can tell its own directories from the user's.
OWNER_STAMP = ".tachygraf"

# The transcript note every reference points at. Notes and transcript share one
# folder, so the short link resolves to the right recording's transcript even
# when several recordings have been moved into one vault.
TRANSCRIPT_LINK = "_Transcript"

SECTION_ORDER = (
    "In short",
    "Explanation",
    "Key claims",
    "Formulas",
    "Glossary",
    "Pitfalls",
)
REQUIRED_SECTIONS = ("In short", "Explanation")

# Sections whose bullets must each cite at least one label.
CITED_SECTIONS = ("Key claims", "Pitfalls")

# Longest title taken from the model's prose, in characters. A file name is
# read in a sidebar and typed into a link; the rest of the sentence is in the
# note's first section anyway.
TITLE_MAX_CHARS = 60

# Openings that name the exercise rather than the subject. The prompt forbids
# them by name; this is what checks for them.
TASK_TITLE_RE = re.compile(
    r"^\s*(?:in\s+)?th(?:is|e)\s+(?:part|section|portion|window|segment)"
    r"(?:\s+of\s+th(?:is|e)\s+\w+)?\s*(?:explains?|covers?|describes?|discusses?|"
    r"introduces?|is\s+about|deals\s+with)?\s*[:,-]?\s*"
    r"|^\s*(?:in\s+)?th(?:is|e)\s+(?:lecture|video|recording|talk|window)\s*"
    r"(?:explains?|covers?|describes?|discusses?|introduces?|is\s+about)?\s*[:,-]?\s*"
    r"|^\s*the\s+speaker\s+(?:explains?|covers?|describes?|discusses?)?\s*",
    re.IGNORECASE,
)

# Words a title must not end on once it has been cut to length.
DANGLING_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "but",
        "by",
        "for",
        "from",
        "how",
        "if",
        "in",
        "into",
        "is",
        "of",
        "on",
        "or",
        "the",
        "their",
        "they",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "why",
        "with",
    }
)

# Characters a file name may not carry. The output folder may be copied to
# another filesystem or synced to a phone, so this is the strict set, not the
# one Linux happens to allow.
UNSAFE_NAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class NoteRejected(ValueError):
    """A response broke a hard rule the parser can check.

    The rule is in the message. The worker records it against the window, so a
    run can be read afterwards as "which rule did this model break, how often",
    which is how the next rule gets written (see docs/DECISIONS.md).
    """


@dataclass
class ParsedNote:
    """What survived the parser, and what did not."""

    sections: dict[str, str]
    title: str
    dropped_claims: int = 0
    dropped_glossary: int = 0
    title_substituted: bool = False
    labels_cited: set[str] = field(default_factory=set)


def load_prompt(name: str) -> str:
    return (config.PROMPT_DIR / name).read_text(encoding="utf-8")


def render(template: str, **fields: object) -> str:
    """Replace ``{{name}}`` placeholders. An unknown placeholder is an error."""

    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in fields:
            raise KeyError(f"template placeholder {{{{{key}}}}} has no value")
        return str(fields[key])

    return PLACEHOLDER_RE.sub(substitute, template)


def estimate_tokens(text: str) -> int:
    """Rough token count. Lives here because the budget check below uses it."""
    return len(text) // config.CHARS_PER_TOKEN


def normalise_math(text: str) -> str:
    """Rewrite LaTeX delimiters to the ones Obsidian renders.

    Copied from glosator.
    """
    text = MATH_DISPLAY_RE.sub(lambda m: f"$${m.group(1).strip()}$$", text)
    return MATH_INLINE_RE.sub(lambda m: f"${m.group(1).strip()}$", text)


def split_sections(markdown: str) -> dict[str, str]:
    """``## Heading`` to body text, for reassembly in the contract's order."""
    matches = list(SECTION_RE.finditer(markdown))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        sections[match.group(1)] = markdown[match.end() : end].strip()
    return sections


def missing_sections(sections: dict[str, str]) -> list[str]:
    return [name for name in REQUIRED_SECTIONS if not sections.get(name, "").strip()]


def strip_wrapping(raw: str) -> str:
    """Drop a fenced block or invented frontmatter around the model's answer.

    Copied from glosator.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            text = parts[2]
    return re.sub(r"^#\s+.*\n", "", text.strip(), count=1).strip()


# --- the rule checks --------------------------------------------------------


def bullets(block: str) -> list[str]:
    """Split a markdown list into bullets, keeping wrapped lines together.

    A model wraps a long claim over three lines and puts the label on the last
    one. Checking line by line would drop the first two as unlabelled.
    """
    items: list[str] = []
    for line in block.splitlines():
        if BULLET_RE.match(line):
            items.append(line.strip())
        elif line.strip() and items:
            items[-1] = f"{items[-1]} {line.strip()}"
    return items


def labels_in(text: str) -> list[str]:
    return LABEL_RE.findall(text)


def link_labels(text: str) -> str:
    """Rewrite every ``[pNNNN]`` into the block reference it stands for.

    The model writes labels because labels are all it is given. The reader
    wants a link, so the labels become links here, after they have been checked
    against the window.
    """
    collapsed = re.sub(r"\[(p\d{4})\]((?:\s*\[p\d{4}\])+)", _group_labels, text)
    return LABEL_RE.sub(lambda m: f"([[{TRANSCRIPT_LINK}#^{m.group(1)}]])", collapsed)


def _group_labels(match: re.Match[str]) -> str:
    """Several labels in a row become one parenthesis, not several."""
    found = labels_in(match.group(0))
    inner = " ".join(f"[[{TRANSCRIPT_LINK}#^{label}]]" for label in found)
    return f"({inner})"


def clean_title(text: str) -> str:
    """Strip a candidate title to something that can be a file name."""
    text = re.sub(r"[*_`#\[\]]", "", text).strip()
    text = " ".join(text.split())
    text = UNSAFE_NAME_RE.sub("", text)
    return text.strip(" .-\u2014")


def is_about_the_task(title: str) -> bool:
    """True when a title names the exercise instead of the subject.

    Measured: asked for an "In short" and given no title to write, the 32B
    model opened with "This part of the lecture explains what diodes are, how
    they work, and their applications" -- which cut to a file name became
    "This part of the lecture explains what diodes are, how they". The prompt
    now asks for a title and forbids exactly these openings; this is the check
    behind that rule.
    """
    return TASK_TITLE_RE.match(title) is not None


def title_from(sections: dict[str, str], order: int) -> str:
    """The window's title: the one the model wrote, or a fallback.

    The model is asked for a ``## Title`` line, because the pilot showed what
    deriving one from its prose produces: a sentence about the recording,
    truncated mid-phrase, repeated in every wikilink that points at the note.

    A missing or task-naming title does not fail the response. A note is ten
    minutes of generation and the rest of it may be perfectly good, so the
    first sentence of "In short" is used instead, trimmed the same way, and
    the substitution is counted. See docs/DECISIONS.md.
    """
    given = clean_title(
        sections.get("Title", "").splitlines()[0] if sections.get("Title") else ""
    )
    if given and not is_about_the_task(given):
        return _trim(given) or f"Window {order:02d}"

    short = clean_title(sections.get("In short", ""))
    sentence = re.split(r"(?<=[.!?])\s", short, maxsplit=1)[0]
    sentence = TASK_TITLE_RE.sub("", sentence).strip(" ,.-")
    sentence = sentence[:1].upper() + sentence[1:] if sentence else ""
    return _trim(sentence) or f"Window {order:02d}"


def _trim(text: str) -> str:
    """Cut to the file-name budget at a clause or word boundary.

    A title cut mid-phrase ("... what diodes are, how they") reads as a
    truncation, so a comma inside the budget wins over a word boundary, and a
    dangling conjunction or preposition is dropped.
    """
    if len(text) <= TITLE_MAX_CHARS:
        return text.strip(" ,.-")
    head = text[:TITLE_MAX_CHARS]
    clause = head.rsplit(",", 1)[0]
    # A clause is preferred to a word boundary whenever it is long enough to
    # name the subject at all: "Aliasing in sampled systems" is a title,
    # "Aliasing in sampled systems, and why a filter afterwards" is a cut.
    if len(clause) >= TITLE_MAX_CHARS // 3:
        head = clause
    else:
        head = head.rsplit(" ", 1)[0] or head
    words = head.split()
    while words and words[-1].lower() in DANGLING_WORDS:
        words.pop()
    return " ".join(words).strip(" ,.-")


def parse(raw: str, labels: set[str], order: int) -> ParsedNote:
    """Check a response against every rule that can be checked, or reject it.

    Three failures reject the whole note, because each means the model was not
    reading the window it was given: a section missing, a label that is not in
    the window, and a clock time it cannot have seen. Two failures cost one
    line: a claim with no label, and a glossary line in the wrong shape.
    """
    text = normalise_math(strip_wrapping(raw))
    sections = split_sections(text)

    absent = missing_sections(sections)
    if absent:
        raise NoteRejected(f"missing section: {', '.join(absent)}")

    unknown = sorted(set(labels_in(text)) - labels)
    if unknown:
        raise NoteRejected(
            f"cited {len(unknown)} label(s) that are not in this window: "
            f"{', '.join(unknown[:5])}"
        )

    found_time = TIME_RE.search(text)
    if found_time:
        raise NoteRejected(f"wrote a timestamp: {found_time.group(0)!r}")

    parsed = ParsedNote(sections=dict(sections), title="")
    parsed.labels_cited = set(labels_in(text))

    for name in CITED_SECTIONS:
        block = parsed.sections.get(name, "").strip()
        if not block:
            continue
        kept = [item for item in bullets(block) if labels_in(item)]
        parsed.dropped_claims += len(bullets(block)) - len(kept)
        parsed.sections[name] = "\n".join(kept)

    formulas = parsed.sections.get("Formulas", "").strip()
    if formulas and not NO_FORMULAS_RE.search(formulas) and not labels_in(formulas):
        # A formula with no label is a formula from the model's memory rather
        # than from the lecture, which the rules call the worst thing a note
        # can contain. There is no way to tell which it is, so neither is kept.
        raise NoteRejected("a Formulas block that cites no label")

    glossary = parsed.sections.get("Glossary", "").strip()
    if glossary:
        lines = [line for line in glossary.splitlines() if line.strip()]
        kept = [line.strip() for line in lines if GLOSSARY_LINE_RE.match(line)]
        parsed.dropped_glossary = len(lines) - len(kept)
        parsed.sections["Glossary"] = "\n".join(kept)

    # A note whose Key claims were all dropped has no link into the transcript
    # at all, which is the one thing every note here is for.
    if "Key claims" in sections and not parsed.sections.get("Key claims", "").strip():
        raise NoteRejected("no claim in Key claims carried a label")

    for name, block in parsed.sections.items():
        parsed.sections[name] = link_labels(block)

    given = clean_title(sections.get("Title", ""))
    parsed.title = title_from(sections, order)
    parsed.title_substituted = not given or is_about_the_task(given)
    return parsed


# --- the note ---------------------------------------------------------------


def span_text(window: Window) -> str:
    return f"{media.hms(window.start_s)} - {media.hms(window.end_s)}"


def frontmatter(
    window: Window,
    *,
    source_name: str,
    slug: str,
    domain: str,
    model: str,
    generated: date,
) -> str:
    return "\n".join(
        [
            "---",
            f"generator: {GENERATOR}",
            f"prompt: {config.PROMPT_VERSION}",
            f'source: "{source_name}"',
            f'span: "{span_text(window)}"',
            f"model: {model}",
            f"generated: {generated.isoformat()}",
            f"tags: [{media.slugify(domain)}, {media.slugify(slug)}]",
            "---",
        ]
    )


def assemble(
    window: Window,
    parsed: ParsedNote,
    *,
    source_name: str,
    slug: str,
    domain: str = config.DEFAULT_DOMAIN,
    model: str = config.TEXT_MODEL,
    generated: date | None = None,
) -> str:
    """Build the note exactly as the output contract specifies."""
    body = [
        frontmatter(
            window,
            source_name=source_name,
            slug=slug,
            domain=domain,
            model=model,
            generated=generated or date.today(),
        ),
        f"# {window.ord:02d} {parsed.title}",
    ]
    for name in SECTION_ORDER:
        text = parsed.sections.get(name, "").strip()
        if text:
            body.append(f"## {name}\n\n{text}")
    return "\n\n".join(body) + "\n"


# --- paths and guards (copied from glosator) --------------------------------


def out_root(out_dir: Path | str | None = None) -> Path:
    return Path(out_dir) if out_dir else config.OUT_DIR


def source_dir(slug: str, out_dir: Path | str | None = None) -> Path:
    """``<out>/<slug>/``. Everything for one recording lands here and nowhere else."""
    return out_root(out_dir) / media.slugify(slug)


def note_name(window: Window, title: str) -> str:
    """``03 Why filtering after sampling cannot undo aliasing``."""
    return f"{window.ord:02d} {title}".strip()


def note_path(
    window: Window, title: str, slug: str, out_dir: Path | str | None = None
) -> Path:
    return source_dir(slug, out_dir) / f"{note_name(window, title)}.md"


def is_ours_text(text: str) -> bool:
    """True when a file's frontmatter carries this generator's marker."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return False
    for line in lines[1:]:
        if line.strip() == "---":
            return False
        if line.strip() == f"generator: {GENERATOR}":
            return True
    return False


def is_ours(path: Path) -> bool:
    """True when the file at ``path`` is one tachygraf wrote.

    Only the head of the file is read: a note is a few kilobytes, but a file
    that happens to sit at the same path may be anything at all.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            head = "".join(next(handle, "") for _ in range(20))
    except OSError:
        return False
    return is_ours_text(head)


def check_inside(path: Path, out_dir: Path | str | None = None) -> Path:
    """Resolve a path and refuse it if it leaves the output folder."""
    root = out_root(out_dir).resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise PermissionError(f"{resolved} is outside the output folder {root}")
    return resolved


def check_writable(path: Path, out_dir: Path | str | None = None) -> Path:
    """Refuse to write outside the output folder or over someone else's file.

    This is the guard that keeps an existing note collection intact: the only
    files tachygraf may replace are the ones it wrote itself. The output folder
    is a staging folder here rather than a vault, and the guard is kept anyway,
    because a staging folder can hold things that matter too.
    """
    resolved = check_inside(path, out_dir)
    if resolved.is_dir():
        raise PermissionError(f"{resolved} is a directory, not a note")
    if resolved.exists() and not is_ours(resolved):
        raise PermissionError(
            f"{resolved} was not written by tachygraf; refusing to overwrite it"
        )
    return resolved


def claim_dir(directory: Path, out_dir: Path | str | None = None) -> Path:
    """Create a directory tachygraf owns, or confirm that it already owns it.

    A non-empty directory without the stamp is the user's and is left
    untouched: the recording's slug may well collide with a folder they made.
    """
    checked = check_inside(directory, out_dir)
    stamp = checked / OWNER_STAMP
    if checked.exists() and not stamp.exists() and any(checked.iterdir()):
        raise PermissionError(f"{checked} already holds files tachygraf did not write")
    checked.mkdir(parents=True, exist_ok=True)
    stamp.touch()
    return checked


def write_atomic(path: Path, text: str) -> None:
    """Write via a temporary file in the same directory, then rename.

    A kill in the middle leaves either the old file or the new one, never half
    of either. No ownership check: callers that write into the output folder
    use ``write_note``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_note(path: Path, text: str, out_dir: Path | str | None = None) -> Path:
    """Atomically write a generated file, after the ownership check."""
    checked = check_writable(path, out_dir)
    write_atomic(checked, text)
    return checked


# --- the stage --------------------------------------------------------------


def run(
    window: Window,
    *,
    slug: str,
    source_name: str,
    windows: int,
    domain: str = config.DEFAULT_DOMAIN,
    model: str = config.TEXT_MODEL,
    out_dir: Path | str | None = None,
) -> Path:
    """Generate one note. Raises when the input or the response is unusable."""
    source_text = window.text_path.read_text(encoding="utf-8")
    tokens = estimate_tokens(source_text)
    if tokens > config.CHUNK_MAX_TOKENS:
        # Ollama would silently drop the overflow and the note would describe
        # half a window without saying so.
        raise ValueError(
            f"window {window.ord}: {tokens} tokens of transcript exceed the "
            f"{config.CHUNK_MAX_TOKENS} token budget; re-run the chunk stage"
        )

    prompt = render(
        load_prompt("note.md"),
        domain=domain,
        source=source_name,
        window=window.ord,
        windows=windows,
        text=source_text,
    )
    raw = ollama.generate(model, prompt)
    parsed = parse(raw, set(window.labels), window.ord)

    note = assemble(
        window,
        parsed,
        source_name=source_name,
        slug=slug,
        domain=domain,
        model=model,
    )
    claim_dir(source_dir(slug, out_dir), out_dir)
    path = write_note(note_path(window, parsed.title, slug, out_dir), note, out_dir)
    log.event(
        "generate",
        "note written",
        window=window.ord,
        path=str(path),
        words=len(note.split()),
        cited=len(parsed.labels_cited),
        dropped_claims=parsed.dropped_claims,
        dropped_glossary=parsed.dropped_glossary,
        title_substituted=parsed.title_substituted,
    )
    return path
