# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""ffprobe and ffmpeg: what is in the file, and its audio as a WAV.

This is the only stage that touches the container format. Everything after it
sees one 16 kHz mono WAV, so a recording that arrived as an mkv with five audio
tracks and one that arrived as an mp3 are the same problem from here on.

Nothing here runs a model, and nothing here touches the GPU: ffmpeg is told
explicitly to decode on the CPU, because the card is about to be handed to the
ASR stage and a hardware decoder holding 300 MB of it is the kind of overlap
CLAUDE.md forbids.
"""

from __future__ import annotations

import json
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from app import config, log


class MediaError(RuntimeError):
    """The file is not something ffmpeg can give us audio from."""


@dataclass(frozen=True)
class MediaInfo:
    """What ffprobe knows about a recording before anything is decoded."""

    path: Path
    duration_s: float
    audio_streams: int
    language_hint: str | None
    codec: str | None
    # Last, with a default, so the positional constructor keeps working.
    # Needed only to tell an audio-only download from the video of the same
    # lecture, which is a question the batch stage asks (see app/batch.py).
    video_streams: int = 0

    @property
    def duration_hms(self) -> str:
        return hms(self.duration_s)

    @property
    def audio_only(self) -> bool:
        return self.video_streams == 0


# NFKD decomposes ą, ć, ę, ń, ó, ś, ź and ż into a plain letter plus a
# combining mark, which the ASCII encode below then drops. It does not
# decompose ł, because ł is a letter in its own right rather than l with a
# stroke, so without this table "Wykład 4" slugifies to "wykad-4" and the
# recording's folder, tag and every link to it carry the typo. Divergence from
# glosator's extract.slugify; see docs/DECISIONS.md.
_UNDECOMPOSED = str.maketrans(
    {"ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ø": "o", "Ø": "O"}
)


def slugify(text: str) -> str:
    """ASCII, lowercase, hyphenated. Polish diacritics fold to plain letters.

    Copied from glosator's extract.slugify, with the stroked letters fixed.
    """
    folded = (
        unicodedata.normalize("NFKD", text.translate(_UNDECOMPOSED))
        .encode("ascii", "ignore")
        .decode()
    )
    return re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-") or "untitled"


def hms(seconds: float) -> str:
    """``4353.0`` -> ``"01:12:33"``. Hours are never dropped.

    Every timestamp in the output is this shape, so a transcript of a
    forty-minute recording sorts and greps the same way as one of three hours.
    """
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def _run(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    """Run a tool and raise MediaError with its own words on failure.

    ffmpeg says what is wrong on stderr and says it well; a CalledProcessError
    that swallows it leaves the log with an exit code and nothing else.
    """
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except FileNotFoundError as error:
        raise MediaError(f"{argv[0]} is not installed in this image") from error
    except subprocess.TimeoutExpired as error:
        raise MediaError(f"{argv[0]} did not finish within {timeout} s") from error
    except subprocess.CalledProcessError as error:
        tail = " / ".join((error.stderr or "").strip().splitlines()[-3:])
        raise MediaError(f"{argv[0]} failed: {tail or 'no output'}") from error


def probe(media_path: Path) -> MediaInfo:
    """Duration, audio stream count and the language tag, if the file has one.

    The language tag is a hint and nothing more: it is whatever the recording
    software wrote, it is often absent and sometimes wrong. It reaches the log
    and the UI so that a person can confirm it, and it is never passed to the
    ASR model as the language to decode in.
    """
    if not media_path.is_file():
        raise MediaError(f"no such recording: {media_path}")
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(media_path),
        ],
        config.FFPROBE_TIMEOUT_S,
    )
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise MediaError(f"ffprobe returned no usable JSON for {media_path}") from error

    streams = parsed.get("streams", [])
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    video = [stream for stream in streams if stream.get("codec_type") == "video"]
    if not audio:
        raise MediaError(f"{media_path} has no audio stream")

    duration = parsed.get("format", {}).get("duration")
    if duration is None:
        duration = audio[0].get("duration")
    if duration is None:
        # A container without a duration happens: a stream dumped to disk, or a
        # growing file. The pipeline needs a number for the frontmatter and the
        # progress estimate, and decoding the whole file to count samples would
        # cost as much as the transcription.
        raise MediaError(
            f"{media_path} declares no duration; remux it with "
            "'ffmpeg -i in -c copy out.mkv' and try again"
        )

    info = MediaInfo(
        path=media_path,
        duration_s=float(duration),
        audio_streams=len(audio),
        language_hint=(audio[0].get("tags") or {}).get("language"),
        codec=audio[0].get("codec_name"),
        video_streams=len(video),
    )
    log.event(
        "extract",
        "probed",
        media=media_path.name,
        duration=info.duration_hms,
        audio_streams=info.audio_streams,
        codec=info.codec,
        language_hint=info.language_hint,
    )
    if info.audio_streams > 1:
        # Which track is the lecturer and which is the room is not something
        # this program can know, and guessing silently is how a transcript of
        # the audience ends up in a study note.
        log.event(
            "extract",
            "more than one audio stream; the first one is used",
            media=media_path.name,
            audio_streams=info.audio_streams,
        )
    return info


def work_dir(slug: str) -> Path:
    return config.WORK_DIR / slug


def audio_path(slug: str) -> Path:
    return work_dir(slug) / "audio.wav"


def extract_audio(slug: str, media_path: Path, force: bool = False) -> Path:
    """Decode the first audio stream to ``/work/<slug>/audio.wav``.

    16 kHz mono signed 16-bit PCM is what faster-whisper wants; handing it
    anything else means it resamples internally, slower and silently lossy.

    An existing WAV is reused unless ``force``, because this is the cheap stage
    and re-running the expensive one after a crash should not repeat it.
    """
    target = audio_path(slug)
    if target.exists() and target.stat().st_size > 0 and not force:
        log.event("extract", "audio already extracted", slug=slug, path=str(target))
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    # Written to a temporary name and renamed, so a kill does not leave a short
    # WAV that looks finished and transcribes to half a lecture.
    partial = target.with_name(target.name + ".tmp")
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "none",
            "-y",
            "-i",
            str(media_path),
            "-vn",
            "-sn",
            "-dn",
            "-map",
            "0:a:0",
            "-ac",
            str(config.AUDIO_CHANNELS),
            "-ar",
            str(config.AUDIO_RATE),
            "-acodec",
            "pcm_s16le",
            "-f",
            "wav",
            str(partial),
        ],
        config.FFMPEG_TIMEOUT_S,
    )
    partial.replace(target)
    log.event(
        "extract",
        "audio extracted",
        slug=slug,
        path=str(target),
        megabytes=round(target.stat().st_size / 1_000_000, 1),
    )
    return target


def run(slug: str, media_path: Path, force: bool = False) -> MediaInfo:
    """The extract stage: probe the recording, then write its audio."""
    info = probe(media_path)
    extract_audio(slug, media_path, force)
    return info
