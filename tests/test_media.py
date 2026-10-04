# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""ffprobe and ffmpeg.

The tests that run the real binaries are marked ``ffmpeg`` and skip when it is
absent. Their fixture is a three-second tone written by the ``wave`` module, so
the repository carries no binary fixtures.
"""

from __future__ import annotations

import array
import math
import subprocess
import wave
from pathlib import Path

import pytest

from app import config, media

# Skipped by the hook in conftest when the binaries are absent.
needs_ffmpeg = pytest.mark.ffmpeg


@pytest.fixture
def tone(tmp_path: Path) -> Path:
    """Three seconds of 440 Hz at 44.1 kHz stereo: something to resample."""
    path = tmp_path / "tone.wav"
    rate, seconds = 44100, 3
    samples = array.array("h")
    for index in range(rate * seconds):
        value = int(12000 * math.sin(2 * math.pi * 440 * index / rate))
        samples.extend((value, value))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return path


# --- hms and slugify --------------------------------------------------------


def test_hms_always_shows_hours() -> None:
    assert media.hms(0) == "00:00:00"
    assert media.hms(41) == "00:00:41"
    assert media.hms(4353) == "01:12:33"
    assert media.hms(360000) == "100:00:00"


def test_hms_of_a_negative_duration_is_zero() -> None:
    assert media.hms(-5) == "00:00:00"


def test_slugify_folds_polish_diacritics() -> None:
    assert media.slugify("Wykład 4: Próbkowanie") == "wyklad-4-probkowanie"
    assert media.slugify("Żółć ćma ęsie ńań źle") == "zolc-cma-esie-nan-zle"
    assert media.slugify("signal processing") == "signal-processing"
    assert media.slugify("???") == "untitled"


def test_slugify_keeps_the_l_in_a_stroked_l() -> None:
    """NFKD does not decompose ł, so "Wykład" would otherwise become "wykad"."""
    assert media.slugify("Wykład") == "wyklad"
    assert media.slugify("ŁÓDŹ") == "lodz"


# --- probe ------------------------------------------------------------------


@needs_ffmpeg
def test_probe_reads_the_duration_and_the_stream(tone: Path) -> None:
    info = media.probe(tone)
    assert 2.9 < info.duration_s < 3.1
    assert info.duration_hms == "00:00:03"
    assert info.audio_streams == 1
    assert info.codec == "pcm_s16le"


def test_probe_of_a_missing_file_says_so() -> None:
    with pytest.raises(media.MediaError, match="no such recording"):
        media.probe(config.IN_DIR / "not-here.mp4")


@needs_ffmpeg
def test_probe_of_a_file_with_no_audio_is_refused(tmp_path: Path) -> None:
    """A screen recording saved without its audio track, which happens."""
    silent = tmp_path / "silent.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:d=1",
            str(silent),
        ],
        check=True,
        capture_output=True,
    )
    with pytest.raises(media.MediaError, match="no audio stream"):
        media.probe(silent)


@needs_ffmpeg
def test_probe_of_something_that_is_not_media_fails_with_ffprobes_words(
    tmp_path: Path,
) -> None:
    text = tmp_path / "notes.txt"
    text.write_text("this is not a recording\n", encoding="utf-8")
    with pytest.raises(media.MediaError, match="ffprobe failed"):
        media.probe(text)


# --- audio extraction -------------------------------------------------------


@needs_ffmpeg
def test_extract_audio_writes_16k_mono_pcm(tone: Path) -> None:
    path = media.extract_audio("lec4", tone)
    assert path == config.WORK_DIR / "lec4" / "audio.wav"
    with wave.open(str(path), "rb") as handle:
        assert handle.getframerate() == config.AUDIO_RATE
        assert handle.getnchannels() == config.AUDIO_CHANNELS
        assert handle.getsampwidth() == 2


@needs_ffmpeg
def test_extract_audio_reuses_what_is_already_there(tone: Path) -> None:
    """The cheap stage must not be repeated after a crash in the expensive one."""
    first = media.extract_audio("lec4", tone)
    stamp = first.stat().st_mtime_ns
    assert media.extract_audio("lec4", tone).stat().st_mtime_ns == stamp


@needs_ffmpeg
def test_force_re_extracts(tone: Path) -> None:
    media.extract_audio("lec4", tone)
    media.audio_path("lec4").write_bytes(b"truncated")
    media.extract_audio("lec4", tone, force=True)
    assert media.audio_path("lec4").stat().st_size > 1000


@needs_ffmpeg
def test_extract_audio_leaves_no_partial_file_behind(tone: Path) -> None:
    media.extract_audio("lec4", tone)
    assert list((config.WORK_DIR / "lec4").glob("*.tmp")) == []


@needs_ffmpeg
def test_run_probes_and_extracts_in_one_call(tone: Path) -> None:
    info = media.run("lec4", tone)
    assert info.audio_streams == 1
    assert media.audio_path("lec4").exists()


def test_a_missing_binary_is_reported_as_such(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_binary(*_args, **_kwargs):
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr(subprocess, "run", no_binary)
    with pytest.raises(media.MediaError, match="not installed in this image"):
        media._run(["ffprobe", "-version"], 5)


def test_a_timeout_is_reported_with_its_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    def hangs(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("ffmpeg", 7)

    monkeypatch.setattr(subprocess, "run", hangs)
    with pytest.raises(media.MediaError, match="within 7 s"):
        media._run(["ffmpeg"], 7)


def test_a_container_without_a_duration_says_what_to_do(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recording = tmp_path / "stream.mkv"
    recording.write_bytes(b"x")

    def ffprobe(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            [], 0, stdout='{"streams":[{"codec_type":"audio"}],"format":{}}', stderr=""
        )

    monkeypatch.setattr(subprocess, "run", ffprobe)
    with pytest.raises(media.MediaError, match="remux it"):
        media.probe(recording)


def test_more_than_one_audio_stream_is_logged_not_guessed_about(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Which track is the lecturer is not something this program can know."""
    recording = tmp_path / "two-tracks.mkv"
    recording.write_bytes(b"x")

    def ffprobe(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            [],
            0,
            stdout=(
                '{"streams":[{"codec_type":"audio","codec_name":"aac",'
                '"tags":{"language":"pol"}},{"codec_type":"audio"}],'
                '"format":{"duration":"60.0"}}'
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", ffprobe)
    info = media.probe(recording)
    assert info.audio_streams == 2
    assert info.language_hint == "pol"

    from app import log

    assert "more than one audio stream" in log.tail()
