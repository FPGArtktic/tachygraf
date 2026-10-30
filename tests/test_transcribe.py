# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""The ASR stage, against a fake model.

No weights, no card and no CTranslate2 are needed: ``load_model`` is the single
seam, and every test replaces it.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from app import config, media, transcribe


@dataclass
class FakeSegment:
    start: float
    end: float
    text: str
    avg_logprob: float = -0.2


@dataclass
class FakeInfo:
    language: str = "pl"
    language_probability: float = 0.98
    duration: float = 180.0


class FakeModel:
    """Returns canned segments and records how it was asked for them."""

    def __init__(self, segments, info=None):
        self._segments = segments
        self._info = info or FakeInfo()
        self.calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({"audio": audio, **kwargs})
        return iter(self._segments), self._info


@pytest.fixture
def on_the_cpu(monkeypatch: pytest.MonkeyPatch):
    """A machine with no usable card, which is what CI and this test suite are."""
    monkeypatch.setattr(transcribe, "gpu_free_mb", lambda: None)


def _install(monkeypatch, model) -> list[tuple[str, str, str]]:
    loaded: list[tuple[str, str, str]] = []

    def load(name, device, compute_type):
        loaded.append((name, device, compute_type))
        return model

    monkeypatch.setattr(transcribe, "load_model", load)
    return loaded


def _write_wav(
    path: Path,
    seconds: float = 1.0,
    rate: int = config.AUDIO_RATE,
    channels: int = config.AUDIO_CHANNELS,
    width: int = 2,
) -> Path:
    """A real WAV, written with the standard library: no binary fixtures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(b"\x00" * int(rate * seconds) * channels * width)
    return path


def _audio(slug: str = "lec4") -> Path:
    return _write_wav(media.audio_path(slug))


# --- device choice ----------------------------------------------------------


def test_a_free_card_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """15000 MB: the reference card, 16 GB, with the desktop on it."""
    monkeypatch.setattr(transcribe, "gpu_free_mb", lambda: 15000)
    device, compute, reason = transcribe.choose_device("large-v3-turbo")
    assert device == "cuda"
    assert compute == config.ASR_COMPUTE_GPU
    assert "15000 MB" in reason


def test_a_busy_card_falls_back_to_the_cpu_and_says_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLAUDE.md: a silent fallback is the docling failure glosator already met."""
    monkeypatch.setattr(transcribe, "gpu_free_mb", lambda: 400)
    device, compute, reason = transcribe.choose_device("large-v3-turbo")
    assert device == "cpu"
    assert compute == config.ASR_COMPUTE_CPU
    assert "400 MB" in reason
    assert f"large-v3-turbo needs {transcribe.needed_mb('large-v3-turbo')} MB" in reason


def test_no_card_at_all_is_a_reason_not_an_error(on_the_cpu) -> None:
    device, _compute, reason = transcribe.choose_device()
    assert device == "cpu"
    assert "no usable NVIDIA card" in reason


def test_a_card_that_fails_during_inference_falls_back_to_the_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the missing CUDA libraries actually did: fail at the first compute.

    nvml said the card was free, the weights loaded, and CTranslate2 then could
    not dlopen libcublas. A failure there must degrade, not fail the job.
    """
    monkeypatch.setattr(transcribe, "gpu_free_mb", lambda: 15000)
    attempts: list[str] = []

    class FailsOnInference:
        def transcribe(self, audio, **kwargs):
            raise RuntimeError("Library libcublas.so.12 is not found")

    def load(name, device, compute_type):
        attempts.append(device)
        if device == "cuda":
            return FailsOnInference()
        return FakeModel([FakeSegment(0.0, 2.0, "Recovered on the CPU.")])

    monkeypatch.setattr(transcribe, "load_model", load)
    # Named explicitly: a model that fits the card, so the stage tries it.
    segments, reported = transcribe.transcribe_audio(_audio(), "large-v3-turbo")

    assert attempts == ["cuda", "cpu"]
    assert reported["device"] == "cpu"
    assert reported["compute_type"] == config.ASR_COMPUTE_CPU
    assert "libcublas" in reported["device_reason"]
    assert segments[0].text == "Recovered on the CPU."

    from app import log

    assert "falling back to the CPU" in log.tail()


def test_a_failure_on_the_cpu_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """There is nowhere left to fall back to, so the job must fail."""
    monkeypatch.setattr(transcribe, "gpu_free_mb", lambda: None)
    attempts: list[str] = []

    def load(name, device, compute_type):
        attempts.append(device)
        raise RuntimeError("the weights are not in the image")

    monkeypatch.setattr(transcribe, "load_model", load)
    with pytest.raises(RuntimeError, match="not in the image"):
        transcribe.transcribe_audio(_audio())
    assert attempts == ["cpu"]


def test_the_card_refusing_the_model_is_retried_on_the_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """nvml can report memory that something else claims a moment later."""
    monkeypatch.setattr(transcribe, "gpu_free_mb", lambda: 15000)
    attempts: list[str] = []

    def load(name, device, compute_type):
        attempts.append(device)
        if device == "cuda":
            raise RuntimeError("CUDA failed with error out of memory")
        return FakeModel([FakeSegment(0.0, 2.0, "Dzień dobry.")])

    monkeypatch.setattr(transcribe, "load_model", load)
    _segments, reported = transcribe.transcribe_audio(_audio(), "large-v3-turbo")
    assert attempts == ["cuda", "cpu"]
    assert reported["device"] == "cpu"


# --- segments ---------------------------------------------------------------


def test_transcribe_audio_passes_the_named_language_through(
    monkeypatch: pytest.MonkeyPatch, on_the_cpu
) -> None:
    model = FakeModel([FakeSegment(0.0, 2.0, " Dzień dobry. ")])
    _install(monkeypatch, model)
    segments, reported = transcribe.transcribe_audio(_audio(), "large-v3", "pl")
    assert model.calls[0]["language"] == "pl"
    assert model.calls[0]["vad_filter"] is config.ASR_VAD_FILTER
    assert segments[0].text == "Dzień dobry."
    assert reported["model"] == "large-v3"


def test_auto_means_the_model_decides(
    monkeypatch: pytest.MonkeyPatch, on_the_cpu
) -> None:
    """'auto' is not a language code; passing it as one transcribes nonsense."""
    model = FakeModel([FakeSegment(0.0, 2.0, "Hello.")])
    _install(monkeypatch, model)
    transcribe.transcribe_audio(_audio(), language="auto")
    assert model.calls[0]["language"] is None


def test_whisper_is_not_conditioned_on_its_own_text(
    monkeypatch: pytest.MonkeyPatch, on_the_cpu
) -> None:
    """faster-whisper's default turned one mistake into five copies of it."""
    model = FakeModel([FakeSegment(0.0, 2.0, "Hello.")])
    _install(monkeypatch, model)
    transcribe.transcribe_audio(_audio())
    assert model.calls[0]["condition_on_previous_text"] is False


def test_empty_segments_are_dropped(
    monkeypatch: pytest.MonkeyPatch, on_the_cpu
) -> None:
    model = FakeModel(
        [FakeSegment(0.0, 1.0, "   "), FakeSegment(1.0, 2.0, "Real speech.")]
    )
    _install(monkeypatch, model)
    segments, _reported = transcribe.transcribe_audio(_audio())
    assert [segment.text for segment in segments] == ["Real speech."]


def test_segments_round_trip_through_the_json(
    monkeypatch: pytest.MonkeyPatch, on_the_cpu
) -> None:
    model = FakeModel([FakeSegment(0.0, 2.0, "Jeden."), FakeSegment(9.0, 11.0, "Dwa.")])
    _install(monkeypatch, model)
    segments, reported = transcribe.transcribe_audio(_audio())
    transcribe.write_segments("lec4", segments, reported)
    loaded, loaded_reported = transcribe.load_segments("lec4")
    assert loaded == segments
    assert loaded_reported["language"] == "pl"


def test_run_refuses_to_write_an_empty_transcript(
    monkeypatch: pytest.MonkeyPatch, on_the_cpu, asr_in_process
) -> None:
    """Silence, the wrong track, or a VAD gate over the whole file."""
    _install(monkeypatch, FakeModel([]))
    _audio()
    with pytest.raises(ValueError, match="no speech"):
        transcribe.run("lec4", Path("/in/lecture-04.mp4"))


def test_run_without_extracted_audio_says_which_stage_is_missing(
    on_the_cpu, asr_in_process
) -> None:
    with pytest.raises(FileNotFoundError, match="extract stage"):
        transcribe.run("lec4", Path("/in/lecture-04.mp4"))


# --- paragraphs -------------------------------------------------------------


def test_a_pause_ends_a_paragraph() -> None:
    built = transcribe.paragraphs(
        [
            transcribe.Segment(0.0, 2.0, "Pierwsze zdanie.", -0.2),
            transcribe.Segment(2.1, 4.0, "Drugie zdanie.", -0.2),
            transcribe.Segment(9.0, 11.0, "Po przerwie.", -0.2),
        ]
    )
    assert [paragraph.label for paragraph in built] == ["p0001", "p0002"]
    assert built[0].text == "Pierwsze zdanie. Drugie zdanie."
    assert built[1].text == "Po przerwie."
    assert built[1].start == 9.0


def test_a_paragraph_is_closed_by_length_when_nobody_pauses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "PARAGRAPH_MAX_S", 10.0)
    segments = [
        transcribe.Segment(float(index) * 2, float(index) * 2 + 1.9, f"S{index}.", -0.2)
        for index in range(12)
    ]
    built = transcribe.paragraphs(segments)
    assert len(built) > 1
    assert all(
        paragraph.end - paragraph.start <= config.PARAGRAPH_MAX_S + 2
        for paragraph in built
    )


def test_paragraphs_are_stable_across_calls() -> None:
    """The labels are the transcript's block IDs and the note's only references."""
    segments = [
        transcribe.Segment(0.0, 2.0, "Jeden.", -0.2),
        transcribe.Segment(9.0, 11.0, "Dwa.", -0.2),
    ]
    assert transcribe.paragraphs(segments) == transcribe.paragraphs(segments)


def test_a_low_confidence_segment_is_marked_unclear() -> None:
    built = transcribe.paragraphs(
        [
            transcribe.Segment(0.0, 2.0, "Wyraźnie.", -0.2),
            transcribe.Segment(2.1, 4.0, "mmm coś tam", -1.4),
        ]
    )
    assert built[0].text == "Wyraźnie. mmm coś tam [unclear]"
    assert built[0].unclear


def test_a_confident_segment_carries_no_marker() -> None:
    built = transcribe.paragraphs([transcribe.Segment(0.0, 2.0, "Wyraźnie.", -0.1)])
    assert transcribe.UNCLEAR not in built[0].text
    assert not built[0].unclear


def test_a_repetition_loop_is_marked_unclear_despite_its_confidence() -> None:
    """Measured: eleven "you run it" at -0.20, in place of what was said."""
    loop = "So you start Claude, you download it" + ", you run it" * 11 + "."
    built = transcribe.paragraphs(
        [
            transcribe.Segment(0.0, 2.0, "There is no setup.", -0.2),
            transcribe.Segment(2.1, 30.0, loop, -0.2),
        ]
    )
    assert built[0].text == f"There is no setup. {loop} [unclear]"


def test_a_speaker_repeating_a_phrase_is_not_a_loop() -> None:
    """Real speech, from the same talk; its repeats add six words at most."""
    said = "You just say use this tool and this tool and this tool and this tool."
    built = transcribe.paragraphs([transcribe.Segment(0.0, 5.0, said, -0.2)])
    assert transcribe.UNCLEAR not in built[0].text


def test_a_loop_across_segments_marks_every_segment_it_runs_through() -> None:
    """A 24-word sentence, once per segment, three segments under a second each."""
    sentence = (
        "keep it as short as you can and for example in our code base we have "
        "a lot of context so we're going to"
    )
    segments = [
        transcribe.Segment(0.0, 3.0, "Try to keep it pretty short.", -0.2),
        transcribe.Segment(3.0, 3.5, sentence, -0.25),
        transcribe.Segment(3.5, 4.5, sentence, -0.25),
        transcribe.Segment(4.5, 5.5, sentence, -0.25),
    ]
    assert transcribe.looping(segments) == {1, 2, 3}
    assert transcribe.paragraphs(segments)[0].text.count(transcribe.UNCLEAR) == 3


def test_no_segments_means_no_paragraphs() -> None:
    assert transcribe.paragraphs([]) == []


# --- the transcript note ----------------------------------------------------


def _transcript(**kwargs) -> str:
    built = transcribe.paragraphs(
        [
            transcribe.Segment(0.0, 2.0, "Pierwszy akapit.", -0.2),
            transcribe.Segment(41.0, 44.0, "Drugi akapit.", -0.2),
        ]
    )
    defaults = {
        "source_name": "lecture-04.mp4",
        "duration_s": 4353.0,
        "language": "pl",
        "model": "large-v3",
    }
    return transcribe.transcript_markdown(built, **(defaults | kwargs))


def test_the_transcript_matches_the_output_contract() -> None:
    text = _transcript()
    assert text.startswith("---\ngenerator: tachygraf\n")
    assert 'source: "lecture-04.mp4"' in text
    assert 'duration: "01:12:33"' in text
    assert "language: pl" in text
    assert "model: large-v3" in text
    assert "# Transcript — lecture-04" in text
    assert "**[00:00:00]** Pierwszy akapit. ^p0001" in text
    assert "**[00:00:41]** Drugi akapit. ^p0002" in text


def test_every_paragraph_carries_exactly_one_timestamp_and_one_block_id() -> None:
    """A timestamp per segment would make the transcript a subtitle file."""
    body = _transcript().split("---", 2)[2]
    paragraphs = [line for line in body.splitlines() if line.startswith("**[")]
    assert len(paragraphs) == 2
    for line in paragraphs:
        assert line.count("**[") == 1
        assert line.count(" ^p") == 1


def test_run_writes_the_transcript_into_the_output_folder(
    monkeypatch: pytest.MonkeyPatch, on_the_cpu, asr_in_process, tmp_path: Path
) -> None:
    _install(
        monkeypatch,
        FakeModel(
            [
                FakeSegment(0.0, 2.0, "Pierwszy akapit."),
                FakeSegment(41.0, 44.0, "Drugi akapit."),
            ]
        ),
    )
    _audio()
    path = transcribe.run("lec4", Path("/in/lecture-04.mp4"), duration_s=4353.0)

    assert path == config.OUT_DIR / "lec4" / "_Transcript.md"
    assert "^p0002" in path.read_text(encoding="utf-8")
    # The folder tachygraf created is stamped, and the segments are kept.
    assert (path.parent / ".tachygraf").exists()
    assert transcribe.segments_path("lec4").exists()


def test_run_refuses_to_overwrite_a_file_it_did_not_write(
    monkeypatch: pytest.MonkeyPatch, on_the_cpu, asr_in_process
) -> None:
    _install(monkeypatch, FakeModel([FakeSegment(0.0, 2.0, "Cokolwiek.")]))
    _audio()
    folder = config.OUT_DIR / "lec4"
    folder.mkdir(parents=True)
    (folder / ".tachygraf").touch()
    foreign = folder / "_Transcript.md"
    foreign.write_text("notes someone else wrote\n", encoding="utf-8")

    with pytest.raises(PermissionError, match="not written by tachygraf"):
        transcribe.run("lec4", Path("/in/lecture-04.mp4"))
    assert foreign.read_text(encoding="utf-8") == "notes someone else wrote\n"


# --- the audio the model is handed ---------------------------------------------


def test_the_model_is_handed_samples_not_a_path(
    monkeypatch: pytest.MonkeyPatch, on_the_cpu
) -> None:
    """Keeps PyAV out of the pipeline; it broke the stage once already."""
    model = FakeModel([FakeSegment(0.0, 2.0, "Hello.")])
    _install(monkeypatch, model)
    transcribe.transcribe_audio(_audio())
    handed = model.calls[0]["audio"]
    assert isinstance(handed, np.ndarray)
    assert handed.dtype == np.float32


def test_read_wav_scales_int16_into_minus_one_to_one(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(config.AUDIO_RATE)
        handle.writeframes(b"\x00\x00\x00\x40\x00\xc0")  # 0, +16384, -16384
    samples = transcribe.read_wav(path)
    assert samples.tolist() == [0.0, 0.5, -0.5]


def test_read_wav_refuses_audio_in_the_wrong_format(tmp_path: Path) -> None:
    """Resampling here would make this a second audio pipeline."""
    stereo = _write_wav(tmp_path / "stereo.wav", channels=2)
    with pytest.raises(ValueError, match="needs 16000 Hz, mono"):
        transcribe.read_wav(stereo)

    wrong_rate = _write_wav(tmp_path / "44k.wav", rate=44100)
    with pytest.raises(ValueError, match="44100 Hz"):
        transcribe.read_wav(wrong_rate)


def test_load_segments_without_a_transcript_says_which_stage_is_missing() -> None:
    with pytest.raises(FileNotFoundError, match="transcribe stage"):
        transcribe.load_segments("lec4")


# --- the subprocess hop -------------------------------------------------------


class FakePopen:
    """Stands in for the ASR child: records its argv, returns a given code."""

    def __init__(self, argv, code=0, stderr=b"", on_run=None, **_kwargs):
        self.argv = argv
        self._code = code
        self._stderr = stderr
        self.returncode = None
        self.killed = False
        if on_run is not None:
            on_run(self)

    def communicate(self, timeout=None):
        self.returncode = self._code
        return b"", self._stderr

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


def test_produce_segments_spawns_a_child_with_the_stage_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process exit is the only thing that reliably frees the card."""
    seen: list[FakePopen] = []
    transcribe.write_segments(
        "lec4", [transcribe.Segment(0.0, 2.0, "Done.", -0.2)], {"language": "pl"}
    )

    def popen(argv, **kwargs):
        child = FakePopen(argv, **kwargs)
        seen.append(child)
        return child

    monkeypatch.setattr(transcribe.subprocess, "Popen", popen)
    segments, reported = transcribe.produce_segments(
        "lec4", Path("/work/lec4/audio.wav"), "large-v3", "en"
    )

    argv = seen[0].argv
    assert argv[1:3] == ["-m", "app.transcribe"]
    assert "--slug" in argv and "lec4" in argv
    assert "large-v3" in argv
    assert "en" in argv
    assert segments[0].text == "Done."
    assert reported["language"] == "pl"


def test_a_child_that_fails_is_reported_with_its_last_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transcribe.subprocess,
        "Popen",
        lambda argv, **kwargs: FakePopen(
            argv, code=1, stderr=b"Traceback\nRuntimeError: no weights in the image\n"
        ),
    )
    with pytest.raises(RuntimeError, match="no weights in the image"):
        transcribe.produce_segments("lec4", Path("/a.wav"), "large-v3", "en")


def test_a_child_that_hangs_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess as sp

    class Hangs(FakePopen):
        def communicate(self, timeout=None):
            raise sp.TimeoutExpired("app.transcribe", timeout or 0)

    held = {}

    def popen(argv, **kwargs):
        held["child"] = Hangs(argv, **kwargs)
        return held["child"]

    monkeypatch.setattr(transcribe.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="did not finish within"):
        transcribe.produce_segments("lec4", Path("/a.wav"), "large-v3", "en")
    assert held["child"].killed


def test_kill_active_child_with_nothing_running_is_quiet() -> None:
    transcribe.kill_active_child()


def test_the_child_entry_point_writes_the_segments(
    monkeypatch: pytest.MonkeyPatch, on_the_cpu
) -> None:
    """What 'python -m app.transcribe' does, without spawning anything."""
    _install(monkeypatch, FakeModel([FakeSegment(0.0, 2.0, "W dziecku.")]))
    audio = _audio()
    code = transcribe.main(
        [
            "--slug",
            "lec4",
            "--audio",
            str(audio),
            "--model",
            "large-v3",
            "--language",
            "pl",
        ]
    )
    assert code == 0
    segments, reported = transcribe.load_segments("lec4")
    assert segments[0].text == "W dziecku."
    assert reported["device"] == "cpu"


def test_a_model_too_large_for_the_card_goes_to_the_cpu_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measured: large-v3 OOM'd with 4004 MB of a 4096 MB laptop card free."""
    monkeypatch.setattr(transcribe, "gpu_free_mb", lambda: 4004)
    device, _compute, reason = transcribe.choose_device("large-v3")
    assert device == "cpu"
    assert f"large-v3 needs {transcribe.needed_mb('large-v3')} MB" in reason


def test_an_unmeasured_model_gets_the_default_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(transcribe, "gpu_free_mb", lambda: 15000)
    assert transcribe.needed_mb("distil-large-v3") == config.ASR_MIN_FREE_DEFAULT_MB
    assert transcribe.choose_device("distil-large-v3")[0] == "cuda"
