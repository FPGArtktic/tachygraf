# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""The ASR stage: audio in, ``segments.json`` and ``_Transcript.md`` out.

This is the only stage that loads a model inside the container, and the only
one that asks for the card. Two rules hold it to that:

- The model is built inside ``transcribe_audio`` and dropped in its ``finally``,
  so the card is free again before Ollama is asked for anything. One model on
  the GPU at a time is not a convention here, it is what 4 GB of VRAM allows.
- Whether the card was available is recorded, not assumed. A stage that quietly
  degraded to the CPU and reported success is the docling failure glosator met
  once; here the device and the reason for it go into the log and the UI.

The transcript is verbatim. No model sees it between faster-whisper and the
file, nothing is cleaned up, and the only thing added to the speaker's words is
an ``[unclear]`` marker where the ASR model's own confidence says so.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import wave
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np

from app import config, generate, log, media

SEGMENTS_JSON = "segments.json"
TRANSCRIPT_NOTE = "_Transcript.md"
UNCLEAR = "[unclear]"


@dataclass(frozen=True)
class Segment:
    """One ASR segment: a few seconds of speech with a confidence score."""

    start: float
    end: float
    text: str
    avg_logprob: float

    @property
    def unclear(self) -> bool:
        return self.avg_logprob < config.UNCLEAR_LOGPROB


@dataclass(frozen=True)
class Paragraph:
    """A run of segments that reads as prose, and the block ID it carries.

    ``label`` is both the Obsidian block ID in the transcript and the only
    reference the note model is given. It never sees a time, so it cannot
    invent one (see docs/DECISIONS.md).
    """

    label: str
    start: float
    end: float
    text: str

    @property
    def unclear(self) -> bool:
        return UNCLEAR in self.text


def work_dir(slug: str) -> Path:
    return config.WORK_DIR / slug


def segments_path(slug: str) -> Path:
    return work_dir(slug) / SEGMENTS_JSON


# --- the model --------------------------------------------------------------


def gpu_free_mb() -> int | None:
    """Free VRAM on the first card, or ``None`` when the card cannot be asked.

    ``None`` is not zero and must not be read as "no GPU": it means nvml was
    not there to answer, which is also what a CPU-only host looks like. The
    caller treats both as a reason to use the CPU, and says which it was.
    """
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception as error:  # no driver, no CDI spec, no card
        log.event("transcribe", "nvml unavailable", error=repr(error))
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).free // 1_000_000)
    except Exception as error:
        log.event("transcribe", "could not read VRAM", error=repr(error))
        return None
    finally:
        # Failing to shut nvml down cannot be allowed to fail a job.
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


def needed_mb(model_name: str) -> int:
    """VRAM this ASR model needs, as measured. See config."""
    return config.ASR_MIN_FREE_MB.get(model_name, config.ASR_MIN_FREE_DEFAULT_MB)


def choose_device(model_name: str = config.ASR_MODEL) -> tuple[str, str, str]:
    """``(device, compute_type, reason)`` for this ASR model.

    The CPU is a correct answer, not a failure. It is the answer when the card
    is busy — Ollama may still hold the text model from the previous job — and
    also when the model is simply too large for the card, which on the
    reference machine is the case for large-v3 whatever else is running.

    The requirement is per model and measured, so a model that cannot fit says
    so in one line before it loads anything, rather than spending ninety
    seconds putting weights on the card and then falling back.
    """
    free = gpu_free_mb()
    if free is None:
        return "cpu", config.ASR_COMPUTE_CPU, "no usable NVIDIA card was found"
    needed = needed_mb(model_name)
    if free < needed:
        return (
            "cpu",
            config.ASR_COMPUTE_CPU,
            f"{free} MB of VRAM is free and {model_name} needs {needed} MB "
            "on this card",
        )
    return "cuda", config.ASR_COMPUTE_GPU, f"{free} MB of VRAM is free"


def load_model(name: str, device: str, compute_type: str):
    """Build a faster-whisper model from the weights baked into the image.

    Imported here rather than at module scope so that the test suite, which has
    neither the weights nor CTranslate2's CUDA libraries, can replace this
    function without importing them. ``local_files_only`` is what makes a
    missing model fail loudly instead of reaching for the network, which the
    container has none of.
    """
    from faster_whisper import WhisperModel

    return WhisperModel(
        name,
        device=device,
        compute_type=compute_type,
        cpu_threads=config.ASR_CPU_THREADS,
        download_root=str(config.WHISPER_DIR),
        local_files_only=True,
    )


def read_wav(path: Path) -> np.ndarray:
    """The WAV the extract stage wrote, as the array the model consumes.

    faster-whisper would decode this file itself if handed a path, through
    PyAV, and resample it again. It does not need to: ``media.extract_audio``
    has already produced exactly 16 kHz mono signed 16-bit PCM, which is what
    the model works in internally, and the standard library reads that.

    Doing it here takes PyAV out of the pipeline, and with it a version
    coupling that has already cost one run: faster-whisper 1.2.1 calls
    ``av.open(metadata_errors=...)``, a keyword PyAV 19 removed, so the stage
    died of a TypeError after loading 1.5 GB of weights onto the card. See
    docs/DECISIONS.md.
    """
    with wave.open(str(path), "rb") as handle:
        found = (handle.getframerate(), handle.getnchannels(), handle.getsampwidth())
        wanted = (config.AUDIO_RATE, config.AUDIO_CHANNELS, 2)
        if found != wanted:
            # Only reachable if something other than media.extract_audio wrote
            # it. Resampling here instead would make this module a second
            # audio pipeline with its own idea of the right format.
            raise ValueError(
                f"{path} is {found[0]} Hz, {found[1]} channel(s), "
                f"{found[2] * 8}-bit; the model needs {wanted[0]} Hz, mono, "
                "16-bit. Re-run the extract stage with --force."
            )
        frames = handle.readframes(handle.getnframes())
    # int16 to the float32 in [-1, 1) that CTranslate2 expects.
    return np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0


def _run_model(
    samples: np.ndarray,
    model_name: str,
    device: str,
    compute_type: str,
    language: str,
) -> tuple[list[Segment], object]:
    """One attempt on one device. The model is dropped before this returns.

    The segments are consumed here, inside, and not handed back as the
    generator faster-whisper returns: the generator is where the work actually
    happens, so a card that fails mid-recording fails in this function, where
    the caller can still do something about it.
    """
    model = load_model(model_name, device, compute_type)
    try:
        raw_segments, info = model.transcribe(
            samples,
            language=None if language in ("", "auto") else language,
            beam_size=config.ASR_BEAM_SIZE,
            vad_filter=config.ASR_VAD_FILTER,
            word_timestamps=False,
        )
        segments = [
            Segment(
                start=float(segment.start),
                end=float(segment.end),
                text=segment.text.strip(),
                avg_logprob=float(segment.avg_logprob),
            )
            for segment in raw_segments
            if segment.text.strip()
        ]
    finally:
        # The card goes back before anything else is asked of it. CTranslate2
        # frees the device memory when the model is collected, and nothing else
        # in this process holds a reference to it.
        del model
    return segments, info


def transcribe_audio(
    audio: Path,
    model_name: str = config.ASR_MODEL,
    language: str = config.LANGUAGE,
) -> tuple[list[Segment], dict[str, object]]:
    """Run the ASR model over the extracted WAV. Returns segments and the report.

    The GPU is tried when nvml says there is room, and one failure of any kind
    on the card sends the whole attempt to the CPU instead of failing the job.
    nvml answering "free" is not a promise: the card may be claimed a moment
    later, the driver may not match, or the CUDA libraries may not be loadable
    at all, and the last of those has already happened once in this project
    (see docs/DECISIONS.md). A recording transcribed slowly is worth having; a
    job that failed after an hour of queueing is not.
    """
    device, compute_type, reason = choose_device(model_name)
    log.event(
        "transcribe",
        "device chosen",
        device=device,
        compute_type=compute_type,
        reason=reason,
        model=model_name,
    )
    samples = read_wav(audio)

    try:
        segments, info = _run_model(samples, model_name, device, compute_type, language)
    except Exception as error:
        if device == "cpu":
            raise
        # Said loudly, with what went wrong: a stage that quietly degraded and
        # reported success is the docling failure glosator met once.
        reason = f"the card was reported free but could not be used: {error}"
        log.event(
            "transcribe",
            "the GPU attempt failed; falling back to the CPU",
            error=repr(error),
            model=model_name,
        )
        device, compute_type = "cpu", config.ASR_COMPUTE_CPU
        segments, info = _run_model(samples, model_name, device, compute_type, language)

    reported = {
        "language": getattr(info, "language", None),
        "language_probability": round(
            float(getattr(info, "language_probability", 0.0) or 0.0), 3
        ),
        "duration_s": float(getattr(info, "duration", 0.0) or 0.0),
        "device": device,
        "compute_type": compute_type,
        "device_reason": reason,
        "model": model_name,
    }
    log.event(
        "transcribe",
        "transcribed",
        segments=len(segments),
        unclear=sum(1 for segment in segments if segment.unclear),
        **reported,
    )
    return segments, reported


# --- the subprocess hop ------------------------------------------------------

# The child currently transcribing, so that a worker being shut down takes it
# with it instead of leaving it holding the card.
_child: subprocess.Popen | None = None


def kill_active_child() -> None:
    """Kill the ASR child, if one is running. Called from the worker's SIGTERM.

    Without this, "Stop now" kills the worker and orphans a child that holds
    1.5 GB of VRAM and keeps decoding a recording nobody is waiting for.
    """
    process = _child
    if process is not None and process.poll() is None:
        process.kill()
        log.event("transcribe", "ASR child killed")


def produce_segments(
    slug: str,
    audio: Path,
    model_name: str,
    language: str,
) -> tuple[list[Segment], dict[str, object]]:
    """Transcribe in a child process, which is what actually frees the card.

    CTranslate2 keeps a caching CUDA allocator: dropping the model hands its
    memory back to that cache rather than to the driver, and the worker
    process outlives the stage by design, because it goes on to the next job.
    Measured on the reference machine: after one ASR attempt the worker still
    held 2008 MiB of a 4096 MiB card while the stage itself had moved to the
    CPU. That is the generate stage's VRAM, and it is what the next job's
    ``choose_device`` would see and misread as room.

    ``del model`` in a ``finally`` is therefore not enough, and no API call is:
    process exit is the only thing that returns device memory
    unconditionally. So the model is loaded in a child that writes
    ``segments.json`` and exits, and this function reads the file back. It is
    the same shape as the rest of the pipeline, where a stage hands the next
    one a file rather than an object.

    Not a second entrypoint: ``python -m app.transcribe`` is internal, exactly
    as ``python -m app.worker`` already is. ``bin/tachygraf`` remains the only
    way in. See docs/DECISIONS.md.
    """
    global _child
    argv = [
        sys.executable,
        "-m",
        "app.transcribe",
        "--slug",
        slug,
        "--audio",
        str(audio),
        "--model",
        model_name,
        "--language",
        language,
    ]
    log.event("transcribe", "starting the ASR child", slug=slug, model=model_name)
    try:
        _child = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _stdout, stderr = _child.communicate(timeout=config.ASR_TIMEOUT_S)
        code = _child.returncode
    except subprocess.TimeoutExpired as error:
        kill_active_child()
        raise RuntimeError(
            f"the ASR child did not finish within {config.ASR_TIMEOUT_S} s"
        ) from error
    finally:
        _child = None

    if code != 0:
        # The child's own last words. It logs to the same file, so the detail
        # is already there; this is what reaches the job's error column.
        tail = (stderr or b"").decode("utf-8", errors="replace").strip()
        last = tail.splitlines()[-1] if tail else "no output"
        raise RuntimeError(f"the ASR child failed ({code}): {last}")

    return load_segments(slug)


# --- segments on disk -------------------------------------------------------


def write_segments(
    slug: str, segments: list[Segment], reported: dict[str, object]
) -> Path:
    """Store the per-segment timing, which the transcript itself does not keep.

    The transcript carries one timestamp per paragraph so that it reads as
    prose; this file is where a second run, a subtitle export or a look at what
    the model was unsure about gets the detail back.
    """
    path = segments_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "asr": reported,
        "segments": [asdict(segment) for segment in segments],
    }
    generate.write_atomic(
        path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    return path


def load_segments(slug: str) -> tuple[list[Segment], dict[str, object]]:
    path = segments_path(slug)
    if not path.exists():
        raise FileNotFoundError(
            f"no segments at {path}; run the transcribe stage for {slug} first"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = [
        Segment(
            start=float(record["start"]),
            end=float(record["end"]),
            text=record["text"],
            avg_logprob=float(record["avg_logprob"]),
        )
        for record in payload["segments"]
    ]
    return segments, payload.get("asr", {})


# --- paragraphs -------------------------------------------------------------


def label_for(position: int) -> str:
    """``1`` -> ``"p0001"``. Four digits hold a nine-hour recording."""
    return f"p{position:04d}"


def paragraphs(segments: list[Segment]) -> list[Paragraph]:
    """Join segments into paragraphs, and label each one.

    A paragraph ends at a pause of ``PARAGRAPH_PAUSE_S`` or after
    ``PARAGRAPH_MAX_S``, whichever comes first. Both numbers are guesses until
    a real transcript says how they read; they are tunables for that reason.

    The labels this assigns are the transcript's block IDs and the only
    references the note model is allowed to cite, so this function is the one
    place that decides what a paragraph is. Calling it twice on the same
    segments must give the same labels, which is why it takes the segments and
    nothing else.
    """
    built: list[Paragraph] = []
    current: list[Segment] = []

    def close() -> None:
        if not current:
            return
        text = " ".join(_marked(segment) for segment in current)
        built.append(
            Paragraph(
                label=label_for(len(built) + 1),
                start=current[0].start,
                end=current[-1].end,
                text=" ".join(text.split()),
            )
        )
        current.clear()

    for segment in segments:
        if current:
            gap = segment.start - current[-1].end
            length = segment.end - current[0].start
            if gap >= config.PARAGRAPH_PAUSE_S or length > config.PARAGRAPH_MAX_S:
                close()
        current.append(segment)
    close()
    return built


def _marked(segment: Segment) -> str:
    """The segment's words, with an ``[unclear]`` marker if ASR was unsure.

    The marker is an annotation, not an edit: the words stay exactly as they
    came out, and the prompt forbids completing a passage that carries it. A
    guess dressed up as a sentence the speaker said is the one thing a verbatim
    transcript must not contain.
    """
    return f"{segment.text} {UNCLEAR}" if segment.unclear else segment.text


# --- the transcript note ----------------------------------------------------


def transcript_markdown(
    built: list[Paragraph],
    *,
    source_name: str,
    duration_s: float,
    language: str,
    model: str,
    generated: date | None = None,
) -> str:
    """``_Transcript.md`` exactly as the output contract specifies."""
    stem = Path(source_name).stem
    lines = [
        "---",
        f"generator: {generate.GENERATOR}",
        f'source: "{source_name}"',
        f'duration: "{media.hms(duration_s)}"',
        f"language: {language or 'unknown'}",
        f"model: {model}",
        f"generated: {(generated or date.today()).isoformat()}",
        "---",
        f"# Transcript — {stem}",
        "",
    ]
    for paragraph in built:
        lines.append(
            f"**[{media.hms(paragraph.start)}]** {paragraph.text} ^{paragraph.label}"
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def transcript_path(slug: str, out_dir: Path | str | None = None) -> Path:
    return generate.source_dir(slug, out_dir) / TRANSCRIPT_NOTE


def run(
    slug: str,
    media_path: Path,
    *,
    model: str = config.ASR_MODEL,
    language: str = config.LANGUAGE,
    out_dir: Path | str | None = None,
    duration_s: float | None = None,
) -> Path:
    """The transcribe stage: WAV to ``segments.json`` and ``_Transcript.md``."""
    audio = media.audio_path(slug)
    if not audio.exists():
        raise FileNotFoundError(
            f"no audio at {audio}; run the extract stage for {slug} first"
        )

    segments, reported = produce_segments(slug, audio, model, language)
    if not segments:
        # Silence, the wrong audio track, or a VAD gate that closed over the
        # whole file. Either way there is nothing to make notes from, and an
        # empty transcript would be written over a good one on a rerun.
        raise ValueError(
            f"{slug}: the ASR model returned no speech from {audio}; check that "
            "the recording has audio and that the right track was extracted"
        )
    write_segments(slug, segments, reported)

    built = paragraphs(segments)
    note = transcript_markdown(
        built,
        source_name=media_path.name,
        duration_s=duration_s or float(reported.get("duration_s") or 0.0),
        language=str(reported.get("language") or language),
        model=model,
    )
    generate.claim_dir(generate.source_dir(slug, out_dir), out_dir)
    path = generate.write_note(transcript_path(slug, out_dir), note, out_dir)
    log.event(
        "transcribe",
        "transcript written",
        slug=slug,
        path=str(path),
        paragraphs=len(built),
        unclear_paragraphs=sum(1 for paragraph in built if paragraph.unclear),
    )
    return path


# --- the child ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """``python -m app.transcribe``: load the model, write segments.json, exit.

    Internal, and the whole point of it is the exit: see ``produce_segments``.
    """
    parser = argparse.ArgumentParser(
        prog="app.transcribe",
        description="internal: the ASR stage, in its own process",
    )
    parser.add_argument("--slug", required=True)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--model", default=config.ASR_MODEL)
    parser.add_argument("--language", default=config.LANGUAGE)
    args = parser.parse_args(argv)

    config.ensure_dirs()
    segments, reported = transcribe_audio(args.audio, args.model, args.language)
    write_segments(args.slug, segments, reported)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
