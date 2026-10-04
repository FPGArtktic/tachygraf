# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""Paths, model names and tunables. The only place that reads the environment.

Copied from glosator's config.py and changed where a recording differs from a
book: the chunking tunables are measured in seconds of silence rather than in
pages, and there is an ASR model to name. Every number carries the reason for
its value, because the first real recording is expected to move several of
them (see docs/PILOT.md).
"""

from __future__ import annotations

import os
from pathlib import Path


def _path(env_name: str, default: str) -> Path:
    return Path(os.environ.get(env_name, default)).expanduser()


# Container mount points. Overridable so the test suite and host runs do not
# need /in and /out to exist.
#
# IN_DIR is the folder the recordings are read from, mounted read-only.
# OUT_DIR is a staging folder, never the vault: bin/tachygraf install refuses a
# NOTES argument that is, or lies inside, an Obsidian vault, and moving a
# finished recording into the vault is the user's step (see docs/DECISIONS.md).
IN_DIR: Path = _path("TACHYGRAF_IN", "/in")
OUT_DIR: Path = _path("TACHYGRAF_OUT", "/out")
WORK_DIR: Path = _path("TACHYGRAF_WORK", "/work")
STATE_DIR: Path = _path("TACHYGRAF_STATE", "/state")

LOG_DIR: Path = STATE_DIR / "logs"
DB_PATH: Path = STATE_DIR / "tachygraf.db"
PROMPT_DIR: Path = Path(__file__).parent / "prompts"

OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://ollama:11434")
TEXT_MODEL: str = os.environ.get("TACHYGRAF_TEXT_MODEL", "qwen2.5:32b")

# Written into every note's frontmatter as "prompt: N". Bumped by the commit
# that changes a file in prompts/, so a note can be traced to the prompt that
# produced it.
PROMPT_VERSION: int = 2

# Ollama request options, applied to every call. NUM_CTX has to hold the window
# text, the labels and the prompt's own rules, which are longer here than in
# glosator; the rest is glosator's, measured there.
NUM_CTX: int = 12288
TEMPERATURE: float = 0.25
NUM_PREDICT: int = 2048

# A 32B model offloaded to the CPU generates at a few tokens per second, so a
# single window can legitimately take twenty minutes. Fail only on a real hang.
# A timeout that fires on a healthy run is a bug in this number.
OLLAMA_TIMEOUT_S: int = 2700

# Keep the model resident between windows; reloading twenty gigabytes of
# weights per call would dominate the run time.
OLLAMA_KEEP_ALIVE: str = "30m"

# --- ASR -------------------------------------------------------------------

# faster-whisper model name. large-v3 is the accurate one; large-v3-turbo is
# roughly four times faster and loses detail on quiet or accented speech.
# Both are baked into the image, so switching is a flag, not a download.
ASR_MODEL: str = os.environ.get("TACHYGRAF_ASR_MODEL", "large-v3")

# Where the image put the CTranslate2 weights. Passed explicitly for the reason
# glosator learned with docling: the library's own default lives under $HOME,
# which is not the same directory under --userns=keep-id.
WHISPER_DIR: Path = _path("TACHYGRAF_WHISPER_DIR", "/opt/models/whisper")

# "auto" lets the model detect the language. Naming it is both faster and more
# accurate, and a wrong guess on the first thirty seconds of a Polish lecture
# produces a whole transcript in another language.
LANGUAGE: str = os.environ.get("TACHYGRAF_LANGUAGE", "auto")

# int8_float16 on the card: large-v3 is 1.5 GB that way, against 3.1 GB in
# float16, which does not fit a 4 GB laptop card beside anything else. On the
# CPU float16 is not implemented, so the fallback uses plain int8.
ASR_COMPUTE_GPU: str = "int8_float16"
ASR_COMPUTE_CPU: str = "int8"

# How much of the card a model needs before the stage will put it there. Per
# model, because the two differ by more than their weights do, and measured
# rather than reasoned: large-v3 at int8_float16 failed with "CUDA failed with
# error out of memory" on the reference machine with 4004 MB of a 4096 MB card
# free. Its weights are about 1.6 GB; the encoder activations and the cuDNN
# workspace are the rest, and on a 4 GB laptop card they do not fit.
#
# So on this machine large-v3 is a CPU model and large-v3-turbo is the one
# that runs on the card: turbo keeps the same encoder but has 4 decoder layers
# instead of 32, which is where both the size and the speed come from.
#
# A figure above the card's capacity is not a mistake here. It is how the
# stage says "not on this card" at the start, in one line in the log, instead
# of spending ninety seconds loading weights and then falling back anyway.
ASR_MIN_FREE_MB: dict[str, int] = {
    "large-v3": 5000,
    "large-v3-turbo": 2000,
}

# For a model nobody has measured. Enough for weights plus working space on
# something turbo-sized, and the fallback still catches it if that is wrong.
ASR_MIN_FREE_DEFAULT_MB: int = 2500

# Threads for the CPU path, which is the fallback and not the design point.
# Half the cores of the reference machine: the interface, the worker and
# whatever else the user is running want the rest, and this stage only has the
# machine to itself when it is on the card.
ASR_CPU_THREADS: int = 8

# Beam search width. 5 is faster-whisper's default and the figure every
# published accuracy number is quoted at.
ASR_BEAM_SIZE: int = 5

# Whisper hallucinates fluent sentences over silence, so the silero VAD gate
# runs first. This is the one place where the transcript is not a plain
# function of the audio, and it is on because the failure it prevents -- a
# paragraph of invented speech with a timestamp and a block ID -- is worse
# than the pause it may clip.
ASR_VAD_FILTER: bool = True

# How long the ASR child may take. The CPU fallback on a three-hour recording
# is the case this has to allow for, and it is slower than real time; a
# timeout that fires on a healthy run is a bug in this number, as with
# OLLAMA_TIMEOUT_S.
ASR_TIMEOUT_S: int = 21600

# A segment whose average token log probability is below this is marked
# "[unclear]" in the transcript, and the prompt forbids completing such a
# passage from the model's own knowledge. A guess until PILOT.md: healthy
# speech sits around -0.2, and anything under -1.0 is usually noise.
UNCLEAR_LOGPROB: float = -0.9

# --- transcript paragraphs -------------------------------------------------

# faster-whisper emits segments of a few seconds. One line each with its own
# timestamp reads as a subtitle file, not as prose, so segments are joined into
# a paragraph until a pause of PARAGRAPH_PAUSE_S or a length of
# PARAGRAPH_MAX_S, whichever comes first. Both are guesses until the first real
# transcript says how they read.
PARAGRAPH_PAUSE_S: float = 1.5
PARAGRAPH_MAX_S: float = 60.0

# --- windows ---------------------------------------------------------------

# One window is one note. Below MIN the window keeps growing; at or above MIN
# it is closed at the first real pause; at MAX it is closed wherever it is,
# because a window that never meets a pause would otherwise swallow the rest of
# the recording.
CHUNK_MIN_TOKENS: int = 3000
CHUNK_MAX_TOKENS: int = 6000

# A window boundary is only placed where the speaker stopped talking, so a note
# does not begin mid-sentence. Longer than PARAGRAPH_PAUSE_S on purpose: every
# paragraph break is a pause, and a window break should be a real one.
PAUSE_SPLIT_S: float = 2.5

# Rough token estimate for English and Polish technical prose. Only used to
# decide where a window ends, never to size a request exactly.
CHARS_PER_TOKEN: int = 4

# The subject of a recording, used in the prompts and as the first tag of every
# note. Recordings come from different fields, so this is per recording; the
# value here is only what an unset recording falls back to.
DEFAULT_DOMAIN: str = os.environ.get("TACHYGRAF_DOMAIN", "general")

# --- audio -----------------------------------------------------------------

# What faster-whisper wants: 16 kHz mono signed 16-bit PCM. Handing it anything
# else means it resamples internally, which is slower and silently lossy.
AUDIO_RATE: int = 16000
AUDIO_CHANNELS: int = 1

# ffmpeg on a two-hour recording is seconds of work, but a corrupt container
# can make it sit forever on a seek. Long enough for any real file.
FFMPEG_TIMEOUT_S: int = 1800
FFPROBE_TIMEOUT_S: int = 60

# How often the worker re-reads the job queue when it is empty.
WORKER_POLL_S: float = 2.0


def ensure_dirs() -> None:
    """Create the directories tachygraf writes to. IN_DIR is read-only."""
    for directory in (WORK_DIR, STATE_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
