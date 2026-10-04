# tachygraf

tachygraf turns a recorded lecture into a verbatim transcript and a set of
linked study notes. It runs entirely on the local machine: no cloud API, no
telemetry, and no network access beyond a local Ollama instance.

*Tachygraf* (Polish, from the Greek *tachygraphos*, "fast writer") is the
ancient shorthand clerk who took speech down as it was spoken. That is the
function of this program. It is the sibling of
[glosator](https://github.com/FPGArtktic/glosator), which does the same for a
book: glosator writes glosses on a text, tachygraf takes down a recording.

Given a media file, it writes one transcript with a timestamp and an Obsidian
block ID on every paragraph, then one study note per window of the recording,
each claim in the note linking back to the paragraph where it was said. The
notes are plain files with YAML frontmatter and `[[wikilinks]]`: Obsidian reads
them, and so do ordinary command-line tools.

## Example output

The transcript. Both the lecture and its content are invented for this
illustration.

```
---
generator: tachygraf
source: "lecture-04.mp4"
duration: "01:12:33"
language: en
model: large-v3
generated: 2026-10-04
---
# Transcript — lecture-04

**[00:12:40]** So the question everyone asks at this point is why we cannot
simply filter after sampling. And the answer is that once the alias is in the
band, it is indistinguishable from signal. ^p0042

**[00:13:21]** Let me write the condition on the board. ^p0043
```

A note, one of several for the same recording:

```
---
generator: tachygraf
prompt: 1
source: "lecture-04.mp4"
span: "00:12:40 - 00:21:05"
model: qwen2.5:32b
generated: 2026-10-04
tags: [signal-processing, lec4]
---
# 03 Why filtering after sampling cannot undo aliasing

## In short
Sampling below twice the highest frequency folds content back into the band
of interest. The speaker's point is that this is not a filtering problem: the
alias occupies the same frequencies as the wanted signal.

## Explanation
Plain-language prose, following the order in which the speaker developed the
material.

## Key claims
- A filter applied after sampling cannot remove an alias, because the alias
  sits in the same band as the signal. ([[_Transcript#^p0042]])
- The sampling condition was written on the board rather than spoken; the
  transcript records only that it was written. ([[_Transcript#^p0043]])

## Formulas
None in this window.

## Glossary
- aliasing — aliasing
- sampling rate — częstotliwość próbkowania

---
[[02 Previous window]] · [[04 Next window]]
```

The transcript is in the language that was spoken. The note body is in English
and the glossary carries the Polish equivalents, for the reason glosator
measured: roughly 40 percent fewer output tokens than Polish prose, with the
terminology needed to look a subject up in either language.

## Requirements

- Podman, rootless. Nothing is installed on the host.
- [Ollama](https://ollama.com) on the host with a text model, by default
  `qwen2.5:32b`. The container reaches it the way glosator does, with
  `pasta --map-host-loopback`; nothing has to be changed on the Ollama side.
- An NVIDIA GPU with a CDI spec, for the transcription stage:
  `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`. Without it,
  transcription falls back to the CPU and says so. The reference machine is a
  laptop with 4 GB of VRAM; the text model runs mostly on the CPU, slowly.
  That is the design point rather than a constraint to be worked around.

## Quick start

Build the image, then install the systemd unit for the two folders you want
to use: the folder your recordings are in, and the folder the output goes to.

```
bin/tachygraf build
bin/tachygraf install ~/Videos/lectures ~/Documents/tachygraf-out
```

The recordings folder is mounted read-only. The output folder is a staging
area, not your vault: `install` refuses a path that is, or lies inside, an
Obsidian vault. When a recording is done, move its folder into the vault by
hand. See [Why the output folder is not the vault](#why-the-output-folder-is-not-the-vault).

```
bin/tachygraf start
```

This starts the container, waits for the server and opens
`http://127.0.0.1:7861`. In the browser: pick a recording, set the language
and the two models, and queue the stages. Progress, the log and the most
recent note are visible while the job runs; closing the browser does not stop
it.

The same work can be done from the command line. Paths are the ones inside the
container, so a recording sits in `/in` and the output goes to `/out`.

```
bin/tachygraf cli add "/in/lecture-04.mp4" --slug lec4 --domain "signal processing"
bin/tachygraf cli transcribe --source lec4 --language en
bin/tachygraf cli chunk      --source lec4
bin/tachygraf cli generate   --source lec4
bin/tachygraf cli index      --source lec4
bin/tachygraf cli queue
```

### A whole course at once

A folder of recordings becomes one set of notes per recording, each in its own
subfolder of the output folder. `--dry-run` prints the plan and queues nothing,
which is worth doing first.

```
bin/tachygraf cli batch /in/kurs-elektroniki --domain electronics --dry-run
bin/tachygraf cli batch /in/kurs-elektroniki --domain electronics --number
```

The folder is searched including its subfolders, because a downloaded playlist
is often one folder per recording. The same lecture in two containers counts
once: a video and the audio-only track beside it have the same title and the
same length, and the audio one is used, since the audio is identical and a
tenth of the size. A file with no audio stream — a thumbnail that happens to
be an `.mp4` — is named and skipped rather than failing the batch.

Recordings are ordered the way a person reads them, so `lekcja-9` comes before
`lekcja-10`. `--number` prefixes each slug with its position so the course
keeps that order in any listing, and `--prefix` namespaces the whole course.

The five stages of one recording are queued together, so the first lecture's
notes are finished before the second lecture's transcription starts. Stopping
a batch halfway leaves finished lectures behind rather than a pile of
transcripts.

When the work is done, stop it. `stop` also unloads the text model from
Ollama, so the GPU is free for whatever else uses it.

```
bin/tachygraf stop
```

`bin/tachygraf status` says whether it is running and what currently holds the
GPU; `bin/tachygraf logs` follows the container log. `bin/tachygraf uninstall`
stops it and removes the systemd unit, and deletes nothing else.

tachygraf is not a service. Nothing starts at login, nothing restarts itself,
and the models leave the GPU when a job ends. Within a single job the text
model stays resident, because reloading twenty gigabytes of weights for every
window would dominate the running time.

## How it works

Five stages. Extraction and transcription produce the transcript; chunking
and generation produce the notes; the index links them.

| stage      | model | description |
| ---------- | ----- | ----------- |
| extract    | none  | ffprobe for the duration, ffmpeg to 16 kHz mono WAV. Seconds. |
| transcribe | faster-whisper, GPU | segments with timestamps and a confidence score; writes `_Transcript.md` with one block ID per paragraph. Minutes. |
| chunk      | none  | windows of a few thousand tokens, cut only at a pause, one paragraph of overlap. |
| generate   | Ollama | one model call per window, producing one note. Every claim must cite a paragraph label that exists in the window; the parser checks. |
| index      | none  | the map of content, the merged glossary, previous/next links. |

Each stage records its completed work in SQLite, keyed on the tuple
`(source, chunk, stage, model)`. The container can be killed at any point: the
next run resumes rather than starting again, a window that failed is retried,
and a window that succeeded is not rewritten.

Two models never share the card. Per job, in this order: ffmpeg on the CPU,
the ASR model on the GPU, released; then Ollama, unloaded when the job ends.

## Why the output folder is not the vault

glosator may write into a vault and guards itself with three rules enforced in
code. tachygraf keeps those rules and adds a fourth, enforced before the
container starts: it is never given the vault path at all. The output goes to
a staging folder; moving a finished recording into the vault is a step the
user takes, by hand, with the notes in front of them.

The three inherited rules still hold inside the staging folder, because a
staging folder can hold things that matter too:

- Every generated file carries `generator: tachygraf` in its frontmatter. A
  file without that marker is never overwritten, and the index stage skips it.
- Every target path is resolved before use and refused if it falls outside the
  output folder.
- Directories created by tachygraf are stamped. A non-empty directory that it
  did not create is left untouched.

Nothing is deleted at any point. Files are written to a temporary name and
renamed into place, so an interruption leaves either the previous file or the
new one, never a partial one.

## Configuration

Configuration is confined to `app/config.py` and the environment.

| variable               | default                | meaning |
| ---------------------- | ---------------------- | ------- |
| `TACHYGRAF_IN`         | `/in`                  | folder the recordings are read from, mounted read-only |
| `TACHYGRAF_OUT`        | `/out`                 | staging folder the output is written to |
| `OLLAMA_URL`           | `http://ollama:11434`  | address of the Ollama instance |
| `TACHYGRAF_TEXT_MODEL` | `qwen2.5:32b`          | model used for the notes |
| `TACHYGRAF_ASR_MODEL`  | `large-v3`             | faster-whisper model; `large-v3-turbo` is the fast one |
| `TACHYGRAF_LANGUAGE`   | `auto`                 | spoken language, or `auto` to let the ASR model decide |
| `TACHYGRAF_DOMAIN`     | `general`              | subject assumed for a recording that specifies none |

## Development

```
uv venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

The test suite requires neither a GPU, nor a running Ollama, nor the whisper
weights. Tests that need the ffmpeg binary are marked and skip without it. The
container runs the same suite without network access:

```
podman run --rm --network none -v .:/src:ro -w /src tachygraf:latest pytest
```

`docs/DECISIONS.md` records the reasoning behind the design, including what
was taken from glosator and why, and the points at which this project departs
from it.

## Out of scope

Speaker diarization, OCR of slides or key frames, subtitle files, translation,
downloading from the network, retrieval-augmented generation, vector
databases, embeddings, cloud APIs and Windows support. This is a batch job,
not a retrieval system, and the recording is already on disk.

## License

GPL-3.0-or-later; see [LICENSE](LICENSE).

The transcripts and notes produced by the program are not covered by that
license. They belong to whoever ran it.
