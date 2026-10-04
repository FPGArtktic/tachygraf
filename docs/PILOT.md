# Pilot

Numbers measured on the reference machine, with the model and the input named.
Nothing here is an estimate. A figure that has not been measured yet says so
and stays empty until it has been, because the whole use of this file is that
CLAUDE.md can cite it instead of guessing.

Reference machine: Arch Linux, RTX 3050 Mobile (4 GB VRAM), 64 GB RAM, Ollama
on the host at `127.0.0.1:11434`.

## 2026-10-04 — The test suite and the extract stage

The suite, on the host, Python 3.12.14 in a `uv` virtualenv:

| measurement | value |
| --- | --- |
| tests | 198, all passing |
| wall clock | 2.5 s |
| needs a GPU | no |
| needs Ollama | no |
| needs the whisper weights | no |
| tests skipped without ffmpeg | 8 (marked `ffmpeg`) |

The extract stage on real material, the audio-only download of an
11½-minute English lecture on diodes (`.m4a`, AAC 128 kbit, one audio stream,
language tag `eng`):

| measurement | value |
| --- | --- |
| input | 10.7 MB `.m4a`, 00:11:32 |
| `ffprobe` duration, codec, stream count, language tag | all read correctly |
| `audio.wav` written | 16000 Hz, 1 channel, 16-bit, 22.1 MB |
| wall clock, probe and extract together | under 2 s |

The 4K AV1 `.mp4` of the same lecture (155.8 MB) probes to the same duration
and the same single AAC stream, so either file is a valid input and the
audio-only one is the cheaper one to hand over.

## 2026-10-04 — The image

Built with both ASR models and the `cuda` extra:

| measurement | value |
| --- | --- |
| `tachygraf:latest` | 11.24 GB |
| without the CUDA wheels | 6.62 GB |
| so the GPU path costs | 4.62 GB |
| test suite inside the image, `--network none` | 213 passing |

CLAUDE.md's open decision 1 estimated the GPU path at "roughly 1.5 GB" from
the wheel sizes. It is three times that: `nvidia-cublas-cu12` and
`nvidia-cudnn-cu12` unpack to 4.6 GB. The decision does not turn on it
anyway, for the reason the next section gives.

## 2026-10-04 — Transcription, and the answer to open decision 1

One recording: an 11½-minute English lecture on diodes, audio-only download
(`.m4a`, AAC 128 kbit, mono, language tag `eng`), 00:11:32.

| model | device | compute | wall clock | × real time | segments | `[unclear]` |
| --- | --- | --- | --- | --- | --- | --- |
| `large-v3-turbo` | GPU | `int8_float16` | **26 s** | 26× faster | 304 | 0 |
| `large-v3` | CPU, 8 threads | `int8` | **988 s** (16 min 28 s) | 1.4× faster | 193 | 0 |

`large-v3` does not appear on the GPU row because it does not fit. It failed
with `CUDA failed with error out of memory` with 4004 MB of the 4096 MB card
free, after about 95 seconds of loading: roughly 1.6 GB of int8 weights plus
encoder activations and cuDNN workspace is more than a 4 GB laptop card has.
That figure is now in `config.ASR_MIN_FREE_MB` per model, so the stage says
"not on this card" in one line instead of discovering it the slow way.

**Open decision 1 is settled: keep the GPU in the container.** The choice is
not the one the decision framed. It is not `large-v3` on the GPU against
`large-v3-turbo` on the CPU; on this card it is `large-v3-turbo` on the GPU at
26 seconds against `large-v3` on the CPU at 16½ minutes — a factor of 38 for
4.6 GB of image. On a recording of the length this tool is for, that is the
difference between a transcript arriving while you wait and one you come back
for.

Two things that follow:

- The default `TACHYGRAF_ASR_MODEL=large-v3` is the accurate model and the
  slow one *on this machine*. It is still the right default, because accuracy
  is what the transcript is for and the ASR stage is minutes against the
  generate stage's hours. A machine with more VRAM runs it on the card.
- Nothing was marked `[unclear]` in either run, on clean studio audio with a
  clear speaker. `UNCLEAR_LOGPROB` at -0.9 is therefore untested rather than
  validated: a recording of a real lecture hall is what will exercise it.

Accuracy, read against the publisher's own `.srt` for the same video:
tachygraf's transcript is at least as good. Where the subtitles have "a black
cylindrical body **as a** stripe at one end", `large-v3-turbo` has "a black
cylindrical body **that has a** stripe at one end", which is what was said;
it also resolves "theengineeringmindset.com" where the subtitles split it into
"Engineering mindset.com". Punctuation and sentence case are better
throughout. 2226 words against the subtitles' 2208.

## 2026-10-04 — Paragraphs and windows, on continuous speech

The same recording, `large-v3-turbo`:

| measurement | value |
| --- | --- |
| paragraphs | 12 |
| paragraph timestamps | 00:00:04, 01:02, 02:02, 03:00, 03:58, 04:57, 05:57, 06:36, 07:34, 08:33, 09:30, 10:29 |
| breaks from a pause ≥ `PARAGRAPH_PAUSE_S` | 1 (at 06:36) |
| breaks from `PARAGRAPH_MAX_S` | 10 |
| words per paragraph | ~185 |
| windows | 1 |
| window span | 5–685 s, 12 labels, ~3000 tokens |

This is the data for open decision 3, and it says the pause rule barely
applies to a scripted, continuously delivered lecture: eleven of the twelve
paragraph breaks are the 60-second cap, and only one is a real pause. The
publisher's subtitles agree — they contain 2 gaps of 1.5 s or more in 142
cues.

So the transcript is, in practice, uniform 60-second blocks of about 185
words. That reads as prose rather than as subtitles, which was the point, but
it is at the upper end of a readable block and nothing in the text marks where
the speaker actually paused. Lowering `PARAGRAPH_MAX_S` to 40 s would give
~17 paragraphs of ~125 words and finer-grained block references for the notes
to cite. Not changed yet: that judgement wants a recording of someone speaking
without a script, where the pause rule has something to work with.

One window, as predicted from the subtitle word count before the run: 2979
estimated tokens against `CHUNK_MIN_TOKENS` of 3000, so the window never
reached the point where a pause could close it. An 11½-minute lecture is one
note. `PAUSE_SPLIT_S` is therefore still unexercised.

## 2026-10-04 — Generation, one window, qwen2.5:32b

The same recording, the single window the chunk stage produced (12 paragraphs,
~3000 tokens of transcript), `qwen2.5:32b` on the host with the card free:

| measurement | value |
| --- | --- |
| wall clock, one window | **592 s** (9 min 52 s) |
| note length | 481 words |
| labels cited | 5 |
| claims dropped for having no label | 0 |
| glossary lines dropped for the wrong shape | 0 |
| responses the parser rejected | 0 |
| rule broken | none |

So the parser rejected nothing on the first real response, and nothing was
dropped. That is one window and not a sample, but it is evidence that the
seven-section contract is one a 32B model can meet.

Quality, reviewed against the transcript:

- Every claim cites a label that exists in the window, and the structure, the
  section order, the frontmatter and the absence of any timestamp are all as
  the contract specifies.
- The explanation is faithful: forward and reverse bias, the 0.7 V drop, the
  depletion region, p- and n-type doping and the bridge rectifier are all in
  the recording and are described the way the speaker described them.
- Two of the five claims say the same thing in different words ("the forward
  bias voltage ... is about 0.7 volts" and "the voltage drop ... is around 0.7
  volts") with different labels. Redundant, not wrong, and not something the
  parser can see.
- Five claims is the floor of the prompt's "5-15" for an eleven-minute window.
- **The title was the worst of it, and the glossary the most misleading.**
  Both produced prompt changes and `PROMPT_VERSION` 2; see docs/DECISIONS.md,
  "The model is asked for a title" and "The Polish glossary is the weakest
  thing the 32B produces". Three of seven Polish terms were calques of the
  English rather than the terms Polish textbooks use.

Open decision 6 asked for minutes per window and the rejection rate on
transcript input. Minutes per window: ten, for a window at the small end of
the budget. The rejection rate needs more than one window.

## Not yet measured

These are the numbers CLAUDE.md's open decisions turn on. Each needs one run.

- **Generation**, `qwen2.5:32b`: minutes per window, and the figure CLAUDE.md's
  open decision 6 asks for — how many responses the parser rejected and which
  rule each rejection broke. One window is not a sample; a 60-minute recording
  is still needed for a per-window average and for a rejection rate.
- **How often a model copies an example label** out of `note.md` rather than
  reading the window, which the parser rejects the whole response for.
- **`PAUSE_SPLIT_S`**, which a single-window recording never reaches. Needs a
  recording long enough to produce several windows.
- **`UNCLEAR_LOGPROB`**, which clean studio audio never triggered. Needs a
  recording made in a room.
- **`PARAGRAPH_MAX_S` at 40 s** against 60 s, on unscripted speech.
