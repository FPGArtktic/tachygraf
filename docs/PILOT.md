# Pilot

Numbers measured on the reference machine, with the model and the input named.
Nothing here is an estimate. A figure that has not been measured yet says so
and stays empty until it has been, because the whole use of this file is that
CLAUDE.md can cite it instead of guessing.

Reference machine from 2026-10-30: Arch Linux (CachyOS kernel), RTX 5070 Ti
(16 GB VRAM, compute capability 12.0, driver 615.78), Ryzen 7 7800X3D, 32 GB
RAM, Ollama on the host at `127.0.0.1:11434`. The desktop is drawn on the
NVIDIA card and holds 1.3–1.5 GB of it.

The entries dated 2026-10-04 were measured on the previous one: a laptop with
an RTX 3050 Mobile (4 GB VRAM) and 64 GB RAM. They stay as they were measured;
where a later entry supersedes one, it says so.

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

## 2026-10-30 — Transcription on the RTX 5070 Ti, and the compute type

The image, rebuilt for this machine with the `cuda` extra's new lower bounds:

| measurement | value |
| --- | --- |
| `tachygraf:latest` | 11.3 GB |
| ctranslate2, faster-whisper | 4.8.4, 1.2.1 |
| nvidia-cublas-cu12, nvidia-cudnn-cu12 | 12.9.2.10, 9.27.0.42 |
| tests on the host, Python 3.12 | 249 passing |

The same recording as on 2026-10-04: the 11½-minute English diodes lecture,
`.m4a`, 692 s. Each row is one run in a fresh container through the stage's
own `_run_model`, language `en`, beam 5, VAD on; the wall clock includes
loading the model. VRAM is the card's used memory sampled every 100 ms, peak
minus what the desktop held before the run. "Repeated 6-grams" counts six-word
sequences that occur more than once; the speaker's own repetition puts a
clean transcript of this lecture at 8 or 9.

| model | compute | wall clock | × real time | segments | words | peak VRAM | repeated 6-grams |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `large-v3-turbo` | `int8_float16` | 27.8 s | 25× | 309 | 2173 | 1423 MiB | 9 |
| `large-v3-turbo` | `float16` | **26.5 s** | 26× | 310 | 2198 | 2516 MiB | 9 |
| `large-v3` | `int8_float16` | 60.9 s | 11× | 191 | 2311 | 4085 MiB | **63** |
| `large-v3` | `float16` | **57.2 s** | 12× | 167 | 2210 | 5517 MiB | 8 |

What the table says:

- **`int8_float16` is wrong on this card, and silently.** `large-v3` produced
  repetition loops the audio does not contain — "The electrons are also
  called electrons. The electrons are also called", "the diode acts as an
  intrusion. Therefore the diode acts as an intrusion", a migration sentence
  three times over — and no exception, so the CPU fallback had nothing to
  catch. `large-v3-turbo` at `int8_float16` dropped the sentence "…this side
  when acting as an insulator. You can see that the diode can only act as an
  insulator…", which float16 kept. On the laptop's Ampere card the same model
  and compute type gave 2198 words and 9 repeated 6-grams, so it is this
  card's int8 path that fails. `ASR_COMPUTE_GPU` is now `float16`; see
  docs/DECISIONS.md.
- **`large-v3` is no longer the slow choice.** 57 seconds on the card against
  988 seconds on the laptop's CPU. Open decision 1's trade — accuracy against
  waiting — is gone on this machine: the accurate model costs half a minute
  more than turbo on an eleven-minute recording.
- `large-v3-turbo` takes the same 26–28 s here as on the laptop's 3050. On a
  recording this short the wall clock is mostly loading the model and the
  VAD pass, not inference; a 60-minute recording is what will separate the
  cards.
- `config.ASR_MIN_FREE_MB` is these peaks converted to the MB `gpu_free_mb`
  reports (5785 MB and 2638 MB) plus a margin: 6500 and 3000.
- `large-v3` at float16 has one error the others do not: the closing
  sentence's list of social networks comes out as "Facebook, Instagram,
  Twitter and Facebook. You can also follow us on Twitter". Nothing in the
  lecture itself.

## 2026-10-30 — Generation on the RTX 5070 Ti, three text models on one window

The window the laptop measured on 2026-10-04: the diodes recording, one window
of 12 paragraphs (`work/dio`), domain "diodes", prompt version 2. Each model
went through the stage's own `generate.run` — the prompt, the parser,
`assemble` and the write guards — with the Ollama call replaced by a copy that
keeps the response's timing fields. `qwen3:32b` was called with thinking off
(`think: false`), which the stage does not do today; with thinking on, its
reasoning would spend `NUM_PREDICT`. Ollama 0.40.13, `ollama-cuda` from Arch,
`num_ctx` 12288, the card otherwise idle with the desktop on it.

| model | layers on the card | in VRAM / total | load | prompt | output | tokens/s | wall clock | rejected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5:32b` | 39/65 | 13.0 / 21.8 GiB | 12.1 s | 4252 tok, 6.1 s | 733 tok | 5.66 | **148 s** | 0 |
| `gemma3:27b` | 44/63 | 11.7 / 16.4 GiB | 10.3 s | 4237 tok, 4.6 s | 1604 tok | 8.76 | **198 s** | 0 |
| `qwen3:32b` | 39/65 | 13.2 / 22.1 GiB | 10.0 s | 4239 tok, 6.4 s | 1204 tok | 5.15 | **250 s** | 0 |

- `qwen2.5:32b` took 148 s for the window the laptop took 592 s for: four
  times faster, with 39 of its 65 layers on the card.
- The run is bound by the layers left on the CPU. GPU utilisation sampled
  every 2 s stayed between 0 and 18 percent while the card's memory was full
  (14.7 GiB of 15.9 at the peak).
- RAM is not a constraint at 32 GB. The weights are memory-mapped, so the part
  on the CPU sits in the page cache: memory used by all processes together
  peaked at 6.4 GiB, and available memory never fell below 24 GiB.
- `OLLAMA_TIMEOUT_S` (2700) stays. A maximal window, 6000 tokens in and 2048
  out at 5 tokens per second, is under ten minutes on this card, so the
  number is now very generous, and that is its job.

The three notes, reviewed against the transcript by one reviewer (Claude),
on one window of an English lecture. That is evidence, not a sample.

| | `qwen2.5:32b` | `gemma3:27b` | `qwen3:32b` |
| --- | --- | --- | --- |
| claims | 6 | 14 | 11 |
| reversed directions | 0 | 0 | 0 |
| content not in the source, or changed | 3 | 2 | 3 |
| citations to a paragraph that does not hold the claim | 1 | ~10 | 2 |
| claims copied verbatim without quotation marks | 0 | 0 | 7 |
| glossary lines wrong, of those the prompt does not supply | 5 of 9 | 3 of 9 | 2 of 14 |

The content errors, each against the paragraph that says otherwise:

- `qwen2.5:32b`: forward bias "reduces the barrier voltage" (the speaker only
  says the source must exceed the 0.7 V barrier, p0008); reverse bias
  "increases the barrier voltage" (the speaker: "the barrier expands", p0008);
  the 0.7 V of the depletion region becomes "the typical forward voltage drop
  for a silicon diode" (p0007; the speaker measures the drop separately, 0.71 V,
  p0009). The last is the error glosator's rule names: a number moved to
  another quantity.
- `gemma3:27b`: silicon has "nearly full valence shells" (the speaker: one
  electron too many to be a conductor, four of eight, p0004–p0005); the
  multimeter's diode-test reading is called a voltage drop (p0012 keeps the
  0.516 V opening voltage apart from the 0.71 V drop).
- `qwen3:32b`: diodes are used "in signal processing" (not said); in diode
  test mode the meter "applies a small voltage and measures the resulting
  current" (not said, and backwards: the mode reports a voltage); a healthy
  diode shows "0.5 to 0.7 volts" (a range the speaker never gives, made by
  merging 0.516 V and 0.71 V, which he measured as two different things).

The glossaries, against established Polish terms:

- `qwen2.5:32b`: "polaryzacja przódowa", "doping p-rodzajowy", "dopping
  n-rodzajowy", "łącze p-n" (złącze p-n), "opad napięcia" (spadek napięcia).
  Five invented terms of nine, after `PROMPT_VERSION` 2 added the rule against
  calques: on this window the rule did not work for this model.
- `gemma3:27b`: "(brak odpowiednika)" for anode and cathode, which are "anoda"
  and "katoda"; "prąd konwencjonalny", a calque of "umowny kierunek prądu".
  The other eight are right. It fails towards the admitted gap, which is the
  safer direction, though not for terms this basic.
- `qwen3:32b`: "polaryzacja w przód", a calque; "pas przewodnictwa" for
  "pasmo przewodnictwa"; and "polaryzacja wsteczna" for reverse bias, where
  the prompt itself gives "polaryzacja zaporowa" as the answer. Thirteen of
  sixteen are right, including "domieszkowanie typu p" and "wiązanie
  kowalencyjne".

What it says, as far as one window can:

- No model reversed a direction. That was the failure that ruled out
  `gemma3:12b` in glosator; the 27B did not show it here.
- `gemma3:27b` is the most faithful: its two errors are the smallest, and it
  adds nothing from outside the recording. Its weakness is citation: its
  explanation cites a label on nearly every sentence, and about ten of them
  point at the paragraph before or after the one that holds the claim.
  Several fall where `PARAGRAPH_MAX_S` cut a sentence in two ("This side is
  how it should perform" / "when acting as a conductor"), which is evidence
  for open decision 3 as much as against the model.
- `qwen3:32b` writes the best Polish and the most outside knowledge, one item
  of it wrong. For a study note that is the worse trade.
- `qwen2.5:32b` is the fastest and the thinnest: six claims, three errors, and
  a glossary that is mostly invented.

Nothing is changed on this. The default stays `qwen2.5:32b` until a longer
recording has been through `gemma3:27b` and `qwen3:32b` too, reviewed the
same way.

## 2026-10-30 — A 28-minute talk end to end, and the speech model compared

The first recording long enough to have more than one window:
`claude-engineer.mp4`, 00:27:52, H.264 1080p with stereo AAC, no language tag.
A conference talk about Claude Code, in English, unscripted, with questions
from the audience at the end. Domain "applied AI".

**The pipeline, end to end, on the RTX 5070 Ti.** `large-v3` at float16
detected English at 0.995 and transcribed the talk in 97 s. The chunk stage
then closed its first window at 6191 tokens against a budget of 6000, which
the generate stage refuses: the budget was checked after a paragraph was
added. Fixed in `chunk.group_paragraphs`; the plan became two windows of 5939
and 1668 tokens. The speaker made no pause of `PAUSE_SPLIT_S` (2.5 s) in the
first 21 minutes, so it was the budget that closed the window: the first
time `PAUSE_SPLIT_S` was exercised, and it never fired. `qwen2.5:32b` wrote
the two notes in 143 s and 107 s; `gemma3:27b`, run on the same windows for
comparison, in 154 s and 102 s.

**`[unclear]` fired for the first time.** Nine segments between 804 and
834 s fell below `UNCLEAR_LOGPROB`, all at exactly -1.04, and the text there
was invented ("on behalf of 20 moders", a Chinese character). Every segment in
a stretch carries the same log probability — -0.21 throughout 248-264 s,
-0.25 throughout 793-803 s — so the marker covers whole stretches, including
the real sentences in them ("these are common bash commands. We have a style
guide.").

**Whisper settings and Parakeet, on the same audio.** Whisper is `large-v3`,
float16, beam 5, VAD on. Parakeet is NVIDIA's `parakeet-tdt-0.6b-v3` in
NeMo 3.0.0 with torch 2.11 (cu128), float32, local attention, in a separate
image. "Repeated" counts six-word sequences that occur more than once; the
speaker's own repetitions put a clean transcript of this talk at about 8.
"Loops" is what `transcribe.looping` marks. "Claude Code" counts the correct
spelling of the product's name.

| run | words | repeated | loops | "Claude Code" | "Cloud Code" | ASR time |
| --- | --- | --- | --- | --- | --- | --- |
| Whisper, default | 5552 | 47 | 2 (263 s, 796–803 s) | 0 (+2 "Claude code") | 9 | 97 s |
| Whisper, `condition_on_previous_text` off | 5396 | 9 | 0 | 0 (+2 "Claude code") | 23 | 67 s |
| … plus hotwords "Claude Code, CLAUDE.md, Anthropic" | 5389 | 8 | 0 | 0 | 15 | 67 s |
| … plus hotwords "Claude, Claude Code, CLAUDE.md, Anthropic" | 5423 | 13 | 1 (278 s) | **39** | 1 | 67 s |
| Parakeet TDT 0.6B v3 | 5555 | 8 | 0 | 0 (+1 "Claude code") | 7 | 16 s + 15 s load |

Against Parakeet's transcript of the same audio, with "uh" and "um" removed
from both (Whisper drops them, Parakeet keeps them), the passages of eight or
more words each Whisper run leaves out:

| run | agreement | passages left out |
| --- | --- | --- |
| default | 0.932 | one, 16 words |
| `condition_on_previous_text` off | 0.959 | none |
| … plus the first hotword list | 0.949 | one, 8 words |
| … plus the second hotword list | 0.949 | one, 25 words ("With Claude Code, you can just ask Claude Code and it'll explore the code base, it'll answer these kind of questions"), and the loop at 278 s, which replaced 17 words |

What the comparison says:

- **Conditioning on previous text causes most of the damage.** Off, the loops
  and the twenty invented seconds are gone, nothing is left out, and it is
  30 seconds faster. It is now off (DECISIONS.md).
- **A loop can still happen inside one window**, and it reads as speech: the
  loop at 278 s was "you run it" twelve times in a 33-second segment at
  -0.20. `transcribe.looping` now marks it.
- **The hotword list fixes the names and costs something.** The second list
  gave "Claude Code" 39 times instead of never, but that run left out one
  sentence, unmarked. The first list, without "Claude" on its own, wrote the
  name as "CLAUDE" fifty times: the list is literal, case included. The
  vocabulary is opt-in (DECISIONS.md).
- **Parakeet is as clean as Whisper with conditioning off, and much faster,
  but not a better fit.** No loops and nothing left out, 16 s for 28 minutes.
  Against it: it also gets the name wrong ("Cloud Code", "quad code"); its
  card memory grows with the recording (9.8 GiB for 11½ minutes, 13.2 GiB
  for 28, against 5.4 GiB for Whisper whatever the length), so a 90-minute
  lecture would need the audio cut into pieces; NeMo's word confidence, which
  `[unclear]` would need, worked on 11½ minutes and raised "Something went
  wrong with word-level confidence aggregation" on 28; and the image is
  20 GB against 11.3. Polish has not been tried on either.

On the diodes recording, for completeness: Whisper with conditioning off gave
2194 words and 8 repeated sequences in 36.5 s, but stopped before the last
words of the outro ("…Instagram, Twitter" without "as well as
theengineeringmindset.com"); Parakeet gave 2199 words and 8 repeated in
14.9 s plus 17.2 s of loading, at 9.8 GiB, with word confidence between 0.71
and 1.00.

**The talk again, with both fixes**, through the real worker: conditioning
off and the vocabulary "Claude, Claude Code, CLAUDE.md, Anthropic". The
transcript has "Claude Code" 39 times and "Cloud Code" once, and the loop at
278 s carries `[unclear]`. The chunk stage closed the first window at a real
pause this time, 3.77 s at 00:13:59, giving two windows of 3775 and 3574
tokens instead of 5939 and 1668. The default transcript had no pause of
2.5 s anywhere in its first 21 minutes; this one has it just after the
stretch where the default invented twenty seconds of speech, which looks like
the invented speech had covered the pause. `qwen2.5:32b` then wrote "01
Claude Code Practical Tips and Tricks" and "02 ClaudeMD and Context
Management", in 127 s and 105 s, with no "Quadcode" or "Cloud Code" in either
note. The earlier notes, with the wrong names, stay in the output folder,
since no stage deletes anything; the map lists only the new ones.

**The two text models on the talk**, reviewed against the transcript made
with conditioning on and no vocabulary, so both notes say "Quadcode":

| | `qwen2.5:32b` | `gemma3:27b` |
| --- | --- | --- |
| claims, window 1 + 2 | 6 + 5 | 12 + 9 |
| content errors | 3: the implementation difficulty and multimodality attributed to the SDK rather than the tool; "80% use it daily" moved to "for machine learning and modelling" | 2: MCP expanded as "Most Common Pattern"; bash safety attributed to the SDK |
| glossary lines wrong | several ("Środowisko Rozwoju Integracyjnego", "bezpieczna protokół", "zeszyt Jupyter") | 1 ("polecenia prefiksowe") |

The same picture as on the diodes window: `gemma3:27b` covers more of the
window and is wrong less often, in the same time.

## Not yet measured

These are the numbers CLAUDE.md's open decisions turn on. Each needs one run.

- **`gemma3:27b` and `qwen3:32b` against `qwen2.5:32b` on a 60-minute
  recording**, reviewed the same way as the one window above: errors per
  window, citations that miss their paragraph, and the glossary.
- **The CPU fallback on this machine**, `int8`, 8 threads: wall clock, and
  whether it loops the way `int8_float16` did on the card.
- **A Polish recording through both speech models**, with and without a
  vocabulary: the comparison above is English only, and Polish is what the
  tool is for.
- **How often a vocabulary costs a sentence.** One recording, one omission;
  it needs several recordings with and without the list.
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
