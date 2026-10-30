# Decisions

Append-only. Each entry records a decision that the code assumes and that a
reader of `CLAUDE.md` would otherwise have to guess. Entries inherited from
glosator are named as such; the reasoning lives in glosator's own
`docs/DECISIONS.md` and is not repeated here.

## 2026-10-04 — tachygraf is glosator's sibling, and copies rather than imports

glosator is a working pipeline on the same machine with the same constraints:
rootless Podman, 4 GB of VRAM, Ollama on the host, notes for Obsidian,
on-demand start and stop. Everything that holds for a book holds for a
recording, except the first two stages. So `log.py`, `db.py`, `ollama.py`,
`worker.py` and the path and ownership guards in `generate.py` are copied
from glosator at commit `89bf4cb`, with their copyright headers, and the
launcher and the Quadlet unit are derived from `bin/glosator` and
`quadlet/glosator.container`.

Copied, not shared: a package used by two projects on one laptop would be
extracted for its own sake. It waits for a third consumer. A file copied from
glosator that is later changed here gets an entry in this document saying
why; the default is that glosator's version is right.

## 2026-10-04 — The output folder is a staging folder, never the vault

glosator may write into an Obsidian vault, guarded by three rules enforced in
code: the `generator:` marker, `check_inside`, and the ownership stamp. Those
rules are kept. On top of them, tachygraf is never given the vault path at
all: `bin/tachygraf install` refuses a NOTES argument that is, or lies inside,
a directory holding `.obsidian`. The user moves a finished recording's folder
into the vault by hand.

The reason is the asymmetry of the two outcomes. A staging step costs one
`mv` per recording. A container with the vault mounted costs, on the first
bug that the three guards do not cover, a note collection that matters. The
guards were written after glosator's first real run; a fourth rule that keeps
the mount away is cheaper than discovering the fourth gap.

## 2026-10-04 — The container is given the GPU, which glosator never is

glosator's only GPU user is Ollama on the host; docling is pinned to the CPU
and the container never sees the card. tachygraf's ASR stage is
faster-whisper inside the container, and `large-v3` on the CPU takes longer
than the recording lasts. So the unit carries `AddDevice=nvidia.com/gpu=all`
and the host needs a CDI spec.

Two consequences in code:

- The ASR model and the text model never hold the card at the same time. A
  job runs `extract → transcribe → release → generate → unload`, strictly in
  that order; the ASR model is instantiated inside the transcribe stage and
  dropped in its `finally`.
- Before the transcribe stage loads the model it asks `nvidia-ml-py` how much
  memory is free. If another process holds the card, the stage runs on the
  CPU and says so in the log and the UI. A silent fallback is the docling
  failure mode glosator met once, where a stage that "completed" had quietly
  degraded.

CTranslate2 brings its CUDA runtime as pip wheels (`nvidia-cublas-cu12`,
`nvidia-cudnn-cu12`), so the image needs neither torch nor a system CUDA
install. Measured size goes in PILOT.md once the image is built.

## 2026-10-04 — The text model is qwen2.5:32b, on glosator's measurement

glosator compared `gemma3:12b-it-qat` and `qwen2.5:32b` on the same chunk,
prompt and reviewers: 19 confirmed errors against 5, at ~4 min against
17 min per note. The 12B errors were reversals that read fluently, which is
the one failure a study note cannot afford. tachygraf starts where glosator
ended up. The model stays per job, as there: `--model` on the command line, a
dropdown in the interface, `TACHYGRAF_TEXT_MODEL` for the default.

The comparison was on book text. Spoken language is looser and the windows
carry paragraph labels the model must echo, so the first PILOT.md entry here
is the 32B model on one 60-minute recording: minutes per window, responses
the parser rejected, and the rule each rejection broke.

## 2026-10-04 — Hard rules name one failure each

glosator reviewed six of its notes against their sources: a general
instruction to be faithful prevented none of sixty errors, and rules naming
one failure each did. tachygraf's `prompts/note.md` is written that way from
the start, and the rules that apply to speech as well as to text are carried
over: an invented formula, an undefined symbol, a value moved to another
quantity, a second name for one thing, a direction reversed because the
reverse sounds familiar. Rules about figures and part designators are
replaced by their spoken equivalents: a slide or figure number may be named
only where the speaker said it.

A rule is added only after a review has produced the failure it forbids. A
rule the parser can check is checked (`generate.parse`); a rule it cannot
check is still a rule. `PROMPT_VERSION` in `config.py` is bumped with every
prompt change and written into each note as `prompt: N`.

## 2026-10-04 — The model sees labels, never times

A note's link to the transcript is an Obsidian block reference,
`[[_Transcript#^p0042]]`. The model is handed each paragraph prefixed with its
label (`[p0042] …`) and asked to end every claim with the labels it rests on.
It never sees a timestamp, so it cannot invent one; the parser maps labels to
block IDs and drops a claim whose label is not in the window.

A claim without a label is dropped and counted; a label that does not exist
fails the whole response. The first is an omission that loses one bullet; the
second means the model is producing references that look right and are not,
and nothing else in that response can be trusted either.

## 2026-10-04 — A transcript paragraph is a pause or a minute, whichever first

faster-whisper emits segments of a few seconds. Shown one per line with a
timestamp each, the transcript reads as a subtitle file. Segments are joined
into a paragraph until a pause of `PARAGRAPH_PAUSE_S` (1.5 s) or
`PARAGRAPH_MAX_S` (60 s), and the paragraph carries one timestamp and one
block ID. The per-segment times stay in `segments.json`.

Both numbers are guesses until the first real transcript. They are tunables
in `config.py` for that reason.

## 2026-10-04 — The name

`tachygraf`: the ancient shorthand clerk, the pair to glosator's glossator.
Checked on 2026-10-04: free as a GitHub repository name, free on PyPI, no
project of that name in web search. `stenograf`, the working name until then,
is a commercial transcription product and a PyPI package.

The published port is 7861, because glosator holds 7860 on the same machine.

## 2026-10-04 — The index stage stays deterministic, so moc.md is a template

CLAUDE.md's pipeline table and the README both give the index stage no model,
and `index.py` carries glosator's reason in its docstring: a stage with no
model can be re-run after any regeneration without changing anything else in
the output folder. `app/prompts/moc.md` as first drafted was a model prompt
instead, with its own hard rules, asking for an `## Overview` and a
`## Thread` written from the notes' "In short" sections.

The table won. `moc.md` is now a `{{placeholder}}` template, as in glosator,
and `_Map.md` is assembled from what the notes already say: one line per note
with the span of the recording it covers and the note's own first sentence,
copied rather than paraphrased. The map therefore cannot state anything about
the lecture that a note does not state, which is a property the model-written
version could not have.

The cost is that the map has no prose connecting one note to the next. If that
turns out to be wanted, the way back is a sixth stage with its own model, not a
model call inside this one: `index` is the stage that runs after a stop, after
a failure and after a `--force`, and it must stay cheap and repeatable.

## 2026-10-04 — A note's title is cut from its own first sentence

CLAUDE.md shows a note headed `# 03 Title the model gave this window`, and
`note.md` forbids the model from writing a title heading, because a heading is
a place for it to restate the task or invent a lecture name. Both hold: the
title is cut in `generate.title_from` from the first sentence of the model's
`## In short`, trimmed at a word boundary to 60 characters and stripped of the
characters a file name may not carry.

So the title is the model's own words without the model being asked for a
title, and no prompt change was needed for it. Two consequences:

- A note's file name is not known until the model has answered, so the index
  stage cannot compute it. It reads the path back from the `output_path` of the
  window's `generate` row (`db.stage_outputs`). A window that produced no note
  is simply absent from the map and from the previous/next chain.
- Regenerating a window with a different model may produce a different title
  and therefore a second file. Nothing is ever deleted, so the old note stays
  until the user removes it. That is the lesser of the two evils: the
  alternative is deleting a note because a title changed.

## 2026-10-04 — Re-planning keeps the windows that did not change

glosator's `replace_chunks` deletes every chunk row and lets the cascade take
the stage rows with it, so re-planning regenerates everything. That is right
there, where a plan is drawn from page ranges the operator chose: re-planning
means the operator wants different notes.

Here a plan is a pure function of `segments.json`. Running the five stages a
second time — which is exactly what happens after the container is killed —
re-runs `chunk` and gets an identical plan, and glosator's behaviour would
then throw away every finished note and spend hours of 32B generation
reproducing them. "The container can be killed at any point; the next run
resumes" is the point of the table, so a window whose position, span, labels
and text file are all unchanged keeps its row and its stage rows. A window
that differs in any of them is replaced and the cascade does drop its stage
rows, because a note made from it was made from different speech.

## 2026-10-04 — A window is identified by its position, not by a name

glosator keys a chunk on its section number, which the book supplies before
any model runs. A recording has no sections, and a window's title does not
exist until the model has read it, so `chunks` is keyed on `(source_id, ord)`
and `stage_runs` points at the row. The `labels` column holds the window's
paragraph labels, which is what makes the previous entry's comparison possible
and what the parser checks a citation against.

## 2026-10-04 — The write guards decide the module order

`generate.py` holds the path and ownership guards, copied from glosator, and
both `transcribe.py` and `chunk.py` need them: the transcript is written into
the output folder too. So `generate.py` imports nothing from `chunk.py` or
`transcribe.py` at run time, even though it sits after them in the pipeline —
the window type is a `TYPE_CHECKING` import and `estimate_tokens` lives beside
the budget check that uses it. Import order is `config → generate → transcribe
→ chunk → index`, which is not the pipeline order and does not need to be.

The alternative was a fourth module holding the guards, which would have put
the three rules that protect the user's files somewhere CLAUDE.md does not
mention.

## 2026-10-04 — slugify keeps the l in ł

`unicodedata.normalize("NFKD", …)` decomposes ą, ć, ę, ń, ó, ś, ź and ż into a
plain letter and a combining mark, which the ASCII encode then drops. It does
not decompose ł, because ł is a letter in its own right rather than l with a
stroke, so glosator's `slugify` turns "Wykład 4" into "wykad-4" and the
recording's folder, its tag and every link to it carry the typo.

`media.slugify` translates ł, Ł, đ, Đ, ø and Ø before the normalisation.
Divergence from `glosator.extract.slugify`, and a bug glosator still has.

## 2026-10-04 — The ASR stage is the only consumer of nvidia-ml-py

`transcribe.gpu_free_mb` returns `None` when nvml is absent or unanswerable,
and `None` is not zero: it means the card could not be asked, which is also
what a host with no card looks like. `choose_device` treats both as a reason
to use the CPU and returns the reason as a string, which goes into the log,
into `segments.json` under `asr.device_reason`, and into the interface before
the job starts rather than after it. A transcription that silently ran on the
CPU is several hours the user did not plan for.

If the card is reported free and then refuses the model — nvml answered a
moment too early, which is the common case when Ollama is still releasing —
the stage retries once on the CPU and says so. CTranslate2 raises a plain
`RuntimeError` for this, so the retry cannot be narrower.

## 2026-10-04 — The prompt's example labels are a known rejection

`note.md` illustrates the citation form with `[p0042]`, and `[p0051] [p0052]`
for the two-label case. A model that copies an example label instead of
reading the window writes a citation that is not in the window, and
`generate.parse` rejects the whole response for it, which is correct and is
also a window's worth of generation spent on nothing.

The examples stay. The prompt needs to show the form, a rule the parser can
check is checked, and the rejection is recorded with the rule it broke, so
the first pilot run will say how often a 32B model actually does this. If it
is common, the fix is a prompt change with its own commit and a
`PROMPT_VERSION` bump, not a looser parser.

## 2026-10-04 — The ASR model is handed samples, not a path

The first real run died here. `transcribe.transcribe_audio` passed the WAV's
path to `WhisperModel.transcribe`, which decodes the file itself through PyAV;
faster-whisper 1.2.1 calls `av.open(metadata_errors="ignore")` and PyAV 19
removed that keyword, so the stage raised
`TypeError: open() got an unexpected keyword argument 'metadata_errors'` after
loading 1.5 GB of weights onto the card, and the three stages behind it failed
in turn on the files it had not written.

Pinning `av` would have fixed the symptom. The cause is that the stage was
decoding the audio twice: `media.extract_audio` has already produced exactly
16 kHz mono signed 16-bit PCM, which is the format CTranslate2 works in, and
handing faster-whisper a path threw that away and asked a second audio library
to reproduce it.

`transcribe.read_wav` now reads that WAV with the standard library's `wave`
module into the float32 array the model consumes, and refuses a file that is
not in the expected format rather than resampling it — resampling here would
make this module a second audio pipeline with its own idea of the right
format, which is the mistake being undone. PyAV is no longer on the path at
all, `numpy` is declared in `pyproject.toml` because the code now uses it
directly, and a test asserts that what reaches the model is an `ndarray`, so
the path cannot come back by accident.

The lesson is the general one: the one stage that normalises the audio should
be the only stage that touches an audio library.

## 2026-10-04 — The CUDA runtime is an extra, and the fallback covers inference

The second thing the first real run found. The Containerfile said CTranslate2
"brings its own CUDA runtime through the nvidia-cublas-cu12 and
nvidia-cudnn-cu12 wheels", and nothing installed them: `ctranslate2` does not
declare those wheels, so the image had no `libcublas.so.12` anywhere in it.
nvml reported 3921 MB free, the stage chose the card, `large-v3` loaded, and
the first inference raised
`RuntimeError: Library libcublas.so.12 is not found or cannot be loaded`.

Two separate faults, fixed separately.

The packaging one: the wheels are now a `cuda` extra in `pyproject.toml`, which
the Containerfile installs as `.[dev,cuda]`, with both CUDA majors pinned —
ctranslate2 4.8 is built against cuBLAS 12 and cuDNN 9, and a cuDNN 10 wheel
would install cleanly and fail at run time. They land under
`site-packages/nvidia/*/lib`, which is on no default search path and which
CTranslate2 dlopens by bare soname, so the image sets `LD_LIBRARY_PATH`. An
extra rather than a dependency because it is 1.5 GB that only the container
needs; the host suite never touches the card.

The design one, which matters more. The CPU fallback only wrapped
`load_model`, and faster-whisper does the work while its generator is
consumed, so a card that failed at the first compute rather than at load
skipped the fallback entirely and failed the job. The segments are now
consumed inside `_run_model`, where the failure is catchable, and any failure
on the card — busy, mismatched driver, unloadable libraries — sends the whole
attempt to the CPU with the reason recorded in the log, in the interface and
in `segments.json` under `asr.device_reason`.

That is the right behaviour even though this particular cause was a packaging
bug: the stage cannot tell the difference between a card that is held and a
card that is broken, and a recording transcribed slowly is worth having where
a job that failed after an hour of queueing is not. "Checked with
nvidia-ml-py, not assumed" was not enough — nvml answering "free" is not a
promise that the card can be used.

## 2026-10-04 — The model is asked for a title, because deriving one failed

The entry above ("A note's title is cut from its own first sentence") was
settled by the first pilot run against it. Asked for an "In short" and given
no title to write, `qwen2.5:32b` opened with

> This part of the lecture explains what diodes are, how they work, and their
> applications in controlling current flow direction in circuits.

which is a correct first sentence and a bad title. Cut to a file name it
became `01 This part of the lecture explains what diodes are, how they.md`:
truncated mid-phrase, naming the exercise instead of the subject, and then
repeated in every wikilink in `_Glossary.md` that pointed at the note.

So `note.md` gains a `## Title` section, which makes it seven sections, and
`PROMPT_VERSION` goes to 2. The prompt asks for four to ten words naming the
subject and forbids by name the openings that produced this — "this window",
"this part of the lecture", "this video", "the speaker" — with the failure
shown as the counter-example, which is how every other rule in that file is
written.

The heading is still assembled in code and `## Title` never appears in the
note. Two things sit behind the prompt:

- `is_about_the_task` checks the forbidden openings, and a title that names
  the task is replaced rather than rejected. A note is ten minutes of
  generation and the rest of it may be perfectly good, so the fallback is the
  old derivation with the lead-in stripped, and `title_substituted` is
  counted in the log. A pilot can then be read as "how often did the model
  need to be told twice".
- `_trim` cuts at a clause before a word boundary and drops a dangling
  conjunction or preposition, because "... what diodes are, how they" is what
  a word-boundary cut produces and it reads as a truncation. The same
  sentence now yields "What diodes are, how they work".

## 2026-10-04 — The Polish glossary is the weakest thing the 32B produces

The same note's glossary, reviewed against what a Polish textbook uses: three
of seven entries were wrong and one was poor.

| the model wrote | established term |
| --- | --- |
| depletion region — *obszar dezaktywacji* | obszar zubożony |
| reverse bias — *przewodzenie wsteczne* | polaryzacja zaporowa |
| forward bias — *przewodzenie do przodu* | polaryzacja w kierunku przewodzenia |
| p-type doping — *doping typu p* | domieszkowanie typu p |

Every one is a calque: the English words translated one at a time into
something that looks Polish and that no Polish source uses. "obszar
dezaktywacji" is not a term at all. This is worse than an admitted gap,
because a reader looking the phrase up finds nothing and has no way to tell
that it was invented — the same shape of failure as glosator's fluent
reversals, in the one section of the note whose whole purpose is to be looked
up.

The parser cannot catch it: it can check that a line has a separator and that
the right-hand side exists, not whether a Polish phrase is the one Poles use.
So this is a prompt rule, added the way CLAUDE.md requires — after a review
produced the failure it forbids. `note.md` now says to give the term a Polish
textbook uses, not a word-by-word translation, shows these two failures as
the counter-examples, and repeats that `(brak odpowiednika)` is the right
answer when the established term is not known.

Whether the rule works is the next thing to measure. If a 32B model cannot
reliably produce Polish technical terminology, that is an argument about
CLAUDE.md's open decision 2 (note language) rather than about the prompt: the
glossary is the one part of the note that exists to be in Polish, and a
glossary that invents terms is worse than no glossary.

## 2026-10-04 — A folder of recordings is one command, and the hard part is duplicates

A course is twenty recordings, not one long one, and `bin/tachygraf cli add`
twenty times is not a thing to ask of anyone. `app/batch.py` takes a folder
and queues all five stages for every recording in it, each with its own slug,
its own output folder and its own notes.

The loop is not the problem. What a downloader leaves on disk is: the same
lecture as a 4K video *and* as an audio-only track, beside a subtitle file, a
thumbnail and a description. Processing every media file would transcribe
every lecture twice and generate two sets of notes from the same words — for
a twenty-part course, a day of the card spent on duplicates. So the module
collapses each recording to one file before anything is queued.

Three things must agree for two files to be one recording: the parent
directory, the title once the format tag is stripped, and the duration to
within 1.5 s. All three, because any two give a false positive on real data,
and the first version of this proved it: duration and directory alone merged
a flat course folder into a single recording, because a course cut to a slot
has many episodes of exactly the same length. Title and directory alone would
merge a lecture with its own trailer.

The asymmetry decides the strictness. Being wrong towards "these are
different" costs a duplicated recording, which the plan shows and the user
can see. Being wrong towards "these are the same" drops a lecture silently.

Three smaller decisions follow from the same data:

- The audio-only file wins. The audio is identical, it is a tenth of the
  bytes, and ffmpeg does not have to walk past a video stream to reach it.
- A title is the file stem with trailing format brackets removed, and only
  those: `(2160p_25fps_AV1-128kbit_AAC-English)` goes, `(part 2)` stays. The
  bracket's contents are split on underscores before being matched, because
  an underscore is a word character and the first version of the pattern
  matched nothing inside `1080p_30fps_H264`.
- Recordings are ordered naturally, so `lekcja-9` precedes `lekcja-10`, and
  the search descends into subdirectories by default, because a downloaded
  playlist is often one folder per recording rather than one folder of them.

The plan is printed before anything is queued, and `--dry-run` prints it and
stops. That is not a convenience: the batch chooses files on the user's
behalf, and a choice made for someone has to be visible to them.

## 2026-10-04 — A batch is ordered by recording, not by stage

The five jobs of one recording are queued together, so the first lecture's
notes are finished before the second lecture's transcription starts. A
complete note is the product and a transcript is an intermediate, so a batch
stopped halfway should leave finished lectures behind rather than twenty
transcripts and nothing to read.

The cost is that the text model is unloaded when each `generate` job ends and
reloaded for the next recording, because the card is handed back when a job
ends. At the measured ten minutes per window that is a few per cent, and the
alternative — keeping the model resident across jobs when the next queued job
wants the same one — would relax a non-negotiable for a saving that small.
Stage-grouped order (every transcription, then every generation) would give
all the transcripts sooner and is a one-line change in `batch.queue` if a long
course ever makes that the better trade.

## 2026-10-30 — Whisper runs in float16 on the card, because int8 corrupted it

`ASR_COMPUTE_GPU` was `int8_float16` for one reason: `large-v3` in float16 did
not fit a 4 GB card. On the RTX 5070 Ti the same setting corrupts the
transcript, and does it without raising anything.

Measured on the diodes recording (PILOT.md, 2026-10-30): `large-v3` at
`int8_float16` wrote loops such as "The electrons are also called electrons.
The electrons are also called" and "the diode acts as an intrusion. Therefore
the diode acts as an intrusion", 63 repeated six-word sequences against 8 at
float16 — 8 being how often the speaker actually repeats himself.
`large-v3-turbo` at `int8_float16` dropped a sentence float16 kept. float16
was no slower for either model, and at 5.8 GB peak for `large-v3` there is no
memory reason left for int8.

The fault belongs to this card's int8 path, not to int8 as such. On the
laptop's RTX 3050 (Ampere) `large-v3-turbo` at `int8_float16` gave 2198 words
and 9 repeated six-word sequences; on this card float16 gives the same 2198
and 8, and `int8_float16` gives 2173.

The failure matters more than its size. The CPU fallback exists for a card
that refuses — out of memory, missing libraries — and it catches exceptions.
A card that computes wrongly raises nothing, the stage reports success, and
the transcript is the one file in the output whose contract is "verbatim, no
model between the ASR and this file". Reports from other RTX 50 users name
`int8_float16` as failing with `CUBLAS_STATUS_NOT_SUPPORTED`; here, with
cuBLAS 12.9 and CTranslate2 4.8.4, it did not fail, which is worse.

A test asserts that the card's compute type contains no `int8`, so the setting
cannot drift back with a copy from an older config. The CPU path stays `int8`:
float16 is not implemented there. It has not been checked for loops on this
machine, and no CPU transcript from the laptop was kept to check.

## 2026-10-30 — The reference machine is a 16 GB desktop card

The machine changed. It was a laptop with an RTX 3050 Mobile (4 GB) and
64 GB of RAM; it is now a desktop with an RTX 5070 Ti (16 GB, Blackwell,
compute capability 12.0), a Ryzen 7 7800X3D (8 cores, 16 threads, AVX-512)
and 32 GB of RAM. The desktop itself is drawn on the NVIDIA card and holds
1.3–1.5 GB of it, so about 15 GB is what a job can have.

What that changes:

- Both ASR models run on the card. On the laptop `large-v3` was a CPU model
  whatever else ran, and `config.ASR_MIN_FREE_MB` deliberately named a figure
  above the card's capacity to say so. The figures are now measured peaks on
  this card (PILOT.md, 2026-10-30), and what they guard against is a card
  Ollama has not yet let go of, not a card that is too small.
- Blackwell needs CUDA libraries that know compute capability 12.0: cuBLAS
  from CUDA 12.8 and cuDNN 9.7. The `cuda` extra's lower bounds say so. The
  old bounds (12.3, 9.1) would resolve to the same wheels today and to a
  broken image the day a resolver picked the floor.
- The CDI spec. `nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` writes
  the driver's library versions into a file nothing regenerates, and Arch
  updates the driver often. nvidia-container-toolkit ships
  `nvidia-cdi-refresh.path` and `.service`, which write `/var/run/cdi` at boot
  and at every driver update; the launcher and the unit now point to those.
  `/var/run` is a tmpfs, so a spec generated there by hand is gone after a
  reboot. The launcher's check also had a bug this machine exposed: `ls` on
  both paths fails whenever one is missing, so a spec in `/var/run/cdi` alone
  produced the "no CDI spec" warning.

What it does not change:

- One model on the GPU at a time. `qwen2.5:32b` is about 20 GB, more than the
  whole card, so Ollama still offloads part of it to the CPU and still sizes
  that offload to the VRAM free when it loads. An ASR model left resident
  would cost the generate stage layers for its whole duration.
- The text model. The hardware removes the reason a smaller model was ever
  worth considering, which was speed; it does not touch the reason
  `qwen2.5:32b` was chosen, which was 5 confirmed errors against 19. What the
  card makes cheap is comparing models of the same class on one window, and
  that is a PILOT.md entry before it is a decision.
- RAM halved, from 64 to 32 GB. A 32B model with part of it offloaded fits;
  nothing larger does, so a 70B model is out of reach on this machine
  whatever the card.

## 2026-10-30 — Whisper is not conditioned on its own previous text

faster-whisper passes each 30-second window the text of the windows before it
as a prompt (`condition_on_previous_text`, on by default). The point is
continuity of spelling and style. The cost is that a mistake becomes the
prompt for the next window, and Whisper continues a loop it has started.

The first recording long enough to show it, a 28-minute talk (PILOT.md,
2026-10-30): with the default, `large-v3` wrote "you were doing technical
onboarding" five times in one second of audio, a 24-word sentence twice in
under three seconds, and then twenty seconds of invented words, including a Chinese
character. With the option off, the same audio gave none of it: 9 repeated
six-word sequences against 47, the speaker's own repetitions being about 9.
It lost no passage of eight words or more against an independent transcript
of the same audio by a different model (NVIDIA Parakeet TDT), where the
default lost one, and it took 67 s instead of 97.

So it is off. What is given up is the carrying of a spelling from one window
to the next; a name the model gets right at minute 3 may be spelled
differently at minute 20. That is the lesser failure: an inconsistent spelling
is visible in the transcript, a loop that replaces what was said is not.

This does not stop a loop inside a single window, which happens without any
prompt from earlier windows; that is a separate guard.

## 2026-10-30 — A repetition loop is marked unclear, measured by its extra words

Turning off conditioning on previous text stops a loop being carried from one
window to the next, not a loop inside one window. With a vocabulary hint in
the prompt, `large-v3` wrote "you run it" twelve times in place of "you start
it, there's no indexing, you don't have to wait, you can just use it right
away", in a 33-second segment with an average log probability of -0.20. The
`[unclear]` check on log probability did not see it, and nothing else would
have: the transcript would have carried the loop as speech, and a note could
have cited it.

So `transcribe.looping` looks at the words. A run of two to
`LOOP_MAX_RUN_WORDS` words said back to back is a loop when the copies after
the first add up to `LOOP_MIN_REPEATED_WORDS` or more. Every segment the run
passes through is marked `[unclear]`, which the note prompt already treats as
"report as unclear, do not complete". The marker is added, the words are not
changed, as with the existing marker.

The threshold is counted in extra words, not in copies, because the two
failure shapes differ: a short phrase many times ("you run it" ×12, 33 extra
words) and a long sentence twice (24 words, 24 extra). The six loops in the
transcripts measured on 2026-10-30 added 12 to 33 words; the speakers' own
repetitions in the same transcripts added at most 6 ("this tool and this tool
and this tool and this tool"). Ten sits between, with no false mark in eight
clean transcripts from two ASR engines and every true loop marked.

The check runs on the whole transcript, not per segment, because a loop of a
long sentence was split across three segments of under a second each.

## 2026-10-30 — A recording's vocabulary goes to the speech model, and only there

The 28-minute talk was about Claude Code. Whisper wrote it as "Quadcode",
"QuadCode", "Cloud Code", "Quad", "ClaudeMD" and "QuadMD" — about a dozen
spellings for two names — and both text models carried them into the notes, titles
included: "01 Quadcode Practical Tips and Tricks".

A text model could have guessed "Claude Code" from context, and the pipeline
deliberately does not let it. The transcript is verbatim, with no model
between the ASR and the file; the note prompt says to use only what the
source contains and to give each thing the one name the speaker used. A model
that corrects names it believes are wrong is the same mechanism as a model
that invents fluently, and it cannot tell a misheard name from an unusual one
that was actually said. So the correction has to happen where the sound is:
in recognition, as a hint, not after it, as an edit.

The transcribe stage therefore takes a vocabulary per recording — a
comma-separated string, like the language a job parameter — and passes it to
faster-whisper as `hotwords`, which goes into the prompt of every 30-second
window. Measured on the talk (PILOT.md, 2026-10-30): with "Claude, Claude
Code, CLAUDE.md, Anthropic", "Claude Code" appeared 39 times and "Cloud
Code" once, against 2 and 23 without it.

Two things the measurement showed that the interface has to say:

- The list is taken literally, case included. With "Claude Code, CLAUDE.md,
  Anthropic" — no "Claude" on its own — the model wrote the name itself as
  "CLAUDE" fifty times. The help
  text and the README say to spell each entry as the transcript should have
  it.
- A different prompt is a different decoding, and it is not free. The run
  with the hint had two faults the run without it did not: a loop ("you run
  it" twelve times), which the loop check marks, and a 25-word sentence left
  out with nothing in its place ("With Claude Code, you can just ask Claude
  Code and it'll explore the code base, it'll answer these kind of
  questions"), which nothing marks. The run without the hint lost no passage
  of eight words or more against Parakeet's transcript of the same audio.

So the vocabulary is opt-in and empty by default. On this talk it traded one
unmarked missing sentence for 39 correct names in place of none, and the
names are what the notes are built on; on a recording whose terms Whisper
already knows it would buy nothing and cost the same risk. One recording is
not enough to say how often the omission happens, which is a PILOT.md item.

The vocabulary is recorded in `segments.json` under `asr.vocabulary`, so a
transcript can be traced to the hint it was made with. It is empty by default,
and empty means no hint at all rather than an empty prompt.
