# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""Gradio front end on :7860, published on the host at :7861.

Copied from glosator's ui.py, with its book fields replaced by a recording's.
The rules are the same ones: this module only queues jobs and reads state, the
worker runs in its own process, and closing the browser cannot kill a running
job. A stop can, which is what the second stop button is for.

Nothing here is given the vault path. The output folder is a staging folder and
``bin/tachygraf install`` has already refused a vault for it; the box below is
only for choosing between staging folders.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import gradio as gr

from app import batch as batch_module
from app import config, db, generate, log, media, ollama, transcribe

WORKER_RESTART_S = 5.0
JOB_COLUMNS = ["id", "stage", "status", "progress", "eta", "model", "error"]

# Containers ffmpeg decodes and a lecture arrives in. Not a limit on what works,
# only on what the dropdown offers: the command line takes any path.
MEDIA_SUFFIXES = (
    ".mp4",
    ".mkv",
    ".webm",
    ".mov",
    ".avi",
    ".m4a",
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
)

LANGUAGES = ["auto", "pl", "en", "de", "fr", "es", "it", "uk", "ru"]

# The ASR models baked into the image. Each one needs a measured VRAM figure in
# config.ASR_MIN_FREE_MB, or the stage cannot say which device it will use
# before it starts.
ASR_MODELS = ["large-v3", "large-v3-turbo"]


def media_choices(media_dir: str | Path) -> list[str]:
    directory = Path(media_dir).expanduser()
    if not directory.is_dir():
        return []
    return sorted(
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES
    )


def resolve_media(media_dir: str | Path, choice: str | None) -> Path:
    """The chosen recording, as a path inside the container.

    There is no upload box, unlike glosator: a lecture recording is hundreds of
    megabytes and it is already on the disk that is mounted at /in. Uploading
    it would copy it into /work for no gain.
    """
    if not choice:
        raise gr.Error("pick a recording from the recordings folder")
    path = Path(media_dir).expanduser() / choice
    if not path.is_file():
        raise gr.Error(f"no such recording: {path}")
    return path


def resolve_out_dir(notes_dir: str) -> Path:
    """The output folder must exist: creating it blindly invites a typo."""
    path = Path(notes_dir.strip()).expanduser()
    if not path.is_dir():
        raise gr.Error(f"output folder does not exist: {path}")
    return path.resolve()


def job_table(conn) -> list[list[str]]:
    rows = []
    for job in db.recent_jobs(conn):
        rows.append(
            [
                str(job["id"]),
                job["stage"],
                "stopping"
                if job["stop"] and job["status"] == db.RUNNING
                else job["status"],
                f"{job['done']}/{job['total']}",
                _eta(job),
                job["model"],
                (job["error"] or "")[:120],
            ]
        )
    return rows


def _eta(job) -> str:
    if job["status"] != db.RUNNING or not job["started"] or job["done"] < 1:
        return "-"
    started = datetime.fromisoformat(job["started"])
    elapsed = (datetime.now(UTC) - started).total_seconds()
    per_unit = elapsed / job["done"]
    remaining = max(0, job["total"] - job["done"]) * per_unit
    return f"{remaining / 60:.0f} min"


def last_note(notes_dir: str | Path) -> str:
    """Most recently written note, for a look at what the model produced."""
    directory = Path(notes_dir).expanduser()
    if not directory.is_dir():
        return "_output folder not found_"
    notes = sorted(
        (
            path
            for path in directory.rglob("*.md")
            if path.name != transcribe.TRANSCRIPT_NOTE and generate.is_ours(path)
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not notes:
        return "_no notes yet_"
    return notes[0].read_text(encoding="utf-8")


def device_note(model_name: str = config.ASR_MODEL) -> str:
    """What the ASR stage would do with this model if it started now.

    Said before a job rather than after it, because the answer changes with
    the model and with whatever else holds the card, and a transcription that
    silently ran on the CPU is hours the user did not plan for.
    """
    device, _compute, reason = transcribe.choose_device(model_name)
    if device == "cuda":
        return f"`{model_name}` will transcribe on the GPU: {reason}."
    return (
        f"**`{model_name}` will transcribe on the CPU**: {reason}. It still "
        "works and it takes longer than the recording does."
    )


def stop_jobs(conn, job_id: float | int | None, hard: bool = False) -> str:
    """Stop one job by id, or everything still pending when none is given.

    Having to read an id out of the table and type it in before anything can
    be stopped is not a reasonable thing to ask of someone watching a job go
    wrong.
    """
    if job_id:
        db.request_stop(conn, int(job_id))
        return f"Stop requested for job {int(job_id)}."
    stopped = db.stop_all(conn)
    if not stopped:
        return "Nothing to stop: no job is running or waiting."
    listed = ", ".join(str(job) for job in stopped)
    if hard and kill_worker():
        return (
            f"Stopped {len(stopped)} job(s): {listed}. The model call was cut "
            "off; the window it was on will be generated again if you rerun."
        )
    return (
        f"Stop requested for {len(stopped)} job(s): {listed}. A running stage "
        "finishes the window it is on first, which can take a few minutes; the "
        "table shows it as stopping until then."
    )


_worker: subprocess.Popen | None = None


def supervise_worker() -> None:
    """Keep one worker process alive next to the UI.

    Restarting here rather than in a shell wrapper keeps the container to a
    single entrypoint while the worker stays a separate process. The handle is
    kept so that a stop can be carried out by killing it.
    """

    def loop() -> None:
        global _worker
        while True:
            _worker = subprocess.Popen([sys.executable, "-m", "app.worker"])
            code = _worker.wait()
            _worker = None
            log.event("ui", "worker exited", code=code)
            time.sleep(WORKER_RESTART_S)

    threading.Thread(target=loop, name="worker-supervisor", daemon=True).start()


def kill_worker() -> bool:
    """End the current model call now, instead of after the window.

    Safe by construction: a note is written through a temporary file and
    renamed, the stage row of an unfinished window stays unfinished, and the
    supervisor starts a fresh worker a few seconds later.
    """
    process = _worker
    if process is None or process.poll() is not None:
        return False
    process.terminate()
    log.event("ui", "worker killed to carry out a stop")
    return True


def build() -> gr.Blocks:
    config.ensure_dirs()
    conn = db.connect()

    with gr.Blocks(title="tachygraf") as blocks:
        gr.Markdown(
            "# tachygraf\nA recorded lecture to a verbatim transcript and "
            "study notes, locally."
        )

        with gr.Row():
            media_dir = gr.Textbox(
                label="Recordings folder (read-only)", value=str(config.IN_DIR)
            )
            notes_dir = gr.Textbox(
                label="Output folder (a staging folder, not your vault)",
                value=str(config.OUT_DIR),
            )
        gr.Markdown(
            "Nothing outside the output folder is written, nothing in it is "
            "deleted, and a file tachygraf did not write is never overwritten. "
            "Move a finished recording's folder into your vault yourself."
        )

        with gr.Row():
            with gr.Column(scale=1):
                source = gr.Dropdown(
                    media_choices(config.IN_DIR), label="Recording", interactive=True
                )
                refresh_media = gr.Button("Refresh recording list", size="sm")
                title = gr.Textbox(
                    label="Title", placeholder="what this lecture is called"
                )
                slug = gr.Textbox(
                    label="Slug (output subfolder, note tag)",
                    placeholder="short, e.g. lec4",
                )
                domain = gr.Textbox(
                    label="Field (first tag, and what the prompt says the "
                    "recording is about)",
                    placeholder="signal processing, calculus, anatomy, ...",
                )
            with gr.Column(scale=2):
                with gr.Row():
                    language = gr.Dropdown(
                        LANGUAGES,
                        value=config.LANGUAGE,
                        label="Spoken language",
                        interactive=True,
                    )
                    asr_model = gr.Dropdown(
                        ASR_MODELS,
                        value=config.ASR_MODEL,
                        label="Transcription model",
                    )
                vocabulary = gr.Textbox(
                    label="Vocabulary: names and terms the recording uses, "
                    "spelled as the transcript should have them (a hint to "
                    "transcription, optional)",
                    placeholder="Claude Code, CLAUDE.md, Anthropic",
                )
                with gr.Row():
                    model = gr.Dropdown(
                        [config.TEXT_MODEL], value=config.TEXT_MODEL, label="Text model"
                    )
                    refresh_models = gr.Button("Refresh models", size="sm")
                probe_button = gr.Button("Inspect the recording")
                probe_output = gr.Markdown()
                device_output = gr.Markdown(device_note())

        gr.Markdown(
            "## A whole folder at once\n"
            "A course, one set of notes per recording. The same lecture in two "
            "containers counts once: a video and the audio-only track beside it "
            "have the same title and the same length, and the audio one is "
            "used. Look at the plan before queueing it."
        )
        with gr.Row():
            batch_dir = gr.Textbox(
                label="Folder of recordings (searched including subfolders)",
                value=str(config.IN_DIR),
            )
            batch_prefix = gr.Textbox(
                label="Slug prefix (optional, e.g. the course name)",
                placeholder="kurs-elektroniki",
            )
            batch_number = gr.Checkbox(
                label="Number the slugs, so the course keeps its order",
                value=False,
            )
        with gr.Row():
            batch_plan = gr.Button("Show me the plan")
            batch_queue = gr.Button("Queue the whole folder", variant="primary")
        batch_output = gr.Code(label="Plan", language=None, lines=12)

        with gr.Row():
            run_all = gr.Button(
                "Run everything  (extract → transcribe → chunk → generate → index)",
                variant="primary",
                scale=2,
            )
        with gr.Row():
            gr.Markdown("Or one stage at a time, in this order:")
            queue_extract = gr.Button("1. extract", size="sm")
            queue_transcribe = gr.Button("2. transcribe", size="sm")
            queue_chunk = gr.Button("3. chunk", size="sm")
            queue_generate = gr.Button("4. generate", size="sm")
            queue_index = gr.Button("5. index", size="sm")

        status = gr.Markdown()

        gr.Markdown("## Jobs")
        jobs = gr.Dataframe(headers=JOB_COLUMNS, value=job_table(conn), wrap=True)
        with gr.Row():
            stop_button = gr.Button("Stop after this window", variant="stop")
            kill_button = gr.Button("Stop now", variant="stop")
            clear_button = gr.Button("Clear finished", size="sm")
            stop_id = gr.Number(
                label="or one job id, if you want to stop only that one",
                precision=0,
            )

        with gr.Row():
            logs = gr.Textbox(label="Log", lines=14, max_lines=14, value=log.tail())
            note_preview = gr.Markdown(label="Last note")

        timer = gr.Timer(5.0)

        def _source_id(
            recordings, notes, title_value, slug_value, domain_value, choice
        ) -> int:
            path = resolve_media(recordings, choice)
            if not title_value.strip() or not slug_value.strip():
                raise gr.Error("give the recording a title and a slug")
            return db.upsert_source(
                conn,
                slug_value.strip(),
                title_value.strip(),
                path,
                resolve_out_dir(notes),
                domain_value.strip() or config.DEFAULT_DOMAIN,
            )

        def on_refresh_media(recordings):
            return gr.update(choices=media_choices(recordings))

        def on_refresh_models():
            try:
                models = ollama.list_models()
            except ollama.OllamaUnavailable as error:
                raise gr.Error(str(error)) from error
            return gr.update(choices=models)

        def on_asr_model(asr_value):
            return device_note(asr_value)

        def on_probe(recordings, choice):
            path = resolve_media(recordings, choice)
            try:
                info = media.probe(path)
            except media.MediaError as error:
                raise gr.Error(str(error)) from error
            return (
                f"**{path.name}** — {info.duration_hms}, {info.codec}, "
                f"{info.audio_streams} audio stream(s), language tag: "
                f"{info.language_hint or 'none'}."
            )

        def _queue(stage, model_value, params, recordings, notes, *fields):
            source_id = _source_id(recordings, notes, *fields)
            job = db.enqueue(conn, source_id, stage, model_value, json.dumps(params))
            return (
                f"Queued {stage} as job {job}; output goes to "
                f"{resolve_out_dir(notes)}.",
                job_table(conn),
            )

        def on_run_all(
            recordings,
            notes,
            title_value,
            slug_value,
            domain_value,
            choice,
            language_value,
            vocabulary_value,
            asr_value,
            model_value,
        ):
            """Queue the whole recording in one press.

            The worker takes one job at a time in the order they were queued,
            so each stage finds the previous one's output waiting for it, and
            the ASR model has given the card back before Ollama is asked for
            anything.
            """
            log.event(
                "ui",
                "run everything pressed",
                title=title_value,
                slug=slug_value,
                domain=domain_value,
                language=language_value,
                vocabulary=vocabulary_value,
                asr_model=asr_value,
                model=model_value,
            )
            source_id = _source_id(
                recordings, notes, title_value, slug_value, domain_value, choice
            )
            queued = [
                db.enqueue(conn, source_id, "extract", "none", json.dumps({})),
                db.enqueue(
                    conn,
                    source_id,
                    "transcribe",
                    asr_value,
                    json.dumps(
                        {
                            "language": language_value,
                            "vocabulary": vocabulary_value.strip(),
                        }
                    ),
                ),
                db.enqueue(conn, source_id, "chunk", "none", json.dumps({})),
                db.enqueue(conn, source_id, "generate", model_value, json.dumps({})),
                db.enqueue(conn, source_id, "index", "none", json.dumps({})),
            ]
            return (
                f"Queued jobs {queued[0]}-{queued[-1]}. The transcript appears "
                "first, then the notes one by one in "
                f"{resolve_out_dir(notes)}. Watch the table below.",
                job_table(conn),
            )

        def _batch_plan(folder, prefix, number):
            try:
                return batch_module.plan(
                    Path(folder), slug_prefix=prefix or "", number=bool(number)
                )
            except media.MediaError as error:
                raise gr.Error(str(error)) from error

        def on_batch_plan(folder, prefix, number):
            return batch_module.describe(_batch_plan(folder, prefix, number))

        def on_batch_queue(
            folder,
            prefix,
            number,
            notes,
            domain_value,
            language_value,
            vocabulary_value,
            asr_value,
            model_value,
        ):
            built = _batch_plan(folder, prefix, number)
            if not built.recordings:
                raise gr.Error(f"no recordings found in {folder}")
            jobs = batch_module.queue(
                conn,
                built,
                domain=domain_value.strip() or config.DEFAULT_DOMAIN,
                out_dir=resolve_out_dir(notes),
                language=language_value,
                vocabulary=vocabulary_value.strip(),
                asr_model=asr_value,
                text_model=model_value,
            )
            return (
                f"Queued {len(jobs)} jobs for {len(built.recordings)} "
                f"recording(s).\n\n{batch_module.describe(built)}",
                job_table(conn),
            )

        def on_stop(job_id):
            return stop_jobs(conn, job_id), job_table(conn)

        def on_kill(job_id):
            return stop_jobs(conn, job_id, hard=True), job_table(conn)

        def on_clear():
            removed = db.clear_finished(conn)
            return f"Cleared {removed} finished job(s).", job_table(conn)

        def on_tick(notes):
            return job_table(conn), log.tail(), last_note(notes)

        identity = [media_dir, notes_dir, title, slug, domain, source]

        asr_model.change(on_asr_model, asr_model, device_output)
        batch_plan.click(
            on_batch_plan, [batch_dir, batch_prefix, batch_number], batch_output
        )
        batch_queue.click(
            on_batch_queue,
            [
                batch_dir,
                batch_prefix,
                batch_number,
                notes_dir,
                domain,
                language,
                vocabulary,
                asr_model,
                model,
            ],
            [batch_output, jobs],
        )
        refresh_media.click(on_refresh_media, media_dir, source)
        refresh_models.click(on_refresh_models, outputs=model)
        probe_button.click(on_probe, [media_dir, source], probe_output)
        queue_extract.click(
            lambda r, n, t, s, d, c: _queue("extract", "none", {}, r, n, t, s, d, c),
            identity,
            [status, jobs],
        )
        queue_transcribe.click(
            lambda a, lang, voc, r, n, t, s, d, c: _queue(
                "transcribe",
                a,
                {"language": lang, "vocabulary": voc.strip()},
                *(r, n, t, s, d, c),
            ),
            [asr_model, language, vocabulary, *identity],
            [status, jobs],
        )
        queue_chunk.click(
            lambda r, n, t, s, d, c: _queue("chunk", "none", {}, r, n, t, s, d, c),
            identity,
            [status, jobs],
        )
        queue_generate.click(
            lambda m, r, n, t, s, d, c: _queue("generate", m, {}, r, n, t, s, d, c),
            [model, *identity],
            [status, jobs],
        )
        queue_index.click(
            lambda r, n, t, s, d, c: _queue("index", "none", {}, r, n, t, s, d, c),
            identity,
            [status, jobs],
        )
        run_all.click(
            on_run_all,
            [*identity, language, vocabulary, asr_model, model],
            [status, jobs],
        )
        stop_button.click(on_stop, stop_id, [status, jobs])
        kill_button.click(on_kill, stop_id, [status, jobs])
        clear_button.click(on_clear, outputs=[status, jobs])
        timer.tick(on_tick, notes_dir, [jobs, logs, note_preview])

    return blocks


# Keyword arguments passed to Blocks.launch. Kept as data so a test can check
# the installed gradio still accepts them: the UI is the one module no unit
# test exercises end to end, and a renamed argument only shows up at startup.
LAUNCH_OPTIONS = {"server_name": "0.0.0.0", "server_port": 7860, "share": False}


def main() -> None:
    supervise_worker()
    build().queue().launch(**LAUNCH_OPTIONS)


if __name__ == "__main__":
    main()
