# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""Queue consumer. Runs in its own process so the UI can come and go.

Copied from glosator's worker.py with the stage names changed. Every unit of
work is guarded by a ``stage_runs`` row, so a job re-queued after a crash picks
up where it stopped.

The one addition is the order the stages hold the card in. A job runs
``extract (CPU) -> transcribe (GPU) -> release -> generate (Ollama) -> unload``,
and the release in the middle is not a politeness: the text model alone is
larger than the card's 16 GB and Ollama sizes its offload to the VRAM that is
free when it loads, so an ASR model still resident would push layers of the
text model onto the CPU for the whole of the generate stage.
"""

from __future__ import annotations

import json
import signal
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app import chunk as chunk_module
from app import config, db, generate, index, log, media, ollama, transcribe

_running = True

Params = dict[str, Any]


def main() -> None:
    config.ensure_dirs()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    conn = db.connect()
    db.reset_running_jobs(conn)
    log.event("worker", "started")

    while _running:
        job = db.claim_job(conn)
        if job is None:
            time.sleep(config.WORKER_POLL_S)
            continue
        _run_job(conn, job)

    log.event("worker", "stopped")


def _stop(_signum: int, _frame: object) -> None:
    global _running
    _running = False
    # The ASR stage runs in a child process. Leaving it behind would leave the
    # card held by something nothing is waiting for, which is exactly what the
    # child exists to prevent.
    transcribe.kill_active_child()


def _run_job(conn: sqlite3.Connection, job: sqlite3.Row) -> None:
    stage = job["stage"]
    handlers: dict[str, Callable[[sqlite3.Connection, sqlite3.Row], None]] = {
        "extract": _do_extract,
        "transcribe": _do_transcribe,
        "chunk": _do_chunk,
        "generate": _do_generate,
        "index": _do_index,
    }
    handler = handlers.get(stage)
    if handler is None:
        db.finish_job(conn, job["id"], db.FAILED, f"unknown stage {stage}")
        return

    log.event(
        "worker", "job started", job=job["id"], job_stage=stage, model=job["model"]
    )
    try:
        handler(conn, job)
    except Exception as error:  # a failed job must not take the worker down
        db.finish_job(conn, job["id"], db.FAILED, repr(error))
        log.event(
            "worker", "job failed", job=job["id"], job_stage=stage, error=repr(error)
        )
        _release_vram(stage, job["model"])
        return

    status = db.STOPPED if db.stop_requested(conn, job["id"]) else db.DONE
    db.finish_job(conn, job["id"], status)
    log.event("worker", "job finished", job=job["id"], job_stage=stage, status=status)
    _release_vram(stage, job["model"])


# Stages that put a model on the card. The ASR model is released inside the
# transcribe stage itself, in a finally, because it lives in this process; only
# Ollama has to be told from outside.
_OLLAMA_STAGES = frozenset({"generate"})


def _release_vram(stage: str, model: str) -> None:
    """Hand the card back when a job ends.

    A model stays resident between the windows of one job, because reloading
    the weights for every window would dominate the run. Between jobs it must
    not: the next job's ASR stage needs the card, this machine runs other local
    models, and tachygraf is a batch job rather than a service.
    """
    if stage in _OLLAMA_STAGES:
        ollama.unload(model)


def _params(job: sqlite3.Row) -> Params:
    return json.loads(job["params"])


def _source(conn: sqlite3.Connection, job: sqlite3.Row) -> sqlite3.Row:
    return db.source_row(conn, job["source_id"])


def _domain(source: sqlite3.Row) -> str:
    """The field this recording belongs to, for the prompts and the note tags."""
    return source["domain"] or config.DEFAULT_DOMAIN


def _out_dir(source: sqlite3.Row) -> Path:
    """Where this recording's output goes. Falls back to the configured mount."""
    return Path(source["out_dir"]) if source["out_dir"] else config.OUT_DIR


def _do_extract(conn: sqlite3.Connection, job: sqlite3.Row) -> None:
    source = _source(conn, job)
    force = bool(_params(job).get("force"))
    db.stage_start(conn, job["source_id"], None, "extract", job["model"])
    info = media.run(source["slug"], Path(source["media_path"]), force)
    db.job_progress(conn, job["id"], 1, 1)
    db.stage_finish(
        conn,
        job["source_id"],
        None,
        "extract",
        job["model"],
        db.DONE,
        media.audio_path(source["slug"]),
    )
    log.event("extract", "ready", slug=source["slug"], duration=info.duration_hms)


def _do_transcribe(conn: sqlite3.Connection, job: sqlite3.Row) -> None:
    source = _source(conn, job)
    model = job["model"]
    params = _params(job)
    db.stage_start(conn, job["source_id"], None, "transcribe", model)
    # The duration comes from the WAV the extract stage wrote, which the ASR
    # model reports back, rather than from a second ffprobe of the original.
    # This stage works from /work, so it still runs when the recording itself
    # has been moved or deleted since the extract stage finished.
    path = transcribe.run(
        source["slug"],
        Path(source["media_path"]),
        model=model,
        language=params.get("language") or config.LANGUAGE,
        vocabulary=params.get("vocabulary") or "",
        out_dir=_out_dir(source),
    )
    db.job_progress(conn, job["id"], 1, 1)
    db.stage_finish(conn, job["source_id"], None, "transcribe", model, db.DONE, path)


def _do_chunk(conn: sqlite3.Connection, job: sqlite3.Row) -> None:
    source = _source(conn, job)
    db.stage_start(conn, job["source_id"], None, "chunk", job["model"])
    windows = chunk_module.plan(source["slug"])
    db.replace_chunks(conn, job["source_id"], windows)
    db.job_progress(conn, job["id"], len(windows), len(windows))
    db.stage_finish(conn, job["source_id"], None, "chunk", job["model"], db.DONE)


def _do_generate(conn: sqlite3.Connection, job: sqlite3.Row) -> None:
    source = _source(conn, job)
    model = job["model"]
    force = bool(_params(job).get("force"))
    windows = chunk_module.load_plan(source["slug"])
    rows = {row["ord"]: row["id"] for row in db.list_chunks(conn, job["source_id"])}
    source_name = Path(source["media_path"]).name

    db.job_progress(conn, job["id"], 0, len(windows))
    for position, window in enumerate(windows, start=1):
        if db.stop_requested(conn, job["id"]):
            return
        chunk_id = rows.get(window.ord)
        if not force and db.stage_done(
            conn, job["source_id"], chunk_id, "generate", model
        ):
            db.job_progress(conn, job["id"], position, len(windows))
            continue
        db.stage_start(conn, job["source_id"], chunk_id, "generate", model)
        # A window takes minutes, during which the progress column does not
        # move. Say what is being written, so the log tail shows a pulse.
        log.event(
            "generate",
            "writing note",
            window=window.ord,
            span=generate.span_text(window),
            position=f"{position}/{len(windows)}",
        )
        try:
            path = generate.run(
                window,
                slug=source["slug"],
                source_name=source_name,
                windows=len(windows),
                domain=_domain(source),
                model=model,
                out_dir=_out_dir(source),
            )
        except generate.NoteRejected as error:
            # The rule it broke is the message. Recorded against the window so
            # that a pilot run can be read as "which rule, how often", which is
            # how the next rule gets written (see docs/DECISIONS.md).
            db.stage_finish(
                conn,
                job["source_id"],
                chunk_id,
                "generate",
                model,
                db.FAILED,
                error=f"rule broken: {error}",
            )
            log.event(
                "generate",
                "response rejected",
                window=window.ord,
                rule=str(error),
                model=model,
            )
        except Exception as error:
            db.stage_finish(
                conn,
                job["source_id"],
                chunk_id,
                "generate",
                model,
                db.FAILED,
                error=repr(error),
            )
            log.event("generate", "window failed", window=window.ord, error=repr(error))
        else:
            db.stage_finish(
                conn, job["source_id"], chunk_id, "generate", model, db.DONE, path
            )
        db.job_progress(conn, job["id"], position, len(windows))


def _recorded_duration(slug: str) -> float:
    """How long the recording is, as the transcribe stage recorded it.

    Read from ``segments.json`` rather than by probing the original, so the
    deterministic stage stays runnable after the recording has been moved
    away. It is the stage that runs again after a stop, a failure or a
    --force, and it should not depend on anything outside /work and /out.
    """
    try:
        _segments, reported = transcribe.load_segments(slug)
    except (OSError, ValueError, KeyError):
        return 0.0
    return float(reported.get("duration_s") or 0.0)


def _do_index(conn: sqlite3.Connection, job: sqlite3.Row) -> None:
    source = _source(conn, job)
    db.stage_start(conn, job["source_id"], None, "index", job["model"])
    windows = chunk_module.load_plan(source["slug"])
    rows = {row["ord"]: row["id"] for row in db.list_chunks(conn, job["source_id"])}
    outputs = db.stage_outputs(conn, job["source_id"], "generate")
    # A note is named after a title the model chose, so its path cannot be
    # computed from the plan; it is read back from the row the generate stage
    # wrote. A window that never produced a note is simply not in the map.
    notes = [
        (window, outputs[rows[window.ord]])
        for window in windows
        if rows.get(window.ord) in outputs
    ]
    written = index.run(
        notes,
        source_name=Path(source["media_path"]).name,
        slug=source["slug"],
        title=source["title"],
        duration_s=_recorded_duration(source["slug"]),
        domain=_domain(source),
        out_dir=_out_dir(source),
    )
    db.job_progress(conn, job["id"], len(written), len(written))
    db.stage_finish(conn, job["source_id"], None, "index", job["model"], db.DONE)


if __name__ == "__main__":
    main()
