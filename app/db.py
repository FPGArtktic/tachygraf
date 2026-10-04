# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""SQLite schema and state queries.

Copied from glosator with its renames: ``books`` is ``sources``, ``pdf_path``
is ``media_path``, and a chunk is bounded by two timestamps instead of two page
numbers (see docs/DECISIONS.md). The part that matters is unchanged: one row
per (source, chunk, stage, model) in ``stage_runs`` is what makes the pipeline
resumable, so a stage that finds its row with status ``done`` skips the work
unless the caller forces a rerun.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from app import config

# Applied in order; the list index is the schema version. Append only, never
# edit an existing entry (see docs/DECISIONS.md).
_MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE sources (
        id          INTEGER PRIMARY KEY,
        slug        TEXT NOT NULL UNIQUE,
        title       TEXT NOT NULL,
        media_path  TEXT NOT NULL,
        out_dir     TEXT,
        domain      TEXT,
        created     TEXT NOT NULL
    );

    -- One row per window of the recording, which is one note. A window is
    -- identified by its position, because unlike a book's section a window has
    -- no name until the generate stage has read it.
    CREATE TABLE chunks (
        id          INTEGER PRIMARY KEY,
        source_id   INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
        ord         INTEGER NOT NULL,
        start_s     REAL NOT NULL,
        end_s       REAL NOT NULL,
        labels      TEXT NOT NULL,
        text_path   TEXT NOT NULL,
        UNIQUE (source_id, ord)
    );

    CREATE TABLE stage_runs (
        id          INTEGER PRIMARY KEY,
        source_id   INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
        chunk_id    INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
        stage       TEXT NOT NULL,
        model       TEXT NOT NULL,
        status      TEXT NOT NULL,
        started     TEXT,
        finished    TEXT,
        output_path TEXT,
        error       TEXT
    );

    -- chunk_id is NULL for whole-recording stages and SQLite treats NULLs as
    -- distinct, so the identity of a run is indexed over IFNULL(chunk_id, -1).
    CREATE UNIQUE INDEX idx_stage_runs_key
        ON stage_runs (source_id, IFNULL(chunk_id, -1), stage, model);

    CREATE TABLE jobs (
        id        INTEGER PRIMARY KEY,
        source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
        stage     TEXT NOT NULL,
        model     TEXT NOT NULL,
        params    TEXT NOT NULL,
        status    TEXT NOT NULL,
        stop      INTEGER NOT NULL DEFAULT 0,
        total     INTEGER NOT NULL DEFAULT 0,
        done      INTEGER NOT NULL DEFAULT 0,
        created   TEXT NOT NULL,
        started   TEXT,
        finished  TEXT,
        error     TEXT
    );

    CREATE INDEX idx_jobs_status ON jobs(status, id);
    """,
)

# Stand-in for a NULL chunk_id in lookups and in the unique index. Rows still
# store NULL, so the foreign key to chunks stays intact.
_NULL_CHUNK = -1


def _key(chunk_id: int | None) -> int:
    return _NULL_CHUNK if chunk_id is None else chunk_id


DONE = "done"
FAILED = "failed"
RUNNING = "running"
QUEUED = "queued"
STOPPED = "stopped"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread is off because Gradio serves each request on its own
    # thread; writes are short and WAL handles the concurrency.
    conn = sqlite3.connect(
        path, timeout=30.0, isolation_level=None, check_same_thread=False
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    init(conn)
    return conn


def init(conn: sqlite3.Connection) -> None:
    """Bring an open database up to the current schema version."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for index in range(version, len(_MIGRATIONS)):
        # BEGIN and COMMIT belong inside the script: executescript commits any
        # transaction opened around it, so an outer BEGIN would be lost.
        conn.executescript(
            f"BEGIN;\n{_MIGRATIONS[index]}\nPRAGMA user_version = {index + 1};\nCOMMIT;"
        )


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[None]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def upsert_source(
    conn: sqlite3.Connection,
    slug: str,
    title: str,
    media_path: Path,
    out_dir: Path | None = None,
    domain: str | None = None,
) -> int:
    conn.execute(
        """
        INSERT INTO sources (slug, title, media_path, out_dir, domain, created)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (slug) DO UPDATE SET title = excluded.title,
                                         media_path = excluded.media_path,
                                         out_dir = excluded.out_dir,
                                         domain = excluded.domain
        """,
        (
            slug,
            title,
            str(media_path),
            str(out_dir) if out_dir else None,
            domain,
            _now(),
        ),
    )
    return int(
        conn.execute("SELECT id FROM sources WHERE slug = ?", (slug,)).fetchone()[0]
    )


def source_row(conn: sqlite3.Connection, source_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
    if row is None:
        raise KeyError(f"no source with id {source_id}")
    return row


def source_id_for(conn: sqlite3.Connection, slug: str) -> int | None:
    row = conn.execute("SELECT id FROM sources WHERE slug = ?", (slug,)).fetchone()
    return None if row is None else int(row[0])


def list_sources(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute("SELECT * FROM sources ORDER BY slug").fetchall())


def replace_chunks(conn: sqlite3.Connection, source_id: int, chunks: Iterable) -> None:
    """Store the window plan, keeping the rows of windows that did not change.

    Divergence from glosator, where re-planning drops every chunk row and the
    cascade takes the stage rows with it (see docs/DECISIONS.md). There a plan
    is drawn from page ranges the operator chose, so re-planning means the
    operator wants different notes. Here a plan is a pure function of
    ``segments.json``: re-running the chunk stage on an unchanged transcript
    produces an identical plan, and the notes already written from it are still
    the right notes. Dropping their stage rows would regenerate hours of 32B
    output for nothing, and "the container can be killed at any point; the next
    run resumes" is the whole point of the table.

    A window is unchanged when its position, its span, its labels and its text
    file are all the same. One that differs in any of them is replaced, and the
    cascade does drop its stage rows, because a note made from it was made from
    different speech.
    """
    wanted = {
        window.ord: (
            window.start_s,
            window.end_s,
            " ".join(window.labels),
            str(window.text_path),
        )
        for window in chunks
    }
    with transaction(conn):
        existing = conn.execute(
            "SELECT * FROM chunks WHERE source_id = ?", (source_id,)
        ).fetchall()
        for row in existing:
            current = (row["start_s"], row["end_s"], row["labels"], row["text_path"])
            if wanted.get(row["ord"]) != current:
                conn.execute("DELETE FROM chunks WHERE id = ?", (row["id"],))
        conn.executemany(
            """
            INSERT INTO chunks
                (source_id, ord, start_s, end_s, labels, text_path)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (source_id, ord) DO NOTHING
            """,
            [(source_id, order, *values) for order, values in sorted(wanted.items())],
        )


def list_chunks(conn: sqlite3.Connection, source_id: int) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT * FROM chunks WHERE source_id = ? ORDER BY ord", (source_id,)
        ).fetchall()
    )


def stage_done(
    conn: sqlite3.Connection,
    source_id: int,
    chunk_id: int | None,
    stage: str,
    model: str,
) -> bool:
    row = conn.execute(
        """
        SELECT status FROM stage_runs
         WHERE source_id = ? AND IFNULL(chunk_id, -1) = ? AND stage = ? AND model = ?
        """,
        (source_id, _key(chunk_id), stage, model),
    ).fetchone()
    return row is not None and row["status"] == DONE


def stage_start(
    conn: sqlite3.Connection,
    source_id: int,
    chunk_id: int | None,
    stage: str,
    model: str,
) -> None:
    conn.execute(
        """
        INSERT INTO stage_runs
            (source_id, chunk_id, stage, model, status, started)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (source_id, IFNULL(chunk_id, -1), stage, model) DO UPDATE
            SET status = excluded.status,
                started = excluded.started,
                finished = NULL,
                error = NULL
        """,
        (source_id, chunk_id, stage, model, RUNNING, _now()),
    )


def stage_finish(
    conn: sqlite3.Connection,
    source_id: int,
    chunk_id: int | None,
    stage: str,
    model: str,
    status: str,
    output_path: Path | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE stage_runs
           SET status = ?, finished = ?, output_path = ?, error = ?
         WHERE source_id = ? AND IFNULL(chunk_id, -1) = ? AND stage = ? AND model = ?
        """,
        (
            status,
            _now(),
            str(output_path) if output_path else None,
            error,
            source_id,
            _key(chunk_id),
            stage,
            model,
        ),
    )


def clear_stage(
    conn: sqlite3.Connection, source_id: int, stage: str, model: str | None = None
) -> int:
    """Forget a stage's results so ``--force`` reruns it."""
    # A stage may store one row per sub-unit as "<stage>:<id>", so a stage name
    # also matches its colon-suffixed variants.
    like = f"{stage}:%"
    if model is None:
        cursor = conn.execute(
            """
            DELETE FROM stage_runs
             WHERE source_id = ? AND (stage = ? OR stage LIKE ?)
            """,
            (source_id, stage, like),
        )
    else:
        cursor = conn.execute(
            """
            DELETE FROM stage_runs
             WHERE source_id = ? AND (stage = ? OR stage LIKE ?) AND model = ?
            """,
            (source_id, stage, like, model),
        )
    return cursor.rowcount


def stage_rows(
    conn: sqlite3.Connection, source_id: int, stage: str
) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT * FROM stage_runs WHERE source_id = ? AND stage = ? ORDER BY id",
            (source_id, stage),
        ).fetchall()
    )


def stage_outputs(
    conn: sqlite3.Connection, source_id: int, stage: str
) -> dict[int, Path]:
    """Chunk id to the file a finished stage wrote for it.

    The index stage needs the notes' paths and cannot compute them, because a
    note is named after a title the model chose (see docs/DECISIONS.md). The
    newest finished row for a chunk wins, whatever model wrote it: regenerating
    one window with another model should not drop it out of the map.
    """
    rows = conn.execute(
        """
        SELECT chunk_id, output_path FROM stage_runs
         WHERE source_id = ? AND stage = ? AND status = ?
               AND chunk_id IS NOT NULL AND output_path IS NOT NULL
         ORDER BY id
        """,
        (source_id, stage, DONE),
    ).fetchall()
    return {int(row["chunk_id"]): Path(row["output_path"]) for row in rows}


def enqueue(
    conn: sqlite3.Connection,
    source_id: int,
    stage: str,
    model: str,
    params: str = "{}",
    total: int = 0,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO jobs (source_id, stage, model, params, status, total, created)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (source_id, stage, model, params, QUEUED, total, _now()),
    )
    return int(cursor.lastrowid)


def claim_job(conn: sqlite3.Connection) -> sqlite3.Row | None:
    """Take the oldest queued job. Safe against a second worker process."""
    with transaction(conn):
        row = conn.execute(
            "SELECT * FROM jobs WHERE status = ? ORDER BY id LIMIT 1", (QUEUED,)
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE jobs SET status = ?, started = ? WHERE id = ?",
            (RUNNING, _now(), row["id"]),
        )
    return conn.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()


def job_progress(conn: sqlite3.Connection, job_id: int, done: int, total: int) -> None:
    conn.execute(
        "UPDATE jobs SET done = ?, total = ? WHERE id = ?", (done, total, job_id)
    )


def finish_job(
    conn: sqlite3.Connection, job_id: int, status: str, error: str | None = None
) -> None:
    conn.execute(
        "UPDATE jobs SET status = ?, finished = ?, error = ? WHERE id = ?",
        (status, _now(), error, job_id),
    )


def request_stop(conn: sqlite3.Connection, job_id: int) -> None:
    """Ask a running job to stop after the window it is on."""
    conn.execute("UPDATE jobs SET stop = 1 WHERE id = ?", (job_id,))
    conn.execute(
        "UPDATE jobs SET status = ?, finished = ? WHERE id = ? AND status = ?",
        (STOPPED, _now(), job_id, QUEUED),
    )


def stop_all(conn: sqlite3.Connection) -> list[int]:
    """Ask every unfinished job to stop. Returns the ids it asked.

    Stopping only the running job would be no use: the stages of one recording
    are queued together, so the next would start a second later.
    """
    rows = conn.execute(
        "SELECT id FROM jobs WHERE status IN (?, ?) ORDER BY id", (QUEUED, RUNNING)
    ).fetchall()
    for row in rows:
        request_stop(conn, row["id"])
    return [int(row["id"]) for row in rows]


def stop_requested(conn: sqlite3.Connection, job_id: int) -> bool:
    row = conn.execute("SELECT stop FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return row is not None and bool(row["stop"])


def clear_finished(conn: sqlite3.Connection) -> int:
    """Forget jobs that are over, so the table shows only live work.

    A failure from an hour ago sitting under a running job reads as a current
    problem. Only the queue rows go; what each stage completed is recorded in
    stage_runs and is what makes a rerun skip finished work.
    """
    cursor = conn.execute(
        "DELETE FROM jobs WHERE status IN (?, ?, ?)", (DONE, FAILED, STOPPED)
    )
    return cursor.rowcount


def recent_jobs(conn: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
    return list(
        conn.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    )


def reset_running_jobs(conn: sqlite3.Connection) -> None:
    """Settle the jobs a killed worker left marked running.

    One that was asked to stop is finished as stopped: the kill is how the
    stop was carried out, and re-queueing it would start the work again.
    """
    conn.execute(
        "UPDATE jobs SET status = ?, started = NULL WHERE status = ? AND stop = 0",
        (QUEUED, RUNNING),
    )
    conn.execute(
        "UPDATE jobs SET status = ?, finished = ? WHERE status = ? AND stop = 1",
        (STOPPED, _now(), RUNNING),
    )
