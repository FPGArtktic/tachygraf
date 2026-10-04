# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""Command line front end. Queues work; the worker does it.

Copied from glosator's cli.py. Having one execution path means the pilot runs
exactly what the interface runs, and a stage queued here is indistinguishable
from the same stage queued in the browser.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app import config, db, media, worker

STAGES = ("extract", "transcribe", "chunk", "generate", "index")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    config.ensure_dirs()

    if args.command == "worker":
        worker.main()
        return 0

    conn = db.connect()
    if args.command == "queue":
        _print_queue(conn)
        return 0
    if args.command == "sources":
        _print_sources(conn)
        return 0
    if args.command == "add":
        source_id = _add(conn, args)
        print(f"source {source_id}")
        return 0
    return _queue_stage(conn, args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tachygraf")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="register a recording and where its notes go")
    add.add_argument("media", type=Path)
    add.add_argument(
        "--title",
        default=None,
        help="what the recording is called; defaults to the file name",
    )
    add.add_argument("--slug", help="defaults to a slug of the title")
    add.add_argument(
        "--domain",
        default=None,
        help="the field the recording belongs to, e.g. electronics, calculus, "
        "anatomy; it reaches the prompts and the note tags (default: "
        f"{config.DEFAULT_DOMAIN})",
    )
    add.add_argument(
        "--out",
        type=Path,
        default=None,
        help="staging folder the output is written to; never your vault "
        f"(default: {config.OUT_DIR})",
    )

    for stage in STAGES:
        stage_parser = sub.add_parser(stage, help=f"queue the {stage} stage")
        stage_parser.add_argument("--source", required=True, help="recording slug")
        stage_parser.add_argument("--model", default=None)
        stage_parser.add_argument("--force", action="store_true")
        if stage == "transcribe":
            stage_parser.add_argument(
                "--language",
                default=config.LANGUAGE,
                help="spoken language as a two-letter code, or 'auto'. Naming "
                "it is faster and safer than letting the model guess from the "
                f"first thirty seconds (default: {config.LANGUAGE})",
            )

    sub.add_parser("queue", help="list jobs")
    sub.add_parser("sources", help="list registered recordings")
    sub.add_parser("worker", help="run the queue consumer in the foreground")
    return parser


def _add(conn, args) -> int:
    title = args.title or args.media.stem
    slug = args.slug or media.slugify(title)
    out_dir = args.out.expanduser().resolve() if args.out else None
    return db.upsert_source(
        conn, slug, title, args.media.resolve(), out_dir, args.domain
    )


def _queue_stage(conn, args) -> int:
    source_id = db.source_id_for(conn, args.source)
    if source_id is None:
        raise SystemExit(
            f"unknown recording {args.source!r}; run 'tachygraf add' first"
        )
    model = args.model or _default_model(args.command)

    if args.force:
        db.clear_stage(conn, source_id, args.command)

    params: dict[str, object] = {"force": args.force}
    if args.command == "transcribe":
        params["language"] = args.language

    job_id = db.enqueue(conn, source_id, args.command, model, json.dumps(params))
    print(f"job {job_id} queued: {args.command} on {args.source} with {model}")
    return 0


def _default_model(stage: str) -> str:
    if stage == "generate":
        return config.TEXT_MODEL
    if stage == "transcribe":
        return config.ASR_MODEL
    return "none"


def _print_queue(conn) -> None:
    for job in db.recent_jobs(conn):
        print(
            f"{job['id']:>4}  {job['stage']:<10} {job['status']:<8} "
            f"{job['done']}/{job['total']}  {job['model']}  {job['error'] or ''}"
        )


def _print_sources(conn) -> None:
    for row in db.list_sources(conn):
        print(
            f"{row['slug']:<16} {row['domain'] or config.DEFAULT_DOMAIN:<20} "
            f"{row['media_path']}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
