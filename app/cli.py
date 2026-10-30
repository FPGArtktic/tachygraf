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

from app import batch as batch_module
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
    if args.command == "batch":
        return _batch(conn, args)
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

    batch = sub.add_parser(
        "batch",
        help="register every recording in a folder and queue all five stages "
        "for each of them",
        description="A folder of recordings, one set of notes per recording. "
        "The same lecture in two containers counts once: a video and the "
        "audio-only track beside it share a folder, a title and a length, and "
        "the audio one is used. Run with --dry-run first to see what was "
        "chosen and what was skipped.",
    )
    batch.add_argument("folder", type=Path)
    batch.add_argument(
        "--domain",
        default=None,
        help="the field every recording in this folder belongs to, e.g. "
        f"electronics (default: {config.DEFAULT_DOMAIN})",
    )
    batch.add_argument("--out", type=Path, default=None, help="staging folder")
    batch.add_argument(
        "--language",
        default=config.LANGUAGE,
        help=f"spoken language, or 'auto' (default: {config.LANGUAGE})",
    )
    batch.add_argument(
        "--vocabulary",
        default="",
        help="names and terms the recording uses, comma-separated, e.g. "
        "'Claude Code, CLAUDE.md, Anthropic'. A hint to speech recognition, "
        "spelled and cased exactly as the transcript should have them",
    )
    batch.add_argument("--asr-model", default=config.ASR_MODEL)
    batch.add_argument("--model", default=config.TEXT_MODEL, help="the text model")
    batch.add_argument(
        "--no-recurse",
        action="store_true",
        help="look only in the folder itself. The default descends, because a "
        "downloaded playlist is often one folder per recording",
    )
    batch.add_argument(
        "--number",
        action="store_true",
        help="prefix each slug with its position, so a course keeps its order "
        "in any listing: 01-diodes, 02-transistors",
    )
    batch.add_argument(
        "--prefix", default="", help="prefix every slug, e.g. the course name"
    )
    batch.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and queue nothing",
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
            stage_parser.add_argument(
                "--vocabulary",
                default="",
                help="names and terms the recording uses, comma-separated, e.g. "
                "'Claude Code, CLAUDE.md, Anthropic'. A hint to speech recognition, "
                "spelled and cased exactly as the transcript should have them",
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


def _batch(conn, args) -> int:
    """Plan the folder, print what was chosen, then queue it.

    The plan is printed either way, not only for --dry-run: the batch picks
    one file per recording on the user's behalf, and a choice made for someone
    has to be visible to them.
    """
    try:
        built = batch_module.plan(
            args.folder,
            recurse=not args.no_recurse,
            slug_prefix=args.prefix,
            number=args.number,
        )
    except media.MediaError as error:
        raise SystemExit(f"tachygraf: {error}") from error

    print(batch_module.describe(built))
    if not built.recordings:
        raise SystemExit("tachygraf: no recordings found; nothing queued")
    if args.dry_run:
        print("\n--dry-run: nothing was queued")
        return 0

    jobs = batch_module.queue(
        conn,
        built,
        domain=args.domain or config.DEFAULT_DOMAIN,
        out_dir=args.out.expanduser().resolve() if args.out else None,
        language=args.language,
        vocabulary=args.vocabulary,
        asr_model=args.asr_model,
        text_model=args.model,
    )
    print(f"\nqueued {len(jobs)} jobs for {len(built.recordings)} recording(s)")
    return 0


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
        params["vocabulary"] = args.vocabulary

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
