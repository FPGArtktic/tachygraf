# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""Structured logging: JSON lines on disk, readable tail for the UI.

Copied from glosator, unchanged but for the file name.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app import config

_MAX_TAIL_BYTES = 64_000


def log_path(day: str | None = None) -> Path:
    stamp = day or datetime.now(UTC).strftime("%Y-%m-%d")
    return config.LOG_DIR / f"tachygraf-{stamp}.jsonl"


def event(stage: str, message: str, **fields: Any) -> None:
    """Append one event. Never raises: losing a log line must not kill a job."""
    record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "stage": stage,
        "message": message,
        **fields,
    }
    line = json.dumps(record, ensure_ascii=False, default=str)
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def tail(lines: int = 60) -> str:
    """Last events of the current day, formatted for a textbox."""
    path = log_path()
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - _MAX_TAIL_BYTES))
        raw = handle.read().decode("utf-8", errors="replace")
    return "\n".join(
        _format(line) for line in raw.splitlines()[-lines:] if line.strip()
    )


def _format(line: str) -> str:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return line
    extra = {
        key: value
        for key, value in record.items()
        if key not in {"ts", "pid", "stage", "message"}
    }
    suffix = f"  {json.dumps(extra, ensure_ascii=False)}" if extra else ""
    stage = record.get("stage", "")
    return f"{record.get('ts', '')} [{stage}] {record.get('message', '')}{suffix}"
