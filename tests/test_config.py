# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""config is the only module that reads the environment, and this checks that."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from app import config

APP = Path(__file__).resolve().parent.parent / "app"


def test_only_config_reads_the_environment() -> None:
    """A tunable read anywhere else is a tunable nobody can find."""
    offenders = []
    for path in sorted(APP.rglob("*.py")):
        if path.name == "config.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "environ"
                and isinstance(node.value, ast.Name)
                and node.value.id == "os"
            ):
                offenders.append(f"{path.name}:{node.lineno}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "getenv"
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []


def test_the_environment_variables_are_the_documented_ones(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The names in the README's configuration table, and nothing else."""
    for name, value in (
        ("TACHYGRAF_IN", str(tmp_path / "recordings")),
        ("TACHYGRAF_OUT", str(tmp_path / "staging")),
        ("TACHYGRAF_WORK", str(tmp_path / "w")),
        ("TACHYGRAF_STATE", str(tmp_path / "s")),
        ("TACHYGRAF_TEXT_MODEL", "gemma3:12b-it-qat"),
        ("TACHYGRAF_ASR_MODEL", "large-v3-turbo"),
        ("TACHYGRAF_LANGUAGE", "pl"),
        ("TACHYGRAF_DOMAIN", "optics"),
        ("OLLAMA_URL", "http://host.containers.internal:11434"),
    ):
        monkeypatch.setenv(name, value)
    reloaded = importlib.reload(config)
    try:
        expected = {
            "IN_DIR": tmp_path / "recordings",
            "OUT_DIR": tmp_path / "staging",
            "WORK_DIR": tmp_path / "w",
            "STATE_DIR": tmp_path / "s",
            "TEXT_MODEL": "gemma3:12b-it-qat",
            "ASR_MODEL": "large-v3-turbo",
            "LANGUAGE": "pl",
            "DEFAULT_DOMAIN": "optics",
            "OLLAMA_URL": "http://host.containers.internal:11434",
            # The log and the database follow the state directory.
            "LOG_DIR": tmp_path / "s" / "logs",
            "DB_PATH": tmp_path / "s" / "tachygraf.db",
        }
        assert {name: getattr(reloaded, name) for name in expected} == expected
    finally:
        # Put the module back the way the rest of the suite expects it.
        monkeypatch.undo()
        importlib.reload(config)


def test_the_defaults_are_the_container_mounts() -> None:
    fresh = importlib.reload(config)
    try:
        expected = {
            "IN_DIR": Path("/in"),
            "OUT_DIR": Path("/out"),
            "WORK_DIR": Path("/work"),
            "STATE_DIR": Path("/state"),
            "TEXT_MODEL": "qwen2.5:32b",
            "ASR_MODEL": "large-v3",
        }
        assert {name: getattr(fresh, name) for name in expected} == expected
    finally:
        importlib.reload(config)


def test_the_window_budget_is_a_range() -> None:
    assert config.CHUNK_MIN_TOKENS < config.CHUNK_MAX_TOKENS


def test_a_window_break_is_a_longer_pause_than_a_paragraph_break() -> None:
    """Every paragraph break is a pause; a window break should be a real one."""
    assert config.PAUSE_SPLIT_S > config.PARAGRAPH_PAUSE_S


def test_the_window_budget_fits_the_context_window() -> None:
    """The prompt's own rules are long; the window must leave room for them."""
    assert config.CHUNK_MAX_TOKENS < config.NUM_CTX


def test_the_asr_vram_requirements_are_measured_per_model() -> None:
    """large-v3 does not fit the reference card; turbo does. Both are measured.

    A requirement above the card's capacity is deliberate: it is how the stage
    says "not on this card" before loading anything.
    """
    assert config.ASR_MIN_FREE_MB["large-v3-turbo"] < 4096
    assert config.ASR_MIN_FREE_MB["large-v3"] > 4096
    assert 0 < config.ASR_MIN_FREE_DEFAULT_MB < 4096
    # Every model the interface offers has a measured figure.
    from app import ui

    assert set(ui.ASR_MODELS) <= set(config.ASR_MIN_FREE_MB)


def test_the_card_does_not_run_whisper_in_int8() -> None:
    """int8_float16 looped and dropped sentences on the RTX 5070 Ti, silently.

    No exception, so the CPU fallback never saw it; only reading the transcript
    did. See docs/DECISIONS.md, 2026-10-30.
    """
    assert "int8" not in config.ASR_COMPUTE_GPU


def test_the_prompt_version_is_an_integer_for_the_frontmatter() -> None:
    assert isinstance(config.PROMPT_VERSION, int)
    assert config.PROMPT_VERSION >= 1


def test_ensure_dirs_creates_what_we_write_to_and_not_the_read_only_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, relative in (
        ("IN_DIR", "in"),
        ("WORK_DIR", "work"),
        ("STATE_DIR", "state"),
        ("LOG_DIR", "state/logs"),
    ):
        monkeypatch.setattr(config, name, tmp_path / "fresh" / relative)
    config.ensure_dirs()
    assert (tmp_path / "fresh" / "work").is_dir()
    assert (tmp_path / "fresh" / "state" / "logs").is_dir()
    assert not (tmp_path / "fresh" / "in").exists()
