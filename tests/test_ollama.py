# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""ollama.py is copied from glosator; these are its tests, against a fake client."""

from __future__ import annotations

import pytest

from app import config, ollama


class FakeClient:
    def __init__(self, response="an answer"):
        self.response = response
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return {"response": self.response}

    def list(self):
        return {"models": [{"model": "qwen2.5:32b"}, {"name": "gemma3:12b-it-qat"}]}


class BrokenClient:
    def generate(self, **kwargs):
        raise OSError("connection refused")

    def list(self):
        raise OSError("connection refused")


def test_generate_applies_the_configured_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeClient()
    monkeypatch.setattr(ollama, "client", lambda: fake)
    assert ollama.generate("qwen2.5:32b", "a prompt") == "an answer"
    call = fake.calls[0]
    assert call["options"]["num_ctx"] == config.NUM_CTX
    assert call["options"]["temperature"] == config.TEMPERATURE
    assert call["keep_alive"] == config.OLLAMA_KEEP_ALIVE


def test_generate_asks_for_no_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    """gemma4 thinks unless told not to, and its thinking spends NUM_PREDICT."""
    fake = FakeClient()
    monkeypatch.setattr(ollama, "client", lambda: fake)
    ollama.generate("gemma4:31b", "a prompt")
    assert fake.calls[0]["think"] is False


def test_generate_rejects_an_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama, "client", lambda: FakeClient("   "))
    with pytest.raises(ollama.OllamaUnavailable, match="empty response"):
        ollama.generate("qwen2.5:32b", "a prompt")


def test_unreachable_ollama_is_reported_with_its_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ollama, "client", lambda: BrokenClient())
    with pytest.raises(ollama.OllamaUnavailable, match=config.OLLAMA_URL):
        ollama.generate("qwen2.5:32b", "a prompt")
    with pytest.raises(ollama.OllamaUnavailable, match=config.OLLAMA_URL):
        ollama.list_models()


def test_list_models_reads_both_key_spellings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ollama, "client", lambda: FakeClient())
    assert ollama.list_models() == ["gemma3:12b-it-qat", "qwen2.5:32b"]


def test_a_failed_unload_is_logged_and_never_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finished job must not fail because the card could not be handed back."""
    monkeypatch.setattr(ollama, "client", lambda: BrokenClient())
    ollama.unload("qwen2.5:32b")

    from app import log

    assert "could not unload model" in log.tail()


def test_unload_asks_for_the_model_to_go_now(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeClient()
    monkeypatch.setattr(ollama, "client", lambda: fake)
    ollama.unload("qwen2.5:32b")
    assert fake.calls[0]["keep_alive"] == 0
