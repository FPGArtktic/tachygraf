# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""The one place that calls Ollama.

Copied from glosator. The ``images`` argument is gone with the vision pass it
served: tachygraf has one model pass, and a slide is only ever described here
if the speaker described it aloud. ``generate`` also asks for no thinking,
which glosator's models never did on their own (docs/DECISIONS.md).
"""

from __future__ import annotations

import ollama

from app import config, log


class OllamaUnavailable(RuntimeError):
    """Ollama could not be reached or the model is not pulled."""


def client() -> ollama.Client:
    return ollama.Client(host=config.OLLAMA_URL, timeout=config.OLLAMA_TIMEOUT_S)


def list_models() -> list[str]:
    """Model tags from ``GET /api/tags``, for the UI dropdown."""
    try:
        response = client().list()
    except Exception as error:  # ollama wraps httpx and socket errors
        raise OllamaUnavailable(f"{config.OLLAMA_URL}: {error}") from error
    return sorted(
        model.get("model") or model.get("name", "") for model in response["models"]
    )


def unload(model: str) -> None:
    """Drop a model from VRAM now, instead of after ``keep_alive`` expires.

    The card is shared with whatever else the user runs locally, and here it is
    also shared with the ASR stage of the next job, so tachygraf holds it only
    while a job is in flight. Failing to unload is not worth failing a finished
    job over; Ollama frees it on its own eventually.
    """
    try:
        client().generate(model=model, prompt="", keep_alive=0)
    except Exception as error:
        log.event("ollama", "could not unload model", model=model, error=repr(error))


def generate(model: str, prompt: str) -> str:
    """One completion. Blocks for as long as ``OLLAMA_TIMEOUT_S`` allows.

    Thinking is turned off explicitly, because gemma4 thinks unless told not
    to, and its thinking counts against ``NUM_PREDICT``. At the pipeline's
    temperature it can loop until that budget is gone and return nothing: a
    glossary check repeated 24 times, measured on gemma4:12b (PILOT.md,
    2026-10-30). Models without thinking accept the field and ignore it.
    """
    try:
        response = client().generate(
            model=model,
            prompt=prompt,
            think=False,
            keep_alive=config.OLLAMA_KEEP_ALIVE,
            options={
                "num_ctx": config.NUM_CTX,
                "temperature": config.TEMPERATURE,
                "num_predict": config.NUM_PREDICT,
            },
        )
    except Exception as error:
        raise OllamaUnavailable(f"{model} on {config.OLLAMA_URL}: {error}") from error
    text = response["response"].strip()
    if not text:
        raise OllamaUnavailable(f"{model} returned an empty response")
    return text
