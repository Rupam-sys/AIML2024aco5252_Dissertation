"""Small Ollama chat client for the local RAG prototype."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_BASE_URL = "http://localhost:11434"


def chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    base_url: str | None = None,
    timeout: int = 120,
) -> str:
    """Send chat messages to Ollama's local API and return the response text."""
    model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
    base_url = (base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
    request = Request(
        f"{base_url}/api/chat",
        data=json.dumps(
            {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.2},
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama returned HTTP {error.code}. Is model '{model}' installed? "
            f"Details: {details}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Could not connect to Ollama at {base_url}. Start Ollama and try again."
        ) from error

    try:
        return result["message"]["content"].strip()
    except (KeyError, TypeError) as error:
        raise RuntimeError("Ollama returned an unexpected chat response.") from error
