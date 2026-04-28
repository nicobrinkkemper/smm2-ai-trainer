"""Minimal Ollama HTTP client.

Two endpoints we need:

- ``/api/generate`` — completion-style with ``system`` + ``prompt``
- ``/api/chat``     — chat-style with ``messages`` (matches dataset_v3 format)

We use ``urllib`` so this module has no third-party dependency. The client
is tiny on purpose: most testing happens at the ``runner`` level by passing
a stub client.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Optional


DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "300"))


class OllamaError(RuntimeError):
    pass


@dataclass
class OllamaResponse:
    text: str
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    total_duration_ns: Optional[int] = None

    @property
    def tokens_total(self) -> int:
        return (self.prompt_eval_count or 0) + (self.eval_count or 0)


class OllamaClient:
    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        *,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        num_predict: int = 4096,
    ) -> OllamaResponse:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
        if system is not None:
            payload["system"] = system
        body = self._post("/api/generate", payload)
        return OllamaResponse(
            text=str(body.get("response", "")).strip(),
            prompt_eval_count=body.get("prompt_eval_count"),
            eval_count=body.get("eval_count"),
            total_duration_ns=body.get("total_duration"),
        )

    def chat(
        self,
        *,
        model: str,
        messages: list[dict],
        temperature: float = 0.1,
        num_predict: int = 4096,
    ) -> OllamaResponse:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
        body = self._post("/api/chat", payload)
        msg = body.get("message", {}) or {}
        return OllamaResponse(
            text=str(msg.get("content", "")).strip(),
            prompt_eval_count=body.get("prompt_eval_count"),
            eval_count=body.get("eval_count"),
            total_duration_ns=body.get("total_duration"),
        )

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except Exception as e:
            raise OllamaError(f"Ollama request to {url} failed: {e}") from e
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise OllamaError(
                f"Ollama returned non-JSON for {url}: {e}\n{raw[:200]}"
            ) from e
