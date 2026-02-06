"""Async wrapper around the Ollama HTTP API for local LLM inference."""

from __future__ import annotations

import httpx
import structlog

from config import get_settings

log = structlog.get_logger(__name__)

_FALLBACK = "[LLM unavailable - configure Ollama]"


class LLMClient:
    """Lightweight async client that talks to a running Ollama instance."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model

    # -- public API -----------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Send a one-shot prompt to Ollama ``/api/generate`` and return the
        completed text.  Returns a fallback string on connection errors."""

        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        return await self._post("/api/generate", payload, key="response")

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
    ) -> str:
        """Send a multi-turn conversation to Ollama ``/api/chat`` and return
        the assistant reply."""

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        return await self._post("/api/chat", payload, key="message")

    # -- internals ------------------------------------------------------------

    async def _post(self, path: str, payload: dict, key: str) -> str:
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()

                if key == "message":
                    # /api/chat nests the text inside message.content
                    return data.get("message", {}).get("content", _FALLBACK)
                return data.get(key, _FALLBACK)

        except httpx.ConnectError:
            log.warning("ollama_connect_error", url=url)
            return _FALLBACK
        except httpx.HTTPStatusError as exc:
            log.warning("ollama_http_error", status=exc.response.status_code)
            return _FALLBACK
        except Exception as exc:  # noqa: BLE001
            log.warning("ollama_unexpected_error", error=str(exc))
            return _FALLBACK
