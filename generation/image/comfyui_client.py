"""Async HTTP client for the ComfyUI API."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog

from config import get_settings

logger = structlog.get_logger(__name__)

_DEFAULT_RETRIES = 3
_DEFAULT_RETRY_DELAY = 2.0


class ComfyUIError(Exception):
    """Raised when the ComfyUI API returns an error."""


class ComfyUIClient:
    """Thin async wrapper around ComfyUI's REST / WebSocket API.

    Parameters
    ----------
    base_url:
        Root URL of the ComfyUI server (e.g. ``http://localhost:8188``).
        Falls back to ``settings.comfyui_base_url`` when *None*.
    """

    def __init__(self, base_url: str | None = None) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.comfyui_base_url).rstrip("/")
        self._client: httpx.AsyncClient | None = None

    # -- lifecycle -------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(30.0, connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # -- helpers ---------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        retries: int = _DEFAULT_RETRIES,
        retry_delay: float = _DEFAULT_RETRY_DELAY,
    ) -> Any:
        """Execute an HTTP request with automatic retries on connection errors."""
        client = await self._get_client()
        last_exc: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                response = await client.request(method, path, json=json)
                response.raise_for_status()
                return response.json()
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_exc = exc
                logger.warning(
                    "comfyui_connection_failed",
                    attempt=attempt,
                    max_retries=retries,
                    error=str(exc),
                )
                if attempt < retries:
                    await asyncio.sleep(retry_delay)
            except httpx.HTTPStatusError as exc:
                raise ComfyUIError(
                    f"ComfyUI returned {exc.response.status_code}: {exc.response.text}"
                ) from exc

        raise ComfyUIError(
            f"Failed to reach ComfyUI after {retries} attempts"
        ) from last_exc

    # -- public API ------------------------------------------------------------

    async def queue_prompt(self, workflow: dict) -> str:
        """Submit a workflow to ComfyUI and return the ``prompt_id``.

        Parameters
        ----------
        workflow:
            Full ComfyUI workflow dict (the ``"prompt"`` payload).
        """
        data = await self._request("POST", "/prompt", json={"prompt": workflow})
        prompt_id: str = data["prompt_id"]
        logger.info("comfyui_prompt_queued", prompt_id=prompt_id)
        return prompt_id

    async def get_history(self, prompt_id: str) -> dict:
        """Fetch execution history for *prompt_id*."""
        data = await self._request("GET", f"/history/{prompt_id}")
        return data.get(prompt_id, {})

    async def get_status(self, prompt_id: str) -> dict:
        """Return a lightweight status dict for *prompt_id*.

        Keys: ``"status"`` (``"pending"`` | ``"running"`` | ``"completed"`` |
        ``"error"``), plus raw ``"history"`` when available.
        """
        history = await self.get_history(prompt_id)
        if not history:
            return {"status": "pending", "history": {}}

        outputs = history.get("outputs", {})
        status_info = history.get("status", {})

        if status_info.get("status_str") == "error":
            return {"status": "error", "history": history}

        if outputs:
            return {"status": "completed", "history": history}

        return {"status": "running", "history": history}

    async def wait_for_completion(
        self,
        prompt_id: str,
        timeout: float = 300.0,
        poll_interval: float = 1.0,
    ) -> dict:
        """Poll ``get_status`` until the prompt completes or *timeout* elapses.

        Returns the final status dict.  Raises ``ComfyUIError`` on timeout or
        if the execution reports an error.
        """
        elapsed = 0.0
        while elapsed < timeout:
            status = await self.get_status(prompt_id)
            current = status["status"]

            if current == "completed":
                logger.info(
                    "comfyui_generation_completed",
                    prompt_id=prompt_id,
                    elapsed=round(elapsed, 1),
                )
                return status

            if current == "error":
                raise ComfyUIError(
                    f"Generation {prompt_id} failed: {status.get('history', {})}"
                )

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise ComfyUIError(
            f"Generation {prompt_id} timed out after {timeout}s"
        )

    async def get_output_images(self, prompt_id: str) -> list[bytes]:
        """Download all output images for a completed *prompt_id*.

        Returns a list of raw image bytes (PNG).
        """
        history = await self.get_history(prompt_id)
        if not history:
            raise ComfyUIError(f"No history found for prompt {prompt_id}")

        outputs = history.get("outputs", {})
        images: list[bytes] = []
        client = await self._get_client()

        for _node_id, node_output in outputs.items():
            for image_info in node_output.get("images", []):
                filename = image_info["filename"]
                subfolder = image_info.get("subfolder", "")
                img_type = image_info.get("type", "output")

                params = {
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": img_type,
                }

                for attempt in range(1, _DEFAULT_RETRIES + 1):
                    try:
                        resp = await client.get("/view", params=params)
                        resp.raise_for_status()
                        images.append(resp.content)
                        logger.debug(
                            "comfyui_image_downloaded",
                            filename=filename,
                            size_bytes=len(resp.content),
                        )
                        break
                    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                        if attempt == _DEFAULT_RETRIES:
                            raise ComfyUIError(
                                f"Failed to download image {filename}"
                            ) from exc
                        await asyncio.sleep(_DEFAULT_RETRY_DELAY)

        return images
