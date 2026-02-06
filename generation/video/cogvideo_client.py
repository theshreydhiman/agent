"""Async HTTP client for a CogVideoX inference server."""

from __future__ import annotations

import asyncio

import httpx
import structlog

from config import get_settings

log = structlog.get_logger(__name__)

# Retry configuration
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # seconds


class CogVideoError(Exception):
    """Raised when the CogVideoX server returns an unrecoverable error."""


class CogVideoClient:
    """Thin async wrapper around a CogVideoX HTTP inference server.

    Parameters
    ----------
    base_url:
        Root URL of the CogVideoX server (e.g. ``http://localhost:8100``).
        Defaults to the value in application settings.
    """

    def __init__(self, base_url: str | None = None) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.cogvideo_base_url).rstrip("/")
        self._client: httpx.AsyncClient | None = None

    # -- lifecycle helpers ---------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(connect=30.0, read=120.0, write=120.0, pool=30.0),
            )
        return self._client

    async def close(self) -> None:
        """Shut down the underlying HTTP connection pool."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # -- private request helper with retries ---------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> httpx.Response:
        """Issue an HTTP request with automatic retries and exponential backoff.

        Retries are only attempted for connection-level errors and 5xx server
        errors; 4xx client errors are raised immediately.
        """
        client = await self._get_client()
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                response = await client.request(method, path, **kwargs)

                if response.status_code >= 500:
                    last_exc = CogVideoError(
                        f"Server error {response.status_code}: {response.text}"
                    )
                    log.warning(
                        "cogvideo_server_error",
                        attempt=attempt + 1,
                        status=response.status_code,
                    )
                else:
                    response.raise_for_status()
                    return response

            except httpx.ConnectError as exc:
                last_exc = exc
                log.warning(
                    "cogvideo_connect_error",
                    attempt=attempt + 1,
                    error=str(exc),
                )
            except httpx.TimeoutException as exc:
                last_exc = exc
                log.warning(
                    "cogvideo_timeout",
                    attempt=attempt + 1,
                    error=str(exc),
                )

            if attempt < _MAX_RETRIES - 1:
                delay = _BACKOFF_BASE ** (attempt + 1)
                log.info("cogvideo_retry_backoff", delay_seconds=delay)
                await asyncio.sleep(delay)

        raise CogVideoError(
            f"CogVideoX request failed after {_MAX_RETRIES} retries"
        ) from last_exc

    # -- public API ----------------------------------------------------------

    async def generate_video(
        self,
        image_path: str,
        prompt: str,
        duration: float = 5.0,
        fps: int = 30,
        width: int = 1080,
        height: int = 1920,
    ) -> str:
        """Submit an image-to-video generation job.

        Parameters
        ----------
        image_path:
            Local path to the source image.
        prompt:
            Text prompt describing the desired motion / scene.
        duration:
            Video length in seconds.
        fps:
            Frames per second.
        width, height:
            Output resolution.

        Returns
        -------
        str
            The server-assigned ``job_id``.
        """
        log.info(
            "cogvideo_generate_request",
            image_path=image_path,
            prompt=prompt,
            duration=duration,
            fps=fps,
            resolution=f"{width}x{height}",
        )

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        response = await self._request(
            "POST",
            "/api/v1/generate",
            files={"image": ("source.png", image_bytes, "image/png")},
            data={
                "prompt": prompt,
                "duration": str(duration),
                "fps": str(fps),
                "width": str(width),
                "height": str(height),
            },
        )

        payload = response.json()
        job_id: str = payload["job_id"]
        log.info("cogvideo_job_created", job_id=job_id)
        return job_id

    async def get_status(self, job_id: str) -> dict:
        """Poll the status of a generation job.

        Returns
        -------
        dict
            ``{"status": "pending"|"processing"|"completed"|"failed",
              "progress": 0.0..1.0}``
        """
        response = await self._request("GET", f"/api/v1/jobs/{job_id}/status")
        return response.json()

    async def wait_for_completion(
        self,
        job_id: str,
        timeout: int = 600,
        poll_interval: float = 3.0,
    ) -> dict:
        """Block until a job reaches a terminal state.

        Parameters
        ----------
        job_id:
            The job identifier returned by :meth:`generate_video`.
        timeout:
            Maximum seconds to wait before raising.
        poll_interval:
            Seconds between status polls.

        Returns
        -------
        dict
            Final status payload from the server.

        Raises
        ------
        CogVideoError
            If the job fails or the timeout is exceeded.
        """
        log.info("cogvideo_wait_start", job_id=job_id, timeout=timeout)
        elapsed = 0.0

        while elapsed < timeout:
            status = await self.get_status(job_id)
            state = status.get("status", "unknown")
            progress = status.get("progress", 0.0)

            log.debug(
                "cogvideo_poll",
                job_id=job_id,
                state=state,
                progress=progress,
            )

            if state == "completed":
                log.info("cogvideo_job_completed", job_id=job_id)
                return status

            if state == "failed":
                error_msg = status.get("error", "unknown error")
                raise CogVideoError(
                    f"Video generation failed for job {job_id}: {error_msg}"
                )

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise CogVideoError(
            f"Timed out waiting for job {job_id} after {timeout}s"
        )

    async def download_video(self, job_id: str) -> bytes:
        """Download the finished video for a completed job.

        Returns
        -------
        bytes
            Raw video file bytes (typically MP4).
        """
        log.info("cogvideo_download", job_id=job_id)
        response = await self._request("GET", f"/api/v1/jobs/{job_id}/download")
        log.info("cogvideo_download_complete", job_id=job_id, size=len(response.content))
        return response.content
