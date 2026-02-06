"""High-level video generation interface built on CogVideoX."""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from config import get_settings
from generation.video.cogvideo_client import CogVideoClient
from generation.video.effects import add_music, add_text_overlay

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Style presets -- each maps to a prompt fragment that steers CogVideoX
# towards a particular kind of motion.
# ---------------------------------------------------------------------------

VIDEO_STYLES: dict[str, str] = {
    "natural": (
        "subtle natural head movement, gentle blinking, soft breathing motion, "
        "photorealistic, smooth motion"
    ),
    "talking_head": (
        "speaking animation with natural lip sync, expressive hand gestures, "
        "slight head tilts, conversational body language"
    ),
    "fashion_showcase": (
        "confident model walk, slow outfit display turn, elegant posing, "
        "studio lighting, fashion editorial movement"
    ),
    "lifestyle_vlog": (
        "casual relaxed movement, looking around naturally, picking up objects, "
        "warm candid vibe, vlog-style framing"
    ),
    "ambient_aesthetic": (
        "very slow subtle motion, cinematic parallax drift, soft focus shifts, "
        "dreamy atmosphere, gentle wind movement"
    ),
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VideoResult:
    """Metadata and payload returned after video generation."""

    video_bytes: bytes
    file_path: str
    duration_seconds: float
    fps: int
    resolution: tuple[int, int]
    generation_time_seconds: float
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

_OUTPUT_DIR = "generated_videos"


class VideoGenerator:
    """Facade that combines CogVideoX generation with post-processing.

    Usage::

        gen = VideoGenerator()
        result = await gen.generate_video("photo.png", "A woman smiling softly")
        print(result.file_path)
    """

    def __init__(self) -> None:
        self._client = CogVideoClient()
        settings = get_settings()
        self._output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            _OUTPUT_DIR,
        )
        os.makedirs(self._output_dir, exist_ok=True)
        log.info("video_generator_init", output_dir=self._output_dir)

    # -- helpers -------------------------------------------------------------

    def _build_prompt(self, user_prompt: str, style: str) -> str:
        """Combine user prompt with style preset."""
        style_fragment = VIDEO_STYLES.get(style)
        if style_fragment is None:
            available = ", ".join(sorted(VIDEO_STYLES))
            log.warning(
                "unknown_video_style",
                style=style,
                available=available,
            )
            style_fragment = VIDEO_STYLES["natural"]
        return f"{user_prompt}. {style_fragment}"

    def _save_video(self, video_bytes: bytes, label: str) -> str:
        """Persist raw video bytes locally and return the absolute path."""
        filename = f"{label}_{uuid.uuid4().hex[:10]}.mp4"
        path = os.path.join(self._output_dir, filename)
        with open(path, "wb") as f:
            f.write(video_bytes)
        return os.path.abspath(path)

    # -- public API ----------------------------------------------------------

    async def generate_video(
        self,
        source_image_path: str,
        prompt: str,
        style: str = "natural",
        duration: float = 5.0,
        fps: int = 30,
        width: int = 1080,
        height: int = 1920,
    ) -> VideoResult:
        """Generate a short video clip from a static image.

        Parameters
        ----------
        source_image_path:
            Path to the source image file.
        prompt:
            Text description of desired motion / scene.
        style:
            One of the keys in :data:`VIDEO_STYLES`.
        duration:
            Length in seconds.
        fps:
            Frames per second of output.
        width, height:
            Output resolution.

        Returns
        -------
        VideoResult
        """
        full_prompt = self._build_prompt(prompt, style)
        log.info(
            "video_generate_start",
            source=source_image_path,
            style=style,
            duration=duration,
        )

        t0 = time.monotonic()

        job_id = await self._client.generate_video(
            image_path=source_image_path,
            prompt=full_prompt,
            duration=duration,
            fps=fps,
            width=width,
            height=height,
        )

        await self._client.wait_for_completion(job_id)
        video_bytes = await self._client.download_video(job_id)

        elapsed = time.monotonic() - t0

        file_path = self._save_video(video_bytes, "clip")

        log.info(
            "video_generate_done",
            file_path=file_path,
            generation_time=round(elapsed, 2),
            size_bytes=len(video_bytes),
        )

        return VideoResult(
            video_bytes=video_bytes,
            file_path=file_path,
            duration_seconds=duration,
            fps=fps,
            resolution=(width, height),
            generation_time_seconds=round(elapsed, 2),
            metadata={
                "job_id": job_id,
                "prompt": full_prompt,
                "style": style,
                "source_image": source_image_path,
            },
        )

    async def generate_reel(
        self,
        source_image_path: str,
        prompt: str,
        music_path: str | None = None,
        text_overlay: str | None = None,
        style: str = "natural",
        duration: float = 10.0,
        fps: int = 30,
        width: int = 1080,
        height: int = 1920,
    ) -> VideoResult:
        """Generate a social-media reel with optional music and text overlay.

        This is a higher-level wrapper around :meth:`generate_video` that also
        applies post-processing effects (music, text burn-in).

        Parameters
        ----------
        source_image_path:
            Source image file.
        prompt:
            Motion / scene description.
        music_path:
            Optional path to a background audio file.
        text_overlay:
            Optional text to burn into the bottom of the video.
        style:
            One of the keys in :data:`VIDEO_STYLES`.
        duration:
            Reel length in seconds.
        fps:
            Frames per second.
        width, height:
            Output resolution.

        Returns
        -------
        VideoResult
        """
        log.info(
            "reel_generate_start",
            source=source_image_path,
            music=music_path is not None,
            text=text_overlay is not None,
            duration=duration,
        )

        t0 = time.monotonic()

        # Step 1 -- generate the raw video clip.
        result = await self.generate_video(
            source_image_path=source_image_path,
            prompt=prompt,
            style=style,
            duration=duration,
            fps=fps,
            width=width,
            height=height,
        )

        current_path = result.file_path

        # Step 2 -- add text overlay if requested.
        if text_overlay:
            text_path = self._save_video(b"", "reel_text").replace(".mp4", "_tmp.mp4")
            current_path = add_text_overlay(
                video_path=current_path,
                text=text_overlay,
                position="bottom",
                font_size=48,
                output_path=text_path,
            )
            log.info("reel_text_applied", path=current_path)

        # Step 3 -- add music if provided.
        if music_path:
            music_out = self._save_video(b"", "reel_music").replace(".mp4", "_tmp.mp4")
            current_path = add_music(
                video_path=current_path,
                audio_path=music_path,
                output_path=music_out,
            )
            log.info("reel_music_applied", path=current_path)

        # Step 4 -- read final bytes and persist with a clean name.
        with open(current_path, "rb") as f:
            final_bytes = f.read()

        final_path = self._save_video(final_bytes, "reel")
        elapsed = time.monotonic() - t0

        log.info(
            "reel_generate_done",
            file_path=final_path,
            generation_time=round(elapsed, 2),
        )

        return VideoResult(
            video_bytes=final_bytes,
            file_path=final_path,
            duration_seconds=duration,
            fps=fps,
            resolution=(width, height),
            generation_time_seconds=round(elapsed, 2),
            metadata={
                "job_id": result.metadata.get("job_id"),
                "prompt": result.metadata.get("prompt"),
                "style": style,
                "source_image": source_image_path,
                "music_path": music_path,
                "text_overlay": text_overlay,
            },
        )

    async def close(self) -> None:
        """Release resources held by the underlying client."""
        await self._client.close()
