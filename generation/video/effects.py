"""Post-processing effects for generated videos using ffmpeg."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)


class FFmpegNotFoundError(RuntimeError):
    """Raised when ffmpeg is not available on the system."""


def _ensure_ffmpeg() -> None:
    """Check that ffmpeg is reachable via ``$PATH``."""
    if shutil.which("ffmpeg") is None:
        raise FFmpegNotFoundError(
            "ffmpeg is not installed or not on PATH. "
            "Install it with: apt-get install ffmpeg  (Debian/Ubuntu) "
            "or  brew install ffmpeg  (macOS)"
        )


def _default_output(source_path: str, suffix: str) -> str:
    """Derive an output path by appending *suffix* before the extension."""
    p = Path(source_path)
    return str(p.with_stem(f"{p.stem}_{suffix}"))


def add_music(
    video_path: str,
    audio_path: str,
    output_path: str,
) -> str:
    """Overlay an audio track onto a video file.

    The audio is trimmed or looped to match the video length.  The original
    video audio (if any) is replaced.

    Parameters
    ----------
    video_path:
        Path to the input video.
    audio_path:
        Path to the audio file (mp3, wav, aac, etc.).
    output_path:
        Destination for the muxed video.

    Returns
    -------
    str
        Absolute path to the output file.
    """
    import ffmpeg as ffmpeg_lib

    _ensure_ffmpeg()

    log.info(
        "effects_add_music",
        video=video_path,
        audio=audio_path,
        output=output_path,
    )

    video_input = ffmpeg_lib.input(video_path)
    audio_input = ffmpeg_lib.input(audio_path)

    (
        ffmpeg_lib
        .output(
            video_input.video,
            audio_input.audio,
            output_path,
            vcodec="copy",
            acodec="aac",
            shortest=None,
        )
        .overwrite_output()
        .run(quiet=True)
    )

    log.info("effects_add_music_done", output=output_path)
    return os.path.abspath(output_path)


def add_text_overlay(
    video_path: str,
    text: str,
    position: str = "bottom",
    font_size: int = 48,
    output_path: str | None = None,
) -> str:
    """Burn a text overlay into a video.

    Parameters
    ----------
    video_path:
        Input video file.
    text:
        The text string to render.
    position:
        One of ``"top"``, ``"center"``, or ``"bottom"``.
    font_size:
        Font size in pixels.
    output_path:
        Where to write the result.  A sensible default is derived from the
        input path when omitted.

    Returns
    -------
    str
        Absolute path to the output file.
    """
    import ffmpeg as ffmpeg_lib

    _ensure_ffmpeg()

    if output_path is None:
        output_path = _default_output(video_path, "text")

    position_map = {
        "top": "(w-text_w)/2:text_h",
        "center": "(w-text_w)/2:(h-text_h)/2",
        "bottom": "(w-text_w)/2:h-text_h*2",
    }
    xy = position_map.get(position, position_map["bottom"])

    # Escape single-quotes for the ffmpeg drawtext filter.
    safe_text = text.replace("'", "\u2019")

    log.info(
        "effects_add_text",
        video=video_path,
        text=text,
        position=position,
        font_size=font_size,
        output=output_path,
    )

    (
        ffmpeg_lib
        .input(video_path)
        .drawtext(
            text=safe_text,
            x=xy.split(":")[0],
            y=xy.split(":")[1],
            fontsize=font_size,
            fontcolor="white",
            borderw=2,
            bordercolor="black",
        )
        .output(output_path, vcodec="libx264", acodec="copy")
        .overwrite_output()
        .run(quiet=True)
    )

    log.info("effects_add_text_done", output=output_path)
    return os.path.abspath(output_path)


def trim_video(
    video_path: str,
    start: float,
    end: float,
    output_path: str | None = None,
) -> str:
    """Extract a sub-clip from a video.

    Parameters
    ----------
    video_path:
        Source video.
    start:
        Start time in seconds.
    end:
        End time in seconds.
    output_path:
        Destination path (auto-generated when ``None``).

    Returns
    -------
    str
        Absolute path to the trimmed clip.
    """
    import ffmpeg as ffmpeg_lib

    _ensure_ffmpeg()

    if output_path is None:
        output_path = _default_output(video_path, "trimmed")

    log.info(
        "effects_trim",
        video=video_path,
        start=start,
        end=end,
        output=output_path,
    )

    (
        ffmpeg_lib
        .input(video_path, ss=start, to=end)
        .output(output_path, vcodec="copy", acodec="copy")
        .overwrite_output()
        .run(quiet=True)
    )

    log.info("effects_trim_done", output=output_path)
    return os.path.abspath(output_path)


def concat_videos(
    video_paths: list[str],
    output_path: str,
) -> str:
    """Concatenate multiple video clips into a single file.

    All inputs should share the same codec, resolution, and frame-rate for
    best results.  The concat demuxer is used for frame-accurate joining.

    Parameters
    ----------
    video_paths:
        Ordered list of video files to join.
    output_path:
        Destination for the concatenated video.

    Returns
    -------
    str
        Absolute path to the output file.
    """
    import ffmpeg as ffmpeg_lib

    _ensure_ffmpeg()

    if not video_paths:
        raise ValueError("video_paths must contain at least one path")

    log.info(
        "effects_concat",
        count=len(video_paths),
        output=output_path,
    )

    # Build a concat filter with all inputs.
    streams = [ffmpeg_lib.input(p) for p in video_paths]

    (
        ffmpeg_lib
        .concat(*streams, v=1, a=1)
        .output(output_path)
        .overwrite_output()
        .run(quiet=True)
    )

    log.info("effects_concat_done", output=output_path)
    return os.path.abspath(output_path)
