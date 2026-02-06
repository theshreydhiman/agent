"""Content validation for Instagram publishing.

Validates images, videos, carousels, and captions against Instagram's
technical requirements before attempting to publish via the Graph API.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import structlog
from PIL import Image

try:
    import ffmpeg
except ImportError:
    ffmpeg = None  # type: ignore[assignment]

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Instagram technical limits
# ---------------------------------------------------------------------------

ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG"}
MAX_IMAGE_SIZE_BYTES = 8 * 1024 * 1024  # 8 MB
MIN_IMAGE_WIDTH = 320
MAX_IMAGE_WIDTH = 1440
# Aspect-ratio bounds expressed as width/height
FEED_MIN_ASPECT = Fraction(4, 5)     # 0.80  (portrait)
FEED_MAX_ASPECT = Fraction(191, 100) # 1.91  (landscape)

ALLOWED_VIDEO_FORMATS = {".mp4"}
MAX_VIDEO_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB for Reels
MIN_VIDEO_DURATION = 3.0   # seconds
MAX_VIDEO_DURATION = 90.0  # seconds
MIN_VIDEO_WIDTH = 500
REEL_ASPECT = Fraction(9, 16)
REEL_ASPECT_TOLERANCE = 0.02  # small tolerance for float comparison

MAX_CAPTION_LENGTH = 2200
MAX_HASHTAG_COUNT = 30
BANNED_CAPTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)\bfollow\s*for\s*follow\b"),
    re.compile(r"(?i)\bf4f\b"),
    re.compile(r"(?i)\bl4l\b"),
    re.compile(r"(?i)\blike\s*for\s*like\b"),
]

CAROUSEL_MIN_ITEMS = 2
CAROUSEL_MAX_ITEMS = 10
CAROUSEL_ASPECT_TOLERANCE = 0.01


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Outcome of a content validation check."""

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def merge(self, other: ValidationResult) -> None:
        """Merge another result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.valid:
            self.valid = False


# ---------------------------------------------------------------------------
# ContentValidator
# ---------------------------------------------------------------------------

class ContentValidator:
    """Validates media and text content against Instagram requirements."""

    # -- Image validation ---------------------------------------------------

    @staticmethod
    def validate_image(image_path: str) -> ValidationResult:
        """Validate a single image for Instagram feed posting.

        Checks performed:
        - File exists
        - Format is JPEG or PNG
        - File size <= 8 MB
        - Width between 320 px and 1440 px
        - Aspect ratio between 4:5 and 1.91:1
        """
        result = ValidationResult()
        path = Path(image_path)

        # Existence
        if not path.exists():
            result.add_error(f"File not found: {image_path}")
            return result

        # File size
        size = path.stat().st_size
        if size > MAX_IMAGE_SIZE_BYTES:
            result.add_error(
                f"Image exceeds 8 MB limit ({size / (1024 * 1024):.1f} MB)"
            )

        # Open with Pillow
        try:
            img = Image.open(path)
        except Exception as exc:
            result.add_error(f"Cannot open image: {exc}")
            return result

        # Format
        fmt = img.format
        if fmt not in ALLOWED_IMAGE_FORMATS:
            result.add_error(
                f"Unsupported format '{fmt}'. Allowed: {', '.join(sorted(ALLOWED_IMAGE_FORMATS))}"
            )

        width, height = img.size

        # Resolution
        if width < MIN_IMAGE_WIDTH:
            result.add_error(f"Width {width}px is below minimum {MIN_IMAGE_WIDTH}px")
        if width > MAX_IMAGE_WIDTH:
            result.add_warning(
                f"Width {width}px exceeds recommended {MAX_IMAGE_WIDTH}px; "
                "Instagram will downscale the image"
            )

        # Aspect ratio
        if height > 0:
            aspect = Fraction(width, height)
            if aspect < FEED_MIN_ASPECT:
                result.add_error(
                    f"Aspect ratio {float(aspect):.2f} is narrower than "
                    f"minimum 4:5 ({float(FEED_MIN_ASPECT):.2f})"
                )
            if aspect > FEED_MAX_ASPECT:
                result.add_error(
                    f"Aspect ratio {float(aspect):.2f} is wider than "
                    f"maximum 1.91:1 ({float(FEED_MAX_ASPECT):.2f})"
                )
        else:
            result.add_error("Image has zero height")

        img.close()

        log.info(
            "image_validation_complete",
            path=image_path,
            valid=result.valid,
            error_count=len(result.errors),
        )
        return result

    # -- Video validation ---------------------------------------------------

    @staticmethod
    def validate_video(video_path: str) -> ValidationResult:
        """Validate a video for Instagram Reels.

        Checks performed:
        - File exists
        - Format is MP4
        - File size <= 100 MB
        - Duration between 3 s and 90 s
        - Width >= 500 px
        - Aspect ratio ~9:16
        """
        result = ValidationResult()
        path = Path(video_path)

        if not path.exists():
            result.add_error(f"File not found: {video_path}")
            return result

        # Extension
        if path.suffix.lower() not in ALLOWED_VIDEO_FORMATS:
            result.add_error(
                f"Unsupported video format '{path.suffix}'. Must be .mp4"
            )

        # File size
        size = path.stat().st_size
        if size > MAX_VIDEO_SIZE_BYTES:
            result.add_error(
                f"Video exceeds 100 MB limit ({size / (1024 * 1024):.1f} MB)"
            )

        # Probe with ffmpeg-python
        if ffmpeg is None:
            result.add_warning(
                "ffmpeg-python is not installed; skipping duration/resolution checks"
            )
            log.warning("ffmpeg_not_available", path=video_path)
            return result

        try:
            probe = ffmpeg.probe(video_path)
        except ffmpeg.Error as exc:
            result.add_error(f"ffprobe failed: {exc}")
            return result

        video_streams = [
            s for s in probe.get("streams", []) if s.get("codec_type") == "video"
        ]
        if not video_streams:
            result.add_error("No video stream found in file")
            return result

        stream = video_streams[0]
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        duration = float(probe.get("format", {}).get("duration", 0))

        # Duration
        if duration < MIN_VIDEO_DURATION:
            result.add_error(
                f"Duration {duration:.1f}s is shorter than minimum {MIN_VIDEO_DURATION}s"
            )
        if duration > MAX_VIDEO_DURATION:
            result.add_error(
                f"Duration {duration:.1f}s exceeds maximum {MAX_VIDEO_DURATION}s"
            )

        # Resolution
        if width < MIN_VIDEO_WIDTH:
            result.add_error(f"Width {width}px is below minimum {MIN_VIDEO_WIDTH}px")

        # Aspect ratio (9:16 for Reels)
        if height > 0 and width > 0:
            actual_aspect = width / height
            expected_aspect = float(REEL_ASPECT)
            if abs(actual_aspect - expected_aspect) > REEL_ASPECT_TOLERANCE:
                result.add_warning(
                    f"Aspect ratio {actual_aspect:.3f} differs from recommended "
                    f"9:16 ({expected_aspect:.3f}) for Reels"
                )

        log.info(
            "video_validation_complete",
            path=video_path,
            valid=result.valid,
            error_count=len(result.errors),
            duration=duration,
            width=width,
            height=height,
        )
        return result

    # -- Carousel validation ------------------------------------------------

    @staticmethod
    def validate_carousel(image_paths: list[str]) -> ValidationResult:
        """Validate a set of images for a carousel post.

        Checks performed:
        - Between 2 and 10 images
        - Each image individually valid
        - All images share the same aspect ratio
        """
        result = ValidationResult()

        count = len(image_paths)
        if count < CAROUSEL_MIN_ITEMS:
            result.add_error(
                f"Carousel requires at least {CAROUSEL_MIN_ITEMS} images, got {count}"
            )
            return result
        if count > CAROUSEL_MAX_ITEMS:
            result.add_error(
                f"Carousel allows at most {CAROUSEL_MAX_ITEMS} images, got {count}"
            )

        aspect_ratios: list[float] = []

        for idx, img_path in enumerate(image_paths):
            child = ContentValidator.validate_image(img_path)
            if not child.valid:
                for err in child.errors:
                    result.add_error(f"Image {idx + 1}: {err}")
            for warn in child.warnings:
                result.add_warning(f"Image {idx + 1}: {warn}")

            # Collect aspect ratio
            p = Path(img_path)
            if p.exists():
                try:
                    img = Image.open(p)
                    w, h = img.size
                    if h > 0:
                        aspect_ratios.append(w / h)
                    img.close()
                except Exception:
                    pass

        # Check consistent aspect ratio
        if len(aspect_ratios) >= 2:
            base = aspect_ratios[0]
            for idx, ar in enumerate(aspect_ratios[1:], start=2):
                if abs(ar - base) > CAROUSEL_ASPECT_TOLERANCE:
                    result.add_error(
                        f"Image {idx} aspect ratio ({ar:.3f}) differs from "
                        f"image 1 ({base:.3f}). Carousel images must share "
                        "the same aspect ratio."
                    )

        if not result.valid:
            # Propagate child errors
            pass

        log.info(
            "carousel_validation_complete",
            image_count=count,
            valid=result.valid,
            error_count=len(result.errors),
        )
        return result

    # -- Caption validation -------------------------------------------------

    @staticmethod
    def validate_caption(caption: str) -> ValidationResult:
        """Validate an Instagram caption.

        Checks performed:
        - Length <= 2200 characters
        - Hashtag count <= 30
        - No banned engagement-bait patterns
        """
        result = ValidationResult()

        if len(caption) > MAX_CAPTION_LENGTH:
            result.add_error(
                f"Caption length {len(caption)} exceeds maximum {MAX_CAPTION_LENGTH} characters"
            )

        hashtags = re.findall(r"#\w+", caption)
        if len(hashtags) > MAX_HASHTAG_COUNT:
            result.add_error(
                f"Caption has {len(hashtags)} hashtags, exceeding maximum of {MAX_HASHTAG_COUNT}"
            )
        elif len(hashtags) > 20:
            result.add_warning(
                f"Caption has {len(hashtags)} hashtags. Fewer than 20 is recommended "
                "for better reach."
            )

        for pattern in BANNED_CAPTION_PATTERNS:
            if pattern.search(caption):
                result.add_error(
                    f"Caption contains banned engagement-bait pattern: "
                    f"'{pattern.pattern}'"
                )

        log.info(
            "caption_validation_complete",
            valid=result.valid,
            length=len(caption),
            hashtag_count=len(hashtags),
        )
        return result
