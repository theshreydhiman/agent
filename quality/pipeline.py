"""Automated quality-assurance pipeline for generated images and videos."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import structlog
from PIL import Image

from character.face_analyzer import FaceAnalyzer

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
MIN_WIDTH_PX = 1080
MAX_IMAGE_SIZE_BYTES = 8 * 1024 * 1024          # 8 MB
MAX_VIDEO_SIZE_BYTES = 100 * 1024 * 1024         # 100 MB
MIN_VIDEO_DURATION_SEC = 3.0
MAX_VIDEO_DURATION_SEC = 90.0
MIN_VIDEO_FPS = 24.0
VIDEO_SAMPLE_FRAMES = 5
FACE_SIMILARITY_THRESHOLD = 0.80
ARTIFACT_BLUR_THRESHOLD = 50.0                   # Laplacian variance floor
ARTIFACT_SOLID_BLOCK_RATIO = 0.25                # fraction of image as one colour

# Accepted Instagram aspect ratios with a small tolerance.
INSTAGRAM_RATIOS: list[tuple[str, float]] = [
    ("1:1", 1.0),
    ("4:5", 4.0 / 5.0),
    ("16:9", 16.0 / 9.0),
    ("9:16", 9.0 / 16.0),
    ("1.91:1", 1.91),
]
ASPECT_RATIO_TOLERANCE = 0.05


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    """Result of a single QA check."""

    name: str
    passed: bool
    value: Any
    threshold: Any
    message: str


@dataclass
class QAResult:
    """Aggregate result of all QA checks."""

    passed: bool
    score: float                                   # 0.0 -- 1.0
    face_similarity_score: float | None = None
    checks: dict[str, CheckResult] = field(default_factory=dict)
    rejection_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class QAPipeline:
    """Run automated quality checks on generated images and videos."""

    def __init__(self, face_analyzer: FaceAnalyzer) -> None:
        self._face_analyzer = face_analyzer
        log.info("qa_pipeline.initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_image(
        self,
        image_path: str,
        reference_embeddings: list[np.ndarray] | None = None,
    ) -> QAResult:
        """Run every image-level quality check and return a *QAResult*."""
        checks: dict[str, CheckResult] = {}
        rejection_reasons: list[str] = []
        warnings: list[str] = []
        face_similarity_score: float | None = None

        path = Path(image_path)

        # --- 1. Resolution ---
        try:
            img = Image.open(path)
            width, height = img.size
        except Exception as exc:
            log.error("qa.image_open_failed", path=image_path, error=str(exc))
            cr = CheckResult(
                name="resolution",
                passed=False,
                value=None,
                threshold=MIN_WIDTH_PX,
                message=f"Cannot open image: {exc}",
            )
            checks["resolution"] = cr
            rejection_reasons.append(cr.message)
            return QAResult(
                passed=False,
                score=0.0,
                checks=checks,
                rejection_reasons=rejection_reasons,
                warnings=warnings,
            )

        cr_res = self._check_resolution(width, height)
        checks["resolution"] = cr_res
        if not cr_res.passed:
            rejection_reasons.append(cr_res.message)

        # --- 2. File size ---
        file_size = os.path.getsize(path)
        cr_size = CheckResult(
            name="file_size",
            passed=file_size <= MAX_IMAGE_SIZE_BYTES,
            value=file_size,
            threshold=MAX_IMAGE_SIZE_BYTES,
            message=(
                f"File size {file_size / (1024 * 1024):.2f}MB is within the "
                f"{MAX_IMAGE_SIZE_BYTES / (1024 * 1024):.0f}MB limit"
                if file_size <= MAX_IMAGE_SIZE_BYTES
                else f"File size {file_size / (1024 * 1024):.2f}MB exceeds "
                f"{MAX_IMAGE_SIZE_BYTES / (1024 * 1024):.0f}MB limit"
            ),
        )
        checks["file_size"] = cr_size
        if not cr_size.passed:
            rejection_reasons.append(cr_size.message)

        # --- 3. Aspect ratio ---
        cr_ar = self._check_aspect_ratio(width, height)
        checks["aspect_ratio"] = cr_ar
        if not cr_ar.passed:
            warnings.append(cr_ar.message)

        # --- 4. Face detection ---
        bgr = cv2.imread(str(path))
        cr_face, embedding = self._check_single_face(bgr, image_path)
        checks["face_detection"] = cr_face
        if not cr_face.passed:
            rejection_reasons.append(cr_face.message)

        # --- 5. Face consistency ---
        if embedding is not None and reference_embeddings:
            cr_fc, sim_score = self._check_face_consistency(
                embedding, reference_embeddings
            )
            checks["face_consistency"] = cr_fc
            face_similarity_score = sim_score
            if not cr_fc.passed:
                rejection_reasons.append(cr_fc.message)
        elif reference_embeddings:
            checks["face_consistency"] = CheckResult(
                name="face_consistency",
                passed=False,
                value=None,
                threshold=FACE_SIMILARITY_THRESHOLD,
                message="No face embedding extracted; cannot compare consistency",
            )
            rejection_reasons.append(checks["face_consistency"].message)

        # --- 6. Artifact detection ---
        if bgr is not None:
            cr_art = self._check_artifacts(bgr)
            checks["artifact_detection"] = cr_art
            if not cr_art.passed:
                warnings.append(cr_art.message)

        # --- aggregate ---
        passed = len(rejection_reasons) == 0
        score = self._compute_score(checks)

        result = QAResult(
            passed=passed,
            score=score,
            face_similarity_score=face_similarity_score,
            checks=checks,
            rejection_reasons=rejection_reasons,
            warnings=warnings,
        )
        log.info(
            "qa.image_check_complete",
            path=image_path,
            passed=passed,
            score=round(score, 3),
        )
        return result

    def check_video(
        self,
        video_path: str,
        reference_embeddings: list[np.ndarray] | None = None,
    ) -> QAResult:
        """Run every video-level quality check and return a *QAResult*."""
        checks: dict[str, CheckResult] = {}
        rejection_reasons: list[str] = []
        warnings: list[str] = []
        face_similarity_score: float | None = None

        path = Path(video_path)

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            log.error("qa.video_open_failed", path=video_path)
            cr = CheckResult(
                name="resolution",
                passed=False,
                value=None,
                threshold=MIN_WIDTH_PX,
                message=f"Cannot open video: {video_path}",
            )
            checks["resolution"] = cr
            rejection_reasons.append(cr.message)
            return QAResult(
                passed=False,
                score=0.0,
                checks=checks,
                rejection_reasons=rejection_reasons,
                warnings=warnings,
            )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0

        # --- 1. Resolution ---
        cr_res = self._check_resolution(width, height)
        checks["resolution"] = cr_res
        if not cr_res.passed:
            rejection_reasons.append(cr_res.message)

        # --- 2. Duration ---
        dur_ok = MIN_VIDEO_DURATION_SEC <= duration <= MAX_VIDEO_DURATION_SEC
        cr_dur = CheckResult(
            name="duration",
            passed=dur_ok,
            value=round(duration, 2),
            threshold=f"{MIN_VIDEO_DURATION_SEC}-{MAX_VIDEO_DURATION_SEC}s",
            message=(
                f"Duration {duration:.1f}s is within limits"
                if dur_ok
                else f"Duration {duration:.1f}s is outside the "
                f"{MIN_VIDEO_DURATION_SEC}-{MAX_VIDEO_DURATION_SEC}s range"
            ),
        )
        checks["duration"] = cr_dur
        if not cr_dur.passed:
            rejection_reasons.append(cr_dur.message)

        # --- 3. FPS ---
        fps_ok = fps >= MIN_VIDEO_FPS
        cr_fps = CheckResult(
            name="fps",
            passed=fps_ok,
            value=round(fps, 2),
            threshold=MIN_VIDEO_FPS,
            message=(
                f"FPS {fps:.1f} meets minimum {MIN_VIDEO_FPS}"
                if fps_ok
                else f"FPS {fps:.1f} is below minimum {MIN_VIDEO_FPS}"
            ),
        )
        checks["fps"] = cr_fps
        if not cr_fps.passed:
            rejection_reasons.append(cr_fps.message)

        # --- 4. File size ---
        file_size = os.path.getsize(path)
        size_ok = file_size <= MAX_VIDEO_SIZE_BYTES
        cr_size = CheckResult(
            name="file_size",
            passed=size_ok,
            value=file_size,
            threshold=MAX_VIDEO_SIZE_BYTES,
            message=(
                f"File size {file_size / (1024 * 1024):.2f}MB is within the "
                f"{MAX_VIDEO_SIZE_BYTES / (1024 * 1024):.0f}MB limit"
                if size_ok
                else f"File size {file_size / (1024 * 1024):.2f}MB exceeds "
                f"{MAX_VIDEO_SIZE_BYTES / (1024 * 1024):.0f}MB limit"
            ),
        )
        checks["file_size"] = cr_size
        if not cr_size.passed:
            rejection_reasons.append(cr_size.message)

        # --- 5. Sample-frame face consistency ---
        if frame_count > 0:
            frames = self._extract_sample_frames(cap, frame_count)
            face_results = self._check_video_face_consistency(
                frames, reference_embeddings
            )
            checks["frame_face_consistency"] = face_results["check"]
            face_similarity_score = face_results.get("similarity_score")
            if not face_results["check"].passed:
                rejection_reasons.append(face_results["check"].message)

        cap.release()

        passed = len(rejection_reasons) == 0
        score = self._compute_score(checks)

        result = QAResult(
            passed=passed,
            score=score,
            face_similarity_score=face_similarity_score,
            checks=checks,
            rejection_reasons=rejection_reasons,
            warnings=warnings,
        )
        log.info(
            "qa.video_check_complete",
            path=video_path,
            passed=passed,
            score=round(score, 3),
        )
        return result

    def run_full_check(
        self,
        content_path: str,
        content_type: str,
        reference_embeddings: list[np.ndarray] | None = None,
    ) -> QAResult:
        """Dispatch to the correct checker based on *content_type*.

        ``content_type`` should be ``"image"`` or ``"video"``.
        """
        ct = content_type.lower().strip()
        if ct == "image":
            return self.check_image(content_path, reference_embeddings)
        if ct == "video":
            return self.check_video(content_path, reference_embeddings)

        log.error("qa.unknown_content_type", content_type=content_type)
        return QAResult(
            passed=False,
            score=0.0,
            rejection_reasons=[f"Unknown content type: {content_type}"],
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_resolution(width: int, height: int) -> CheckResult:
        ok = width >= MIN_WIDTH_PX
        return CheckResult(
            name="resolution",
            passed=ok,
            value=f"{width}x{height}",
            threshold=f"{MIN_WIDTH_PX}px width",
            message=(
                f"Resolution {width}x{height} meets minimum width {MIN_WIDTH_PX}px"
                if ok
                else f"Resolution {width}x{height} is below minimum width "
                f"{MIN_WIDTH_PX}px"
            ),
        )

    @staticmethod
    def _check_aspect_ratio(width: int, height: int) -> CheckResult:
        ratio = width / height if height > 0 else 0.0
        for label, target in INSTAGRAM_RATIOS:
            if abs(ratio - target) <= ASPECT_RATIO_TOLERANCE:
                return CheckResult(
                    name="aspect_ratio",
                    passed=True,
                    value=round(ratio, 3),
                    threshold=[label for label, _ in INSTAGRAM_RATIOS],
                    message=f"Aspect ratio {ratio:.3f} matches Instagram {label}",
                )
        return CheckResult(
            name="aspect_ratio",
            passed=False,
            value=round(ratio, 3),
            threshold=[label for label, _ in INSTAGRAM_RATIOS],
            message=f"Aspect ratio {ratio:.3f} does not match any standard "
            "Instagram ratio",
        )

    def _check_single_face(
        self, bgr_image: np.ndarray | None, image_path: str
    ) -> tuple[CheckResult, np.ndarray | None]:
        """Ensure exactly one face is detected. Return the check and embedding."""
        if bgr_image is None:
            return (
                CheckResult(
                    name="face_detection",
                    passed=False,
                    value=0,
                    threshold=1,
                    message=f"Could not read image at {image_path}",
                ),
                None,
            )

        # Use the underlying InsightFace model if available.
        if self._face_analyzer._app is None:
            return (
                CheckResult(
                    name="face_detection",
                    passed=False,
                    value=None,
                    threshold=1,
                    message="Face analyzer model not available",
                ),
                None,
            )

        faces = self._face_analyzer._app.get(bgr_image)
        num_faces = len(faces) if faces else 0

        if num_faces == 1:
            embedding = faces[0].embedding
            return (
                CheckResult(
                    name="face_detection",
                    passed=True,
                    value=num_faces,
                    threshold=1,
                    message="Exactly one face detected",
                ),
                embedding,
            )

        return (
            CheckResult(
                name="face_detection",
                passed=False,
                value=num_faces,
                threshold=1,
                message=f"Expected 1 face, detected {num_faces}",
            ),
            None,
        )

    def _check_face_consistency(
        self,
        embedding: np.ndarray,
        reference_embeddings: list[np.ndarray],
    ) -> tuple[CheckResult, float]:
        result = self._face_analyzer.check_consistency(
            embedding, reference_embeddings
        )
        score = result["mean_score"]
        ok = result["passes_threshold"]
        cr = CheckResult(
            name="face_consistency",
            passed=ok,
            value=round(score, 4),
            threshold=FACE_SIMILARITY_THRESHOLD,
            message=(
                f"Face similarity {score:.3f} meets threshold "
                f"{FACE_SIMILARITY_THRESHOLD}"
                if ok
                else f"Face similarity {score:.3f} is below threshold "
                f"{FACE_SIMILARITY_THRESHOLD}"
            ),
        )
        return cr, score

    @staticmethod
    def _check_artifacts(bgr_image: np.ndarray) -> CheckResult:
        """Basic heuristic check for common generation artifacts."""
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        issues: list[str] = []

        # Extreme blur detection via Laplacian variance.
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < ARTIFACT_BLUR_THRESHOLD:
            issues.append(
                f"Extreme blur detected (laplacian variance {laplacian_var:.1f} "
                f"< {ARTIFACT_BLUR_THRESHOLD})"
            )

        # Solid colour block detection: check if a large portion of pixels
        # share the exact same value (quantised to reduce noise).
        quantised = (gray // 16) * 16
        unique, counts = np.unique(quantised, return_counts=True)
        total_pixels = gray.shape[0] * gray.shape[1]
        max_ratio = float(counts.max()) / total_pixels
        if max_ratio > ARTIFACT_SOLID_BLOCK_RATIO:
            dominant_val = int(unique[counts.argmax()])
            issues.append(
                f"Large solid colour block detected ({max_ratio:.1%} of pixels "
                f"at value ~{dominant_val})"
            )

        if issues:
            return CheckResult(
                name="artifact_detection",
                passed=False,
                value=issues,
                threshold="no artifacts",
                message="Artifacts detected: " + "; ".join(issues),
            )
        return CheckResult(
            name="artifact_detection",
            passed=True,
            value=[],
            threshold="no artifacts",
            message="No obvious generation artifacts detected",
        )

    @staticmethod
    def _extract_sample_frames(
        cap: cv2.VideoCapture, frame_count: int
    ) -> list[np.ndarray]:
        """Extract *VIDEO_SAMPLE_FRAMES* evenly-spaced frames from the video."""
        indices = np.linspace(0, frame_count - 1, VIDEO_SAMPLE_FRAMES, dtype=int)
        frames: list[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        return frames

    def _check_video_face_consistency(
        self,
        frames: list[np.ndarray],
        reference_embeddings: list[np.ndarray] | None,
    ) -> dict:
        """Check face detection and consistency across sampled video frames."""
        embeddings: list[np.ndarray] = []

        for frame in frames:
            emb = self._face_analyzer.extract_embedding_from_array(frame)
            if emb is not None:
                embeddings.append(emb)

        detected_ratio = len(embeddings) / len(frames) if frames else 0.0

        if detected_ratio < 0.6:
            return {
                "check": CheckResult(
                    name="frame_face_consistency",
                    passed=False,
                    value=f"{len(embeddings)}/{len(frames)} frames with faces",
                    threshold=">=60% frames with a face",
                    message=f"Face detected in only {len(embeddings)} of "
                    f"{len(frames)} sampled frames",
                ),
                "similarity_score": None,
            }

        # Cross-frame consistency: every consecutive pair should be similar.
        if len(embeddings) >= 2:
            pair_scores = [
                FaceAnalyzer.compare_faces(embeddings[i], embeddings[i + 1])
                for i in range(len(embeddings) - 1)
            ]
            min_pair = min(pair_scores)
            mean_pair = float(np.mean(pair_scores))
        else:
            min_pair = 1.0
            mean_pair = 1.0

        # Reference comparison (if provided).
        ref_score: float | None = None
        ref_ok = True
        if reference_embeddings and embeddings:
            ref_scores = []
            for emb in embeddings:
                result = self._face_analyzer.check_consistency(
                    emb, reference_embeddings
                )
                ref_scores.append(result["mean_score"])
            ref_score = float(np.mean(ref_scores))
            ref_ok = ref_score >= FACE_SIMILARITY_THRESHOLD

        internal_ok = min_pair >= FACE_SIMILARITY_THRESHOLD
        overall_ok = internal_ok and ref_ok

        parts: list[str] = [
            f"cross-frame min={min_pair:.3f} mean={mean_pair:.3f}",
        ]
        if ref_score is not None:
            parts.append(f"reference mean={ref_score:.3f}")

        return {
            "check": CheckResult(
                name="frame_face_consistency",
                passed=overall_ok,
                value="; ".join(parts),
                threshold=FACE_SIMILARITY_THRESHOLD,
                message=(
                    "Video face consistency is acceptable"
                    if overall_ok
                    else "Video face consistency is below threshold: "
                    + "; ".join(parts)
                ),
            ),
            "similarity_score": ref_score if ref_score is not None else mean_pair,
        }

    @staticmethod
    def _compute_score(checks: dict[str, CheckResult]) -> float:
        """Compute a 0.0 -- 1.0 aggregate score from individual checks."""
        if not checks:
            return 0.0
        return sum(1.0 for c in checks.values() if c.passed) / len(checks)
