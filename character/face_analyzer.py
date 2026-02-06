"""Face embedding extraction and comparison using InsightFace/ArcFace."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import structlog

from config import get_settings

log = structlog.get_logger(__name__)


class FaceAnalyzer:
    """Wraps InsightFace for face embedding extraction and similarity checks."""

    def __init__(self) -> None:
        self._app = None
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(
                name="buffalo_l", providers=["CPUExecutionProvider"]
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            self._app = app
            log.info("face_analyzer.initialized")
        except Exception as exc:
            log.warning(
                "face_analyzer.init_failed",
                error=str(exc),
                hint="InsightFace models may not be downloaded yet. "
                "Face analysis will be unavailable.",
            )

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def extract_embedding(self, image_path: str | Path) -> np.ndarray | None:
        """Extract a 512-dim face embedding from an image file.

        Returns ``None`` when no face is detected or the model is unavailable.
        """
        if self._app is None:
            log.warning("face_analyzer.no_model", action="extract_embedding")
            return None

        img = cv2.imread(str(image_path))
        if img is None:
            log.warning("face_analyzer.image_read_failed", path=str(image_path))
            return None

        return self._detect_and_embed(img)

    def extract_embedding_from_array(
        self, image_array: np.ndarray
    ) -> np.ndarray | None:
        """Extract a 512-dim face embedding from a BGR numpy array.

        Returns ``None`` when no face is detected or the model is unavailable.
        """
        if self._app is None:
            log.warning(
                "face_analyzer.no_model", action="extract_embedding_from_array"
            )
            return None

        return self._detect_and_embed(image_array)

    # ------------------------------------------------------------------
    # Comparison helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compare_faces(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Return cosine similarity (0.0 -- 1.0) between two embeddings."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        similarity = float(np.dot(embedding1, embedding2) / (norm1 * norm2))
        # Clamp to [0, 1] -- cosine similarity on normalised ArcFace
        # embeddings is typically in [0, 1] but floating-point arithmetic
        # can nudge it slightly outside.
        return max(0.0, min(1.0, similarity))

    def check_consistency(
        self,
        embedding: np.ndarray,
        reference_embeddings: list[np.ndarray],
    ) -> dict:
        """Compare *embedding* against every reference embedding.

        Returns a dict with ``mean_score``, ``min_score``, ``max_score``, and
        ``passes_threshold`` (compared to the hard-floor setting).
        """
        if not reference_embeddings:
            return {
                "mean_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "passes_threshold": False,
            }

        scores = [
            self.compare_faces(embedding, ref) for ref in reference_embeddings
        ]
        mean_score = float(np.mean(scores))
        min_score = float(min(scores))
        max_score = float(max(scores))
        threshold = get_settings().face_similarity_hard_floor

        return {
            "mean_score": mean_score,
            "min_score": min_score,
            "max_score": max_score,
            "passes_threshold": mean_score >= threshold,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _detect_and_embed(self, bgr_image: np.ndarray) -> np.ndarray | None:
        """Run detection on a BGR image and return the largest face embedding."""
        faces = self._app.get(bgr_image)
        if not faces:
            log.debug("face_analyzer.no_face_detected")
            return None

        # Pick the largest face by bounding-box area.
        largest = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        embedding: np.ndarray = largest.embedding  # 512-dim
        return embedding
