"""Character identity management -- loads config, builds prompts, stores references."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import numpy as np
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from config import get_settings
from database.models import Character, ReferenceImage
from character.face_analyzer import FaceAnalyzer

log = structlog.get_logger(__name__)


class CharacterManager:
    """High-level facade for character identity operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._face = FaceAnalyzer()
        self._settings = get_settings()

    # ------------------------------------------------------------------
    # Character CRUD helpers
    # ------------------------------------------------------------------

    async def load_character(self, character_id: UUID) -> Character | None:
        """Load a character row (with reference images) from the database."""
        stmt = (
            select(Character)
            .options(selectinload(Character.reference_images))
            .where(Character.id == character_id)
        )
        result = await self._session.execute(stmt)
        character = result.scalar_one_or_none()
        if character is None:
            log.warning("character.not_found", character_id=str(character_id))
        return character

    # ------------------------------------------------------------------
    # Reference-image management
    # ------------------------------------------------------------------

    async def get_reference_embeddings(
        self, character_id: UUID
    ) -> list[np.ndarray]:
        """Return all stored face embeddings for a character."""
        stmt = (
            select(ReferenceImage)
            .where(
                ReferenceImage.character_id == character_id,
                ReferenceImage.embedding.isnot(None),
            )
        )
        result = await self._session.execute(stmt)
        rows = result.scalars().all()

        embeddings: list[np.ndarray] = []
        for row in rows:
            arr = np.frombuffer(row.embedding, dtype=np.float32).copy()
            embeddings.append(arr)
        return embeddings

    async def add_reference_image(
        self,
        character_id: UUID,
        image_path: str | Path,
        *,
        angle: str = "front",
        expression: str = "neutral",
        s3_key: str = "",
    ) -> ReferenceImage:
        """Process *image_path*, extract its face embedding, and store a new
        :class:`ReferenceImage` row.

        Returns the newly created row.
        """
        image_path = Path(image_path)

        # Extract embedding (may be None if model isn't loaded or no face).
        embedding_arr = self._face.extract_embedding(image_path)
        embedding_bytes: bytes | None = None
        if embedding_arr is not None:
            embedding_bytes = embedding_arr.astype(np.float32).tobytes()
        else:
            log.warning(
                "character.no_embedding_extracted",
                character_id=str(character_id),
                image_path=str(image_path),
            )

        ref = ReferenceImage(
            character_id=character_id,
            file_path=str(image_path),
            s3_key=s3_key,
            angle=angle,
            expression=expression,
            embedding=embedding_bytes,
        )
        self._session.add(ref)
        await self._session.flush()

        log.info(
            "character.reference_image_added",
            character_id=str(character_id),
            ref_id=str(ref.id),
            has_embedding=embedding_bytes is not None,
        )
        return ref

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    async def get_appearance_prompt(
        self,
        character_id: UUID,
        *,
        scene: str | None = None,
        outfit: str | None = None,
    ) -> str:
        """Build a Stable Diffusion prompt from the character's appearance config.

        Optional *scene* and *outfit* overrides are merged into the final
        prompt when provided.
        """
        character = await self.load_character(character_id)
        if character is None:
            return ""

        cfg = character.appearance_config or {}

        # Core appearance tokens
        parts: list[str] = []

        if cfg.get("base_prompt"):
            parts.append(cfg["base_prompt"])

        # Demographics / physical
        for key in ("gender", "age_range", "ethnicity", "body_type"):
            if cfg.get(key):
                parts.append(cfg[key])

        # Hair
        if cfg.get("hair"):
            parts.append(cfg["hair"])

        # Eyes
        if cfg.get("eyes"):
            parts.append(cfg["eyes"])

        # Outfit -- use override or config default
        effective_outfit = outfit or cfg.get("default_outfit")
        if effective_outfit:
            parts.append(effective_outfit)

        # Scene / background -- use override or config default
        effective_scene = scene or cfg.get("default_scene")
        if effective_scene:
            parts.append(effective_scene)

        # Style modifiers (e.g. "photorealistic, 8k, studio lighting")
        if cfg.get("style_modifiers"):
            parts.append(cfg["style_modifiers"])

        prompt = ", ".join(p.strip() for p in parts if p.strip())
        log.debug(
            "character.prompt_built",
            character_id=str(character_id),
            prompt_length=len(prompt),
        )
        return prompt
