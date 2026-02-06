"""Weekly content-calendar planner backed by a local LLM."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Sequence

import structlog

from content.llm_client import LLMClient

log = structlog.get_logger(__name__)

# ── data model ───────────────────────────────────────────────────────────────

CONTENT_TYPES = ("post", "story", "reel", "carousel")

FALLBACK_THEMES = [
    "morning routine",
    "outfit of the day",
    "motivational quote",
    "behind the scenes",
    "product spotlight",
    "travel moment",
    "fitness check-in",
]

FALLBACK_MOODS = [
    "confident",
    "playful",
    "serene",
    "bold",
    "warm",
    "edgy",
    "dreamy",
]

FALLBACK_SCENES = [
    "urban rooftop",
    "cozy bedroom",
    "coffee shop",
    "studio backdrop",
    "park bench",
    "beach sunset",
    "city street",
]

FALLBACK_OUTFITS = [
    "casual streetwear",
    "athleisure set",
    "elegant evening",
    "minimalist chic",
    "boho layers",
    "sporty crop top",
    "cozy knitwear",
]


@dataclass
class ContentPlanEntry:
    """One slot in the weekly content calendar."""

    date: str
    time_slot: str
    content_type: str
    theme: str
    scene_preset: str
    outfit_preset: str
    mood: str
    caption_prompt: str


# ── optimal posting times (industry data) ────────────────────────────────────

OPTIMAL_TIMES: dict[str, list[str]] = {
    "Monday":    ["11:00", "14:00", "19:00"],
    "Tuesday":   ["09:00", "13:00", "18:00"],
    "Wednesday": ["11:00", "15:00", "19:00"],
    "Thursday":  ["10:00", "14:00", "20:00"],
    "Friday":    ["09:00", "13:00", "17:00"],
    "Saturday":  ["10:00", "12:00", "18:00"],
    "Sunday":    ["10:00", "14:00", "19:00"],
}

_SYSTEM_PROMPT = (
    "You are a social-media strategist for a virtual influencer. "
    "Reply ONLY with valid JSON — no markdown fences, no commentary."
)

_PLAN_PROMPT_TEMPLATE = """\
Create a 7-day Instagram content plan for an AI virtual influencer.

Character: {character_name}
Niche: {niche}
Brand voice: {brand_voice}

{existing_context}

Return a JSON array of exactly 7 objects, one per day starting from {start_date}.
Each object MUST have these keys:
- "date"          (YYYY-MM-DD)
- "time_slot"     (HH:MM, 24-hour)
- "content_type"  (one of: post, story, reel, carousel)
- "theme"         (short phrase, e.g. "morning coffee ritual")
- "scene_preset"  (visual setting, e.g. "sunny kitchen")
- "outfit_preset" (clothing description, e.g. "casual linen set")
- "mood"          (single word, e.g. "warm")
- "caption_prompt"(brief direction for caption generation)

Ensure variety across content types, themes, and moods.
"""


# ── planner ──────────────────────────────────────────────────────────────────

class ContentPlanner:
    """Generates weekly content calendars using a local LLM with deterministic
    fallbacks when the LLM is unreachable."""

    def __init__(self) -> None:
        self.llm = LLMClient()

    # -- public API -----------------------------------------------------------

    async def generate_weekly_plan(
        self,
        character_name: str,
        niche: str,
        brand_voice: str,
        existing_posts: list[str] | None = None,
    ) -> list[ContentPlanEntry]:
        """Return seven :class:`ContentPlanEntry` items, one per day."""

        start = date.today()

        existing_context = ""
        if existing_posts:
            recent = ", ".join(existing_posts[:10])
            existing_context = f"Recent posts (avoid repeating): {recent}"

        prompt = _PLAN_PROMPT_TEMPLATE.format(
            character_name=character_name,
            niche=niche,
            brand_voice=brand_voice,
            existing_context=existing_context,
            start_date=start.isoformat(),
        )

        raw = await self.llm.generate(
            prompt=prompt,
            system_prompt=_SYSTEM_PROMPT,
            temperature=0.8,
            max_tokens=2048,
        )

        entries = self._parse_plan(raw, start)
        if entries:
            log.info("weekly_plan_generated", source="llm", count=len(entries))
            return entries

        log.info("weekly_plan_generated", source="fallback")
        return self._fallback_plan(start, character_name, niche)

    def get_optimal_posting_times(self, timezone: str) -> dict[str, list[str]]:
        """Return best posting windows per weekday.

        Times are hardcoded from industry engagement studies and expressed in
        the requested *timezone* label (the caller is responsible for any UTC
        conversion).
        """
        return {day: list(times) for day, times in OPTIMAL_TIMES.items()}

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _parse_plan(
        raw: str,
        start: date,
    ) -> list[ContentPlanEntry]:
        """Try to extract a JSON array from the LLM response."""

        # Strip markdown code fences if the model wrapped its reply.
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()

        # Find the outermost JSON array.
        match = re.search(r"\[.*]", cleaned, re.DOTALL)
        if not match:
            return []

        try:
            items = json.loads(match.group())
        except json.JSONDecodeError:
            log.warning("plan_json_parse_error")
            return []

        entries: list[ContentPlanEntry] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                entries.append(
                    ContentPlanEntry(
                        date=str(item.get("date", "")),
                        time_slot=str(item.get("time_slot", "12:00")),
                        content_type=str(item.get("content_type", "post")),
                        theme=str(item.get("theme", "")),
                        scene_preset=str(item.get("scene_preset", "")),
                        outfit_preset=str(item.get("outfit_preset", "")),
                        mood=str(item.get("mood", "")),
                        caption_prompt=str(item.get("caption_prompt", "")),
                    )
                )
            except Exception:  # noqa: BLE001
                continue

        return entries

    @staticmethod
    def _fallback_plan(
        start: date,
        character_name: str,
        niche: str,
    ) -> list[ContentPlanEntry]:
        """Deterministic plan that cycles through preset pools."""

        entries: list[ContentPlanEntry] = []
        for i in range(7):
            day = start + timedelta(days=i)
            day_name = day.strftime("%A")
            time_slot = OPTIMAL_TIMES.get(day_name, ["12:00"])[0]

            entries.append(
                ContentPlanEntry(
                    date=day.isoformat(),
                    time_slot=time_slot,
                    content_type=CONTENT_TYPES[i % len(CONTENT_TYPES)],
                    theme=FALLBACK_THEMES[i % len(FALLBACK_THEMES)],
                    scene_preset=FALLBACK_SCENES[i % len(FALLBACK_SCENES)],
                    outfit_preset=FALLBACK_OUTFITS[i % len(FALLBACK_OUTFITS)],
                    mood=FALLBACK_MOODS[i % len(FALLBACK_MOODS)],
                    caption_prompt=(
                        f"{character_name} shares a {FALLBACK_THEMES[i % len(FALLBACK_THEMES)]} "
                        f"moment in the {niche} space"
                    ),
                )
            )

        return entries
