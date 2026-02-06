"""Instagram caption and alt-text generation backed by a local LLM."""

from __future__ import annotations

import random

import structlog

from content.llm_client import LLMClient

log = structlog.get_logger(__name__)

# ── fallback templates (used when the LLM is unavailable) ────────────────────

FALLBACK_CAPTIONS: list[str] = [
    "Living my best {theme} life. {name} approved.",
    "Current mood: {theme}. Who else is feeling this?",
    "A little bit of {theme} goes a long way.",
    "{theme} state of mind. No filter needed.",
    "Channeling all the {theme} energy today.",
    "Let's talk about {theme}. Drop your thoughts below!",
    "This is what {theme} looks like through my eyes.",
    "If {theme} was a vibe, this would be it.",
    "POV: You just discovered {theme} and you're never going back.",
    "{name} here, bringing you a fresh take on {theme}.",
    "New day, new {theme} moment. Let's go!",
    "They told me to be ordinary. I chose {theme} instead.",
    "No caption needed... but here's one anyway. {theme} forever.",
    "Swipe for the full {theme} story.",
    "Obsessed with this {theme} energy right now.",
    "Reminder: {theme} is always a good idea.",
    "Another day, another {theme} adventure with {name}.",
    "Stepping into {theme} mode. Are you with me?",
    "Caught in a {theme} moment and I'm not mad about it.",
    "Your daily dose of {theme}, served fresh by {name}.",
    "{theme} hits different when you're living it.",
    "Tell me you love {theme} without telling me.",
]

CTAS: list[str] = [
    "Double-tap if you agree!",
    "Tag someone who needs to see this!",
    "Save this for later!",
    "Drop a comment and let me know your thoughts!",
    "Share this with your bestie!",
    "What do you think? Tell me below!",
    "Follow for more content like this!",
    "Hit that bookmark button!",
    "Which one is your fave? Comment below!",
    "Turn on post notifications so you never miss out!",
    "Send this to someone who needs it today!",
    "Agree? Disagree? Let's discuss!",
]

_CAPTION_SYSTEM = """\
You are {character_name}, a virtual influencer known for being {brand_voice}.
Write Instagram captions in first person as {character_name}.
Keep the tone consistent with the brand voice.
Do NOT include hashtags — those are added separately.
Return ONLY the caption text, nothing else.
"""

_CAPTION_USER = """\
Write an Instagram {content_type} caption about "{theme}".
Mood: {mood}
Maximum length: {max_length} characters.
"""

_ALT_TEXT_SYSTEM = (
    "You write concise, descriptive alt-text for Instagram images. "
    "Keep it under 150 characters. Return ONLY the alt text."
)


class CaptionGenerator:
    """Generates Instagram captions, alt-text, and calls-to-action."""

    def __init__(self) -> None:
        self.llm = LLMClient()

    # -- public API -----------------------------------------------------------

    async def generate_caption(
        self,
        theme: str,
        mood: str,
        character_name: str,
        brand_voice: str,
        content_type: str,
        max_length: int = 300,
    ) -> str:
        """Return an Instagram caption for the given parameters."""

        system = _CAPTION_SYSTEM.format(
            character_name=character_name,
            brand_voice=brand_voice,
        )

        user_msg = _CAPTION_USER.format(
            content_type=content_type,
            theme=theme,
            mood=mood,
            max_length=max_length,
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]

        result = await self.llm.chat(messages, temperature=0.8)

        if self._is_fallback(result):
            log.info("caption_fallback", theme=theme)
            return self._fallback_caption(theme, character_name)

        # Trim to requested length.
        caption = result.strip().strip('"')
        if len(caption) > max_length:
            caption = caption[: max_length - 1].rsplit(" ", 1)[0] + "\u2026"

        log.info("caption_generated", source="llm", length=len(caption))
        return caption

    async def generate_alt_text(self, image_description: str) -> str:
        """Return accessibility alt-text for an image."""

        messages = [
            {"role": "system", "content": _ALT_TEXT_SYSTEM},
            {"role": "user", "content": f"Image: {image_description}"},
        ]

        result = await self.llm.chat(messages, temperature=0.3)

        if self._is_fallback(result):
            # Simple deterministic fallback.
            return image_description[:125]

        return result.strip()[:150]

    def generate_call_to_action(self) -> str:
        """Return a random engaging CTA string."""
        return random.choice(CTAS)  # noqa: S311

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _is_fallback(text: str) -> bool:
        return text.startswith("[LLM unavailable")

    @staticmethod
    def _fallback_caption(theme: str, name: str) -> str:
        template = random.choice(FALLBACK_CAPTIONS)  # noqa: S311
        return template.format(theme=theme, name=name)
