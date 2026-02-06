"""Content moderation and safety checks."""

from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()

BANNED_PHRASES = [
    "guaranteed results", "lose weight fast", "get rich quick",
    "miracle cure", "100% effective", "no side effects",
    "financial freedom guaranteed", "secret method",
    "doctors hate this", "one weird trick",
]

BANNED_HASHTAGS = [
    "followforfollow", "f4f", "likeforlike", "l4l",
    "followback", "instagain", "instalike",
    "spam", "adult", "nsfw",
]

AI_DISCLOSURE = "\n\n✨ Created with AI"


@dataclass
class ModerationResult:
    approved: bool
    flags: list[str] = field(default_factory=list)
    severity: str = "none"  # none, warning, block
    modified_content: dict | None = None


class ContentModerator:
    """Checks content against safety and policy rules."""

    def check_caption(self, caption: str) -> ModerationResult:
        """Check caption text for policy violations."""
        flags = []
        caption_lower = caption.lower()

        # Check banned phrases
        for phrase in BANNED_PHRASES:
            if phrase in caption_lower:
                flags.append(f"banned_phrase: {phrase}")

        # Check length
        if len(caption) > 2200:
            flags.append(f"caption_too_long: {len(caption)} chars (max 2200)")

        if len(caption) < 5:
            flags.append("caption_too_short")

        severity = "none"
        if flags:
            severity = "block" if any("banned_phrase" in f for f in flags) else "warning"

        return ModerationResult(
            approved=severity != "block", flags=flags, severity=severity,
        )

    def check_hashtags(self, hashtags: list[str]) -> ModerationResult:
        """Validate hashtags against policy rules."""
        flags = []
        cleaned = [h.lstrip("#").lower() for h in hashtags]

        # Check banned hashtags
        for tag in cleaned:
            if tag in BANNED_HASHTAGS:
                flags.append(f"banned_hashtag: #{tag}")

        # Check count
        if len(hashtags) > 30:
            flags.append(f"too_many_hashtags: {len(hashtags)} (max 30)")

        # Check for spam patterns
        for tag in hashtags:
            if tag.upper() == tag and len(tag) > 3:
                flags.append(f"all_caps_hashtag: {tag}")

        severity = "none"
        if flags:
            severity = "warning" if len(flags) <= 2 else "block"

        return ModerationResult(
            approved=severity != "block", flags=flags, severity=severity,
        )

    def add_ai_disclosure(self, caption: str) -> str:
        """Add AI-generated content disclosure if not already present."""
        disclosure_markers = ["created with ai", "ai-generated", "ai generated", "made with ai"]
        caption_lower = caption.lower()

        for marker in disclosure_markers:
            if marker in caption_lower:
                return caption

        return caption + AI_DISCLOSURE

    def full_moderation_check(
        self,
        caption: str,
        hashtags: list[str],
        image_path: str | None = None,
    ) -> ModerationResult:
        """Run all moderation checks on content."""
        all_flags = []
        worst_severity = "none"

        caption_result = self.check_caption(caption)
        all_flags.extend(caption_result.flags)

        hashtag_result = self.check_hashtags(hashtags)
        all_flags.extend(hashtag_result.flags)

        # Determine worst severity
        for result in [caption_result, hashtag_result]:
            if result.severity == "block":
                worst_severity = "block"
            elif result.severity == "warning" and worst_severity != "block":
                worst_severity = "warning"

        # Auto-add AI disclosure
        modified_caption = self.add_ai_disclosure(caption)
        modified = None
        if modified_caption != caption:
            modified = {"caption": modified_caption}

        approved = worst_severity != "block"

        if not approved:
            logger.warning("content_blocked", flags=all_flags)
        elif all_flags:
            logger.info("content_warnings", flags=all_flags)

        return ModerationResult(
            approved=approved, flags=all_flags,
            severity=worst_severity, modified_content=modified,
        )
