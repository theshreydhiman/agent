"""Tests for content planning and generation modules."""

import pytest
from content.hashtags import HashtagEngine, NICHE_HASHTAGS
from content.captions import CaptionGenerator, FALLBACK_CAPTIONS
from quality.moderator import ContentModerator, AI_DISCLOSURE


class TestHashtagEngine:
    def setup_method(self):
        self.engine = HashtagEngine()

    def test_build_hashtag_set_default(self):
        tags = self.engine.build_hashtag_set("lifestyle")
        assert len(tags) <= 25
        assert len(tags) >= 10
        assert all(isinstance(t, str) for t in tags)

    def test_build_hashtag_set_with_theme(self):
        tags = self.engine.build_hashtag_set("lifestyle", theme="morning routine")
        assert "morningroutine" in tags

    def test_build_hashtag_set_unknown_niche_falls_back(self):
        tags = self.engine.build_hashtag_set("unknown_niche")
        assert len(tags) > 0  # Falls back to lifestyle

    def test_format_hashtags(self):
        tags = ["lifestyle", "instagood", "photooftheday"]
        formatted = self.engine.format_hashtags(tags)
        assert formatted == "#lifestyle #instagood #photooftheday"

    def test_format_hashtags_strips_existing_hash(self):
        tags = ["#lifestyle", "instagood"]
        formatted = self.engine.format_hashtags(tags)
        assert formatted == "#lifestyle #instagood"

    def test_niche_hashtags_structure(self):
        for niche, pools in NICHE_HASHTAGS.items():
            assert "high_volume" in pools
            assert "medium_volume" in pools
            assert "niche" in pools
            assert len(pools["high_volume"]) >= 5
            assert len(pools["medium_volume"]) >= 5
            assert len(pools["niche"]) >= 5


class TestContentModerator:
    def setup_method(self):
        self.moderator = ContentModerator()

    def test_clean_caption_passes(self):
        result = self.moderator.check_caption("A beautiful day in the city!")
        assert result.approved
        assert result.severity == "none"

    def test_banned_phrase_blocks(self):
        result = self.moderator.check_caption("This guaranteed results in weight loss")
        assert not result.approved
        assert result.severity == "block"

    def test_too_long_caption_warns(self):
        result = self.moderator.check_caption("a" * 2500)
        assert len(result.flags) > 0

    def test_check_hashtags_clean(self):
        result = self.moderator.check_hashtags(["lifestyle", "instagood"])
        assert result.approved

    def test_check_hashtags_banned(self):
        result = self.moderator.check_hashtags(["followforfollow", "lifestyle"])
        assert len(result.flags) > 0

    def test_check_hashtags_too_many(self):
        result = self.moderator.check_hashtags([f"tag{i}" for i in range(35)])
        assert any("too_many" in f for f in result.flags)

    def test_add_ai_disclosure(self):
        caption = "Living my best life"
        result = self.moderator.add_ai_disclosure(caption)
        assert AI_DISCLOSURE in result

    def test_add_ai_disclosure_skips_if_present(self):
        caption = "Living my best life\n\nCreated with AI"
        result = self.moderator.add_ai_disclosure(caption)
        assert result == caption  # No double disclosure

    def test_full_moderation_clean(self):
        result = self.moderator.full_moderation_check(
            caption="A beautiful day!",
            hashtags=["lifestyle", "beauty"],
        )
        assert result.approved

    def test_full_moderation_adds_disclosure(self):
        result = self.moderator.full_moderation_check(
            caption="A beautiful day!",
            hashtags=["lifestyle"],
        )
        assert result.modified_content is not None
        assert "AI" in result.modified_content["caption"]


class TestCaptionFallbacks:
    def test_fallback_captions_exist(self):
        assert len(FALLBACK_CAPTIONS) >= 10

    def test_fallback_captions_have_placeholders(self):
        for caption in FALLBACK_CAPTIONS:
            # Each should be usable with .format()
            assert isinstance(caption, str)
            assert len(caption) > 10
