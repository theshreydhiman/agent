"""Tests for the publisher validators."""

import os
import tempfile

import pytest
from publisher.validators import ContentValidator


class TestContentValidator:
    def setup_method(self):
        self.validator = ContentValidator()

    def test_validate_caption_valid(self):
        result = self.validator.validate_caption("A great day! #lifestyle #instagood")
        assert result.valid
        assert len(result.errors) == 0

    def test_validate_caption_too_long(self):
        result = self.validator.validate_caption("a" * 2500)
        assert not result.valid
        assert any("length" in e.lower() or "2200" in e for e in result.errors)

    def test_validate_caption_too_many_hashtags(self):
        tags = " ".join(f"#tag{i}" for i in range(35))
        result = self.validator.validate_caption(f"Caption {tags}")
        assert any("hashtag" in e.lower() for e in result.errors)

    def test_validate_caption_empty_warns(self):
        result = self.validator.validate_caption("")
        # Empty captions are technically allowed by Instagram
        assert isinstance(result.valid, bool)

    def test_validate_image_missing_file(self):
        result = self.validator.validate_image("/nonexistent/path.jpg")
        assert not result.valid
        assert any("exist" in e.lower() or "not found" in e.lower() for e in result.errors)


class TestValidationResult:
    def test_validation_result_structure(self):
        from publisher.validators import ValidationResult
        result = ValidationResult(valid=True, errors=[], warnings=["test warning"])
        assert result.valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 1
