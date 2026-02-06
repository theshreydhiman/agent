"""Tests for analytics reporting module."""

import pytest
from analytics.reporter import AnalyticsReporter


class TestAnalyticsReporter:
    def setup_method(self):
        self.reporter = AnalyticsReporter()

    def test_engagement_rate_calculation(self):
        rate = self.reporter.calculate_engagement_rate(
            likes=100, comments=20, saves=10, reach=1000,
        )
        assert rate == pytest.approx(0.13)

    def test_engagement_rate_zero_reach(self):
        rate = self.reporter.calculate_engagement_rate(
            likes=100, comments=20, saves=10, reach=0,
        )
        assert rate == 0.0

    def test_performance_summary_empty(self):
        summary = self.reporter.generate_performance_summary([])
        assert summary.total_posts == 0
        assert summary.avg_likes == 0
        assert summary.best_post_id is None

    def test_performance_summary_basic(self):
        insights = [
            {"media_id": "post1", "likes": 100, "comments": 10, "saves": 5, "reach": 1000},
            {"media_id": "post2", "likes": 200, "comments": 20, "saves": 10, "reach": 2000},
        ]
        summary = self.reporter.generate_performance_summary(insights)
        assert summary.total_posts == 2
        assert summary.avg_likes == 150
        assert summary.avg_comments == 15

    def test_compare_periods_trending_up(self):
        current = [
            {"media_id": "c1", "likes": 200, "comments": 20, "saves": 10, "reach": 1000},
        ]
        previous = [
            {"media_id": "p1", "likes": 100, "comments": 10, "saves": 5, "reach": 1000},
        ]
        comparison = self.reporter.compare_periods(current, previous)
        assert comparison.trending_up
        assert comparison.engagement_change_pct > 0

    def test_compare_periods_trending_down(self):
        current = [
            {"media_id": "c1", "likes": 50, "comments": 5, "saves": 2, "reach": 1000},
        ]
        previous = [
            {"media_id": "p1", "likes": 200, "comments": 20, "saves": 10, "reach": 1000},
        ]
        comparison = self.reporter.compare_periods(current, previous)
        assert not comparison.trending_up

    def test_recommendations_low_engagement(self):
        insights = [
            {"media_id": "p1", "likes": 10, "comments": 1, "saves": 0, "reach": 1000},
        ]
        recs = self.reporter.get_posting_recommendations(insights)
        assert len(recs) > 0
        assert any("engagement" in r.lower() for r in recs)

    def test_recommendations_low_volume(self):
        insights = [
            {"media_id": "p1", "likes": 100, "comments": 10, "saves": 5, "reach": 100},
        ]
        recs = self.reporter.get_posting_recommendations(insights)
        assert any("volume" in r.lower() or "posting" in r.lower() for r in recs)
