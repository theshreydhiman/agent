"""Analytics reporting and performance summaries."""

from dataclasses import dataclass
from datetime import datetime

import structlog

logger = structlog.get_logger()


@dataclass
class PerformanceSummary:
    total_posts: int
    avg_likes: float
    avg_comments: float
    avg_saves: float
    avg_reach: float
    avg_engagement_rate: float
    best_post_id: str | None
    worst_post_id: str | None
    best_hour: int | None
    period_start: datetime | None
    period_end: datetime | None


@dataclass
class PeriodComparison:
    current_avg_engagement: float
    previous_avg_engagement: float
    engagement_change_pct: float
    reach_change_pct: float
    trending_up: bool
    summary_text: str


class AnalyticsReporter:
    """Generates analytics reports from collected engagement data."""

    def calculate_engagement_rate(
        self, likes: int, comments: int, saves: int, reach: int
    ) -> float:
        if reach == 0:
            return 0.0
        return (likes + comments + saves) / reach

    def generate_performance_summary(
        self, insights: list[dict]
    ) -> PerformanceSummary:
        if not insights:
            return PerformanceSummary(
                total_posts=0, avg_likes=0, avg_comments=0, avg_saves=0,
                avg_reach=0, avg_engagement_rate=0, best_post_id=None,
                worst_post_id=None, best_hour=None,
                period_start=None, period_end=None,
            )

        total = len(insights)
        avg_likes = sum(i.get("likes", 0) for i in insights) / total
        avg_comments = sum(i.get("comments", 0) for i in insights) / total
        avg_saves = sum(i.get("saves", 0) for i in insights) / total
        avg_reach = sum(i.get("reach", 0) for i in insights) / total

        engagement_rates = []
        for i in insights:
            rate = self.calculate_engagement_rate(
                i.get("likes", 0), i.get("comments", 0),
                i.get("saves", 0), i.get("reach", 1),
            )
            engagement_rates.append((i.get("media_id", ""), rate))

        avg_engagement = sum(r for _, r in engagement_rates) / total
        best = max(engagement_rates, key=lambda x: x[1])
        worst = min(engagement_rates, key=lambda x: x[1])

        # Find best posting hour
        hour_engagement: dict[int, list[float]] = {}
        for i in insights:
            hour = i.get("hour", 12)
            rate = self.calculate_engagement_rate(
                i.get("likes", 0), i.get("comments", 0),
                i.get("saves", 0), i.get("reach", 1),
            )
            hour_engagement.setdefault(hour, []).append(rate)

        best_hour = None
        if hour_engagement:
            best_hour = max(
                hour_engagement,
                key=lambda h: sum(hour_engagement[h]) / len(hour_engagement[h]),
            )

        dates = [i.get("collected_at") for i in insights if i.get("collected_at")]

        return PerformanceSummary(
            total_posts=total, avg_likes=avg_likes, avg_comments=avg_comments,
            avg_saves=avg_saves, avg_reach=avg_reach,
            avg_engagement_rate=avg_engagement,
            best_post_id=best[0], worst_post_id=worst[0],
            best_hour=best_hour,
            period_start=min(dates) if dates else None,
            period_end=max(dates) if dates else None,
        )

    def compare_periods(
        self, current: list[dict], previous: list[dict]
    ) -> PeriodComparison:
        curr_summary = self.generate_performance_summary(current)
        prev_summary = self.generate_performance_summary(previous)

        eng_change = 0.0
        if prev_summary.avg_engagement_rate > 0:
            eng_change = (
                (curr_summary.avg_engagement_rate - prev_summary.avg_engagement_rate)
                / prev_summary.avg_engagement_rate * 100
            )

        reach_change = 0.0
        if prev_summary.avg_reach > 0:
            reach_change = (
                (curr_summary.avg_reach - prev_summary.avg_reach)
                / prev_summary.avg_reach * 100
            )

        trending = eng_change > 0

        summary = (
            f"Engagement {'up' if trending else 'down'} {abs(eng_change):.1f}%. "
            f"Reach {'up' if reach_change > 0 else 'down'} {abs(reach_change):.1f}%."
        )

        return PeriodComparison(
            current_avg_engagement=curr_summary.avg_engagement_rate,
            previous_avg_engagement=prev_summary.avg_engagement_rate,
            engagement_change_pct=eng_change,
            reach_change_pct=reach_change,
            trending_up=trending,
            summary_text=summary,
        )

    def get_posting_recommendations(self, insights: list[dict]) -> list[str]:
        recommendations = []
        summary = self.generate_performance_summary(insights)

        if summary.best_hour is not None:
            recommendations.append(
                f"Best performing hour: {summary.best_hour}:00 — consider scheduling more posts around this time."
            )

        if summary.avg_engagement_rate < 0.03:
            recommendations.append(
                "Engagement rate is below 3%. Try more engaging captions with questions or CTAs."
            )

        if summary.avg_engagement_rate > 0.05:
            recommendations.append(
                "Strong engagement rate! Consider increasing posting frequency."
            )

        if summary.total_posts < 8:
            recommendations.append(
                "Low posting volume this period. Aim for 8+ posts per week for growth."
            )

        return recommendations
