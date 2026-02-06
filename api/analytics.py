"""Analytics API endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/summary")
async def get_performance_summary(days: int = 7):
    """Get engagement performance summary."""
    from analytics.tracker import EngagementTracker
    from analytics.reporter import AnalyticsReporter

    tracker = EngagementTracker()
    reporter = AnalyticsReporter()

    insights = await tracker.collect_all_recent(days=days)

    # Convert PostInsights to dicts for the reporter
    insights_dicts = [
        {
            "media_id": i.media_id,
            "likes": i.likes,
            "comments": i.comments,
            "saves": i.saves,
            "reach": i.reach,
            "impressions": i.impressions,
            "collected_at": i.collected_at,
        }
        for i in insights
    ]

    summary = reporter.generate_performance_summary(insights_dicts)

    return {
        "total_posts": summary.total_posts,
        "avg_likes": summary.avg_likes,
        "avg_comments": summary.avg_comments,
        "avg_saves": summary.avg_saves,
        "avg_reach": summary.avg_reach,
        "avg_engagement_rate": summary.avg_engagement_rate,
        "best_post_id": summary.best_post_id,
        "best_hour": summary.best_hour,
    }


@router.get("/followers")
async def get_follower_count():
    """Get current follower count."""
    from analytics.tracker import EngagementTracker

    tracker = EngagementTracker()
    count = await tracker.fetch_follower_count()
    return {"followers": count}


@router.get("/recommendations")
async def get_recommendations(days: int = 14):
    """Get posting recommendations based on recent performance."""
    from analytics.tracker import EngagementTracker
    from analytics.reporter import AnalyticsReporter

    tracker = EngagementTracker()
    reporter = AnalyticsReporter()

    insights = await tracker.collect_all_recent(days=days)
    insights_dicts = [
        {
            "media_id": i.media_id,
            "likes": i.likes,
            "comments": i.comments,
            "saves": i.saves,
            "reach": i.reach,
        }
        for i in insights
    ]

    recommendations = reporter.get_posting_recommendations(insights_dicts)
    return {"recommendations": recommendations}
