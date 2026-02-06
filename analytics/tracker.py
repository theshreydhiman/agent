"""Engagement tracking via the Instagram Graph API."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import httpx
import structlog

from config import get_settings

logger = structlog.get_logger(__name__)

BASE_URL = "https://graph.instagram.com/v21.0"
MAX_RETRIES = 3
RETRY_BACKOFF = 1.5  # seconds, multiplied each attempt


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PostInsights:
    """Engagement data for a single published post."""

    media_id: str
    likes: int = 0
    comments: int = 0
    shares: int = 0
    saves: int = 0
    reach: int = 0
    impressions: int = 0
    engagement_rate: float = 0.0
    collected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class AccountInsights:
    """Account-level metrics over a given period."""

    followers: int = 0
    following: int = 0
    posts_count: int = 0
    reach: int = 0
    impressions: int = 0
    profile_views: int = 0
    period_start: datetime | None = None
    period_end: datetime | None = None


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class EngagementTracker:
    """Collects engagement metrics from the Instagram Graph API."""

    def __init__(self) -> None:
        settings = get_settings()
        self._access_token: str = settings.instagram_access_token
        self._ig_user_id: str = settings.instagram_account_id
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=httpx.Timeout(30.0),
        )

    # -- public API ---------------------------------------------------------

    async def fetch_post_insights(self, media_id: str) -> PostInsights:
        """Fetch engagement data for a single post.

        Two calls are made:
        1. ``GET /{media_id}/insights`` with engagement metrics.
        2. ``GET /{media_id}`` with public counts and timestamp.
        """
        insights_params = {
            "metric": "engagement,impressions,reach,saved",
        }
        fields_params = {
            "fields": "like_count,comments_count,timestamp",
        }

        insights_data, media_data = await asyncio.gather(
            self._api_get(f"/{media_id}/insights", params=insights_params),
            self._api_get(f"/{media_id}", params=fields_params),
        )

        # Parse the insights response (/insights returns a list of metric objects)
        metrics: dict[str, int] = {}
        for entry in insights_data.get("data", []):
            name = entry.get("name", "")
            values = entry.get("values", [{}])
            metrics[name] = values[0].get("value", 0) if values else 0

        likes = media_data.get("like_count", 0)
        comments = media_data.get("comments_count", 0)
        shares = metrics.get("engagement", 0)  # total engagement actions
        saves = metrics.get("saved", 0)
        reach = metrics.get("reach", 0)
        impressions = metrics.get("impressions", 0)

        engagement_rate = self._compute_engagement_rate(likes, comments, saves, reach)

        timestamp_str = media_data.get("timestamp")
        collected_at = (
            datetime.fromisoformat(timestamp_str.replace("+0000", "+00:00"))
            if timestamp_str
            else datetime.now(timezone.utc)
        )

        post = PostInsights(
            media_id=media_id,
            likes=likes,
            comments=comments,
            shares=shares,
            saves=saves,
            reach=reach,
            impressions=impressions,
            engagement_rate=engagement_rate,
            collected_at=collected_at,
        )
        logger.info("post_insights_fetched", media_id=media_id, engagement_rate=engagement_rate)
        return post

    async def fetch_account_insights(
        self,
        period: str = "day",
        days: int = 7,
    ) -> AccountInsights:
        """Fetch account-level metrics over the given period.

        ``period`` must be ``"day"`` or ``"week"``.
        """
        if period not in ("day", "week"):
            raise ValueError(f"period must be 'day' or 'week', got {period!r}")

        now = datetime.now(timezone.utc)
        since = now - timedelta(days=days)

        params = {
            "metric": "impressions,reach,follower_count,profile_views",
            "period": period,
            "since": int(since.timestamp()),
            "until": int(now.timestamp()),
        }
        data = await self._api_get(f"/{self._ig_user_id}/insights", params=params)

        metrics: dict[str, int] = {}
        for entry in data.get("data", []):
            name = entry.get("name", "")
            values = entry.get("values", [])
            total = sum(v.get("value", 0) for v in values)
            metrics[name] = total

        account = AccountInsights(
            followers=metrics.get("follower_count", 0),
            reach=metrics.get("reach", 0),
            impressions=metrics.get("impressions", 0),
            profile_views=metrics.get("profile_views", 0),
            period_start=since,
            period_end=now,
        )
        logger.info("account_insights_fetched", period=period, days=days)
        return account

    async def fetch_follower_count(self) -> int:
        """Return the current follower count for the connected account."""
        data = await self._api_get(
            f"/{self._ig_user_id}",
            params={"fields": "followers_count"},
        )
        count = data.get("followers_count", 0)
        logger.info("follower_count_fetched", followers=count)
        return count

    async def collect_all_recent(self, days: int = 7) -> list[PostInsights]:
        """Fetch insights for every post published in the last *days* days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        # Step 1 -- get recent media ids
        media_data = await self._api_get(
            f"/{self._ig_user_id}/media",
            params={"fields": "id,timestamp", "limit": "100"},
        )

        media_ids: list[str] = []
        for item in media_data.get("data", []):
            ts_str = item.get("timestamp", "")
            if ts_str:
                ts = datetime.fromisoformat(ts_str.replace("+0000", "+00:00"))
                if ts >= cutoff:
                    media_ids.append(item["id"])

        if not media_ids:
            logger.info("no_recent_posts", days=days)
            return []

        # Step 2 -- fetch insights concurrently
        tasks = [self.fetch_post_insights(mid) for mid in media_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        insights: list[PostInsights] = []
        for result in results:
            if isinstance(result, BaseException):
                logger.error("post_insight_failed", error=str(result))
                continue
            insights.append(result)

        logger.info("recent_insights_collected", count=len(insights), days=days)
        return insights

    # -- internals ----------------------------------------------------------

    async def _api_get(self, endpoint: str, params: dict | None = None) -> dict:
        """Execute a GET request against the Graph API with retry logic."""
        params = dict(params) if params else {}
        params["access_token"] = self._access_token

        url = endpoint  # httpx resolves relative to base_url

        last_exc: BaseException | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self._client.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                last_exc = exc
                wait = RETRY_BACKOFF * attempt
                logger.warning(
                    "api_get_retry",
                    endpoint=endpoint,
                    attempt=attempt,
                    error=str(exc),
                    wait=wait,
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(wait)

        logger.error("api_get_failed", endpoint=endpoint, attempts=MAX_RETRIES)
        raise last_exc  # type: ignore[misc]

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _compute_engagement_rate(
        likes: int,
        comments: int,
        saves: int,
        reach: int,
    ) -> float:
        if reach <= 0:
            return 0.0
        return (likes + comments + saves) / reach

    async def close(self) -> None:
        """Shut down the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> EngagementTracker:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
