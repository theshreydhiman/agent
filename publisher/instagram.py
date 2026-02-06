"""Instagram Graph API publisher for all content types."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime

import httpx
import structlog

from config import get_settings

logger = structlog.get_logger()

BASE_URL = "https://graph.instagram.com/v21.0"


@dataclass
class PublishResult:
    success: bool
    media_id: str | None = None
    creation_id: str | None = None
    error_message: str | None = None
    api_response: dict | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class InstagramPublisher:
    """Publishes content to Instagram via Meta Graph API."""

    def __init__(self):
        settings = get_settings()
        self._account_id = settings.instagram_account_id
        self._access_token = settings.instagram_access_token
        self._app_id = settings.meta_app_id
        self._app_secret = settings.meta_app_secret

    async def _api_request(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
        retries: int = 3,
    ) -> dict:
        """Make an API request with retry logic and rate limit handling."""
        url = f"{BASE_URL}/{endpoint}"
        params = params or {}
        params["access_token"] = self._access_token

        for attempt in range(1, retries + 1):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    if method.upper() == "GET":
                        resp = await client.get(url, params=params)
                    else:
                        resp = await client.post(url, data=params)

                    if resp.status_code == 429:
                        wait = 2 ** attempt
                        logger.warning("rate_limited", wait=wait, attempt=attempt)
                        await asyncio.sleep(wait)
                        continue

                    resp.raise_for_status()
                    return resp.json()

            except httpx.HTTPStatusError as e:
                logger.error(
                    "api_error", status=e.response.status_code,
                    body=e.response.text, attempt=attempt,
                )
                if attempt == retries:
                    return {"error": str(e), "status_code": e.response.status_code}
                await asyncio.sleep(2 ** attempt)

            except httpx.RequestError as e:
                logger.error("request_error", error=str(e), attempt=attempt)
                if attempt == retries:
                    return {"error": str(e)}
                await asyncio.sleep(2 ** attempt)

        return {"error": "max retries exceeded"}

    async def publish_single_image(
        self,
        image_url: str,
        caption: str,
        location_id: str | None = None,
        alt_text: str | None = None,
    ) -> PublishResult:
        """Publish a single image post. Two-step: create container, then publish."""
        # Step 1: Create media container
        params = {"image_url": image_url, "caption": caption}
        if location_id:
            params["location_id"] = location_id
        if alt_text:
            params["alt_text"] = alt_text

        container = await self._api_request(
            "POST", f"{self._account_id}/media", params
        )
        if "error" in container:
            return PublishResult(
                success=False, error_message=container.get("error"),
                api_response=container,
            )

        creation_id = container.get("id")

        # Step 2: Publish
        result = await self._api_request(
            "POST", f"{self._account_id}/media_publish",
            {"creation_id": creation_id},
        )

        if "id" in result:
            logger.info("image_published", media_id=result["id"])
            return PublishResult(
                success=True, media_id=result["id"],
                creation_id=creation_id, api_response=result,
            )

        return PublishResult(
            success=False, creation_id=creation_id,
            error_message=result.get("error"), api_response=result,
        )

    async def publish_carousel(
        self,
        image_urls: list[str],
        caption: str,
        alt_text: str | None = None,
    ) -> PublishResult:
        """Publish a carousel post (2-10 images). Three-step process."""
        if not 2 <= len(image_urls) <= 10:
            return PublishResult(
                success=False,
                error_message=f"Carousel requires 2-10 images, got {len(image_urls)}",
            )

        # Step 1: Create item containers
        item_ids = []
        for url in image_urls:
            params = {"image_url": url, "is_carousel_item": "true"}
            if alt_text:
                params["alt_text"] = alt_text
            result = await self._api_request(
                "POST", f"{self._account_id}/media", params
            )
            if "id" in result:
                item_ids.append(result["id"])
            else:
                return PublishResult(
                    success=False,
                    error_message=f"Failed to create carousel item: {result}",
                    api_response=result,
                )

        # Step 2: Create carousel container
        container = await self._api_request(
            "POST", f"{self._account_id}/media",
            {"media_type": "CAROUSEL", "caption": caption,
             "children": ",".join(item_ids)},
        )
        if "error" in container:
            return PublishResult(
                success=False, error_message=container.get("error"),
                api_response=container,
            )

        creation_id = container.get("id")

        # Step 3: Publish
        result = await self._api_request(
            "POST", f"{self._account_id}/media_publish",
            {"creation_id": creation_id},
        )

        if "id" in result:
            logger.info("carousel_published", media_id=result["id"])
            return PublishResult(
                success=True, media_id=result["id"],
                creation_id=creation_id, api_response=result,
            )

        return PublishResult(
            success=False, creation_id=creation_id,
            error_message=result.get("error"), api_response=result,
        )

    async def publish_reel(
        self,
        video_url: str,
        caption: str,
        cover_url: str | None = None,
        share_to_feed: bool = True,
    ) -> PublishResult:
        """Publish a Reel video."""
        params = {
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "share_to_feed": str(share_to_feed).lower(),
        }
        if cover_url:
            params["cover_url"] = cover_url

        # Step 1: Create container
        container = await self._api_request(
            "POST", f"{self._account_id}/media", params
        )
        if "error" in container:
            return PublishResult(
                success=False, error_message=container.get("error"),
                api_response=container,
            )

        creation_id = container.get("id")

        # Step 2: Wait for video processing
        for _ in range(30):  # Poll up to 5 minutes
            status = await self.get_media_status(creation_id)
            code = status.get("status_code")
            if code == "FINISHED":
                break
            if code == "ERROR":
                return PublishResult(
                    success=False, creation_id=creation_id,
                    error_message="Video processing failed",
                    api_response=status,
                )
            await asyncio.sleep(10)

        # Step 3: Publish
        result = await self._api_request(
            "POST", f"{self._account_id}/media_publish",
            {"creation_id": creation_id},
        )

        if "id" in result:
            logger.info("reel_published", media_id=result["id"])
            return PublishResult(
                success=True, media_id=result["id"],
                creation_id=creation_id, api_response=result,
            )

        return PublishResult(
            success=False, creation_id=creation_id,
            error_message=result.get("error"), api_response=result,
        )

    async def publish_story(
        self, media_url: str, media_type: str = "image"
    ) -> PublishResult:
        """Publish a Story (image or video)."""
        params: dict = {"media_type": "STORIES"}
        if media_type == "video":
            params["video_url"] = media_url
        else:
            params["image_url"] = media_url

        container = await self._api_request(
            "POST", f"{self._account_id}/media", params
        )
        if "error" in container:
            return PublishResult(
                success=False, error_message=container.get("error"),
                api_response=container,
            )

        creation_id = container.get("id")

        # For video stories, wait for processing
        if media_type == "video":
            for _ in range(30):
                status = await self.get_media_status(creation_id)
                if status.get("status_code") == "FINISHED":
                    break
                if status.get("status_code") == "ERROR":
                    return PublishResult(
                        success=False, creation_id=creation_id,
                        error_message="Story video processing failed",
                    )
                await asyncio.sleep(10)

        result = await self._api_request(
            "POST", f"{self._account_id}/media_publish",
            {"creation_id": creation_id},
        )

        if "id" in result:
            logger.info("story_published", media_id=result["id"])
            return PublishResult(
                success=True, media_id=result["id"],
                creation_id=creation_id, api_response=result,
            )

        return PublishResult(
            success=False, creation_id=creation_id,
            error_message=result.get("error"), api_response=result,
        )

    async def get_media_status(self, creation_id: str) -> dict:
        """Check media container processing status."""
        return await self._api_request(
            "GET", creation_id, {"fields": "status_code,status"}
        )

    async def refresh_token(self) -> str:
        """Exchange short-lived token for long-lived token."""
        result = await self._api_request(
            "GET", "oauth/access_token",
            {
                "grant_type": "ig_exchange_token",
                "client_secret": self._app_secret,
                "access_token": self._access_token,
            },
        )
        new_token = result.get("access_token", "")
        if new_token:
            self._access_token = new_token
            logger.info("token_refreshed")
        return new_token
