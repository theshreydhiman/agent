"""Publishing and scheduling API endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class PublishRequest(BaseModel):
    image_path: str | None = None
    video_path: str | None = None
    caption: str
    hashtags: list[str] = []
    content_type: str = "post"  # post, carousel, reel, story
    image_paths: list[str] = []  # for carousel


class ScheduleRequest(BaseModel):
    content_plan_id: str
    scheduled_time: str  # ISO format


@router.post("/publish")
async def publish_now(request: PublishRequest):
    """Publish content to Instagram immediately."""
    from publisher.instagram import InstagramPublisher
    from publisher.media_upload import MediaUploader
    from content.hashtags import HashtagEngine

    publisher = InstagramPublisher()
    uploader = MediaUploader()
    hashtag_engine = HashtagEngine()

    full_caption = request.caption
    if request.hashtags:
        full_caption += "\n\n" + hashtag_engine.format_hashtags(request.hashtags)

    if request.content_type == "reel" and request.video_path:
        video_url = uploader.upload_video(request.video_path)
        result = await publisher.publish_reel(video_url, full_caption)
    elif request.content_type == "carousel" and request.image_paths:
        image_urls = [uploader.upload_image(p) for p in request.image_paths]
        result = await publisher.publish_carousel(image_urls, full_caption)
    elif request.content_type == "story" and request.image_path:
        image_url = uploader.upload_image(request.image_path)
        result = await publisher.publish_story(image_url)
    elif request.image_path:
        image_url = uploader.upload_image(request.image_path)
        result = await publisher.publish_single_image(image_url, full_caption)
    else:
        return {"error": "No media provided for publishing"}

    return {
        "success": result.success,
        "media_id": result.media_id,
        "error": result.error_message,
    }


@router.post("/schedule")
async def schedule_post(request: ScheduleRequest):
    """Schedule content for future publishing."""
    # In production: update content plan entry with scheduled time
    return {
        "status": "scheduled",
        "content_plan_id": request.content_plan_id,
        "scheduled_time": request.scheduled_time,
    }


@router.get("/queue")
async def get_publish_queue():
    """Get the current publishing queue."""
    # In production: query DB for pending content
    return {"queue": [], "count": 0}


@router.post("/approve/{content_id}")
async def approve_content(content_id: str):
    """Approve flagged content for publishing."""
    # In production: update content status in DB, trigger publish
    return {"status": "approved", "content_id": content_id}


@router.post("/reject/{content_id}")
async def reject_content(content_id: str, reason: str = ""):
    """Reject content from the review queue."""
    return {"status": "rejected", "content_id": content_id, "reason": reason}
