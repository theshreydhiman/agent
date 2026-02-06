"""Content generation and planning API endpoints."""

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

router = APIRouter()


class GenerateImageRequest(BaseModel):
    prompt: str
    width: int = 1080
    height: int = 1350
    scene_preset: str = ""
    outfit_preset: str = ""


class GenerateVideoRequest(BaseModel):
    image_path: str
    prompt: str
    style: str = "natural"
    duration: float = 5.0


class ContentPlanRequest(BaseModel):
    days: int = 7


@router.post("/generate-image")
async def generate_image(request: GenerateImageRequest, background_tasks: BackgroundTasks):
    """Queue an image generation job."""
    from tasks.workers import generate_single_image
    task = generate_single_image.delay(
        prompt=request.prompt, width=request.width, height=request.height,
    )
    return {"task_id": task.id, "status": "queued"}


@router.post("/generate-video")
async def generate_video(request: GenerateVideoRequest, background_tasks: BackgroundTasks):
    """Queue a video generation job."""
    from tasks.workers import generate_single_video
    task = generate_single_video.delay(
        image_path=request.image_path,
        prompt=request.prompt,
        duration=request.duration,
    )
    return {"task_id": task.id, "status": "queued"}


@router.post("/plan")
async def generate_content_plan(request: ContentPlanRequest):
    """Generate a content plan for the next N days."""
    from content.planner import ContentPlanner
    from config import get_settings

    settings = get_settings()
    planner = ContentPlanner()
    plan = await planner.generate_weekly_plan(
        character_name=settings.character_name,
        niche=settings.character_niche,
        brand_voice=settings.character_voice,
    )
    return {
        "plan": [
            {
                "date": str(entry.date),
                "time_slot": entry.time_slot,
                "content_type": entry.content_type,
                "theme": entry.theme,
                "scene_preset": entry.scene_preset,
                "outfit_preset": entry.outfit_preset,
            }
            for entry in plan
        ],
        "count": len(plan),
    }


@router.get("/plan")
async def get_current_plan():
    """Get the current content plan."""
    # In production, this would query the database
    return {"plan": [], "count": 0}


@router.post("/caption")
async def generate_caption(theme: str = "lifestyle", mood: str = "positive"):
    """Generate a caption for a given theme."""
    from content.captions import CaptionGenerator
    from config import get_settings

    settings = get_settings()
    gen = CaptionGenerator()
    caption = await gen.generate_caption(
        theme=theme, mood=mood,
        character_name=settings.character_name,
        brand_voice=settings.character_voice,
        content_type="post",
    )
    return {"caption": caption}


@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Check the status of a background task."""
    from tasks.celery_app import celery_app
    result = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None,
    }
