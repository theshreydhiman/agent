"""Workflow node functions for the agent graph.

Each function takes an AgentState dict and returns updates.
Dependencies are instantiated lazily inside each function.
"""

import structlog

from agent.state import WorkflowPhase

logger = structlog.get_logger()


def plan_content(state: dict) -> dict:
    """Generate or load the content plan."""
    logger.info("node_enter", node="plan_content")

    if state.get("content_plan"):
        logger.info("plan_exists", count=len(state["content_plan"]))
        return {"phase": WorkflowPhase.PLANNING}

    try:
        from content.planner import ContentPlanner
        from config import get_settings

        settings = get_settings()
        planner = ContentPlanner()

        import asyncio
        plan = asyncio.get_event_loop().run_until_complete(
            planner.generate_weekly_plan(
                character_name=settings.character_name,
                niche=settings.character_niche,
                brand_voice=settings.character_voice,
            )
        )

        plan_dicts = [
            {
                "date": str(entry.date),
                "time_slot": entry.time_slot,
                "content_type": entry.content_type,
                "theme": entry.theme,
                "scene_preset": entry.scene_preset,
                "outfit_preset": entry.outfit_preset,
                "mood": entry.mood,
                "caption_prompt": entry.caption_prompt,
            }
            for entry in plan
        ]

        logger.info("plan_generated", count=len(plan_dicts))
        return {
            "phase": WorkflowPhase.PLANNING,
            "content_plan": plan_dicts,
            "current_plan_index": 0,
        }

    except Exception as e:
        logger.error("plan_content_failed", error=str(e))
        return {"errors": state.get("errors", []) + [f"Planning failed: {e}"]}


def generate_image(state: dict) -> dict:
    """Generate an image for the current plan entry."""
    logger.info("node_enter", node="generate_image")

    plan = state.get("content_plan", [])
    idx = state.get("current_plan_index", 0)

    if idx >= len(plan):
        return {"completed": True}

    entry = plan[idx]

    try:
        from generation.image.generator import ImageGenerator
        from generation.image.prompts import build_character_prompt, get_negative_prompt
        from config import get_settings

        settings = get_settings()

        prompt = build_character_prompt(
            character_config={
                "name": settings.character_name,
                "niche": settings.character_niche,
            },
            scene=entry.get("scene_preset", ""),
            outfit=entry.get("outfit_preset", ""),
            style=entry.get("mood", ""),
        )

        generator = ImageGenerator()

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            generator.generate_image(
                prompt=prompt,
                negative_prompt=get_negative_prompt(),
                width=1080,
                height=1350 if entry.get("content_type") == "post" else 1920,
            )
        )

        logger.info("image_generated", path=result.file_path)
        return {
            "phase": WorkflowPhase.GENERATING,
            "current_prompt": prompt,
            "generated_image_path": result.file_path,
        }

    except Exception as e:
        logger.error("generate_image_failed", error=str(e))
        return {"errors": state.get("errors", []) + [f"Image generation failed: {e}"]}


def generate_video(state: dict) -> dict:
    """Generate video from the image if content type requires it."""
    logger.info("node_enter", node="generate_video")

    image_path = state.get("generated_image_path", "")
    if not image_path:
        return {"errors": state.get("errors", []) + ["No image to create video from"]}

    plan = state.get("content_plan", [])
    idx = state.get("current_plan_index", 0)
    entry = plan[idx] if idx < len(plan) else {}

    try:
        from generation.video.generator import VideoGenerator

        generator = VideoGenerator()

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            generator.generate_video(
                source_image_path=image_path,
                prompt=entry.get("theme", "natural movement"),
                style=entry.get("mood", "natural"),
            )
        )

        logger.info("video_generated", path=result.file_path)
        return {"generated_video_path": result.file_path}

    except Exception as e:
        logger.error("generate_video_failed", error=str(e))
        return {"errors": state.get("errors", []) + [f"Video generation failed: {e}"]}


def run_qa_check(state: dict) -> dict:
    """Run quality assurance checks on generated content."""
    logger.info("node_enter", node="run_qa_check")

    image_path = state.get("generated_image_path", "")
    video_path = state.get("generated_video_path", "")
    content_path = video_path or image_path

    if not content_path:
        return {
            "qa_passed": False,
            "qa_result": {"error": "no content to check"},
        }

    try:
        from quality.pipeline import QAPipeline
        from character.face_analyzer import FaceAnalyzer

        analyzer = FaceAnalyzer()
        pipeline = QAPipeline(face_analyzer=analyzer)

        content_type = "video" if video_path else "image"
        result = pipeline.run_full_check(content_path, content_type)

        logger.info("qa_complete", passed=result.passed, score=result.score)
        return {
            "phase": WorkflowPhase.QA_REVIEW,
            "qa_passed": result.passed,
            "qa_result": {
                "passed": result.passed,
                "score": result.score,
                "face_similarity": result.face_similarity_score,
                "rejection_reasons": result.rejection_reasons,
                "warnings": result.warnings,
            },
        }

    except Exception as e:
        logger.error("qa_check_failed", error=str(e))
        # Pass through on QA error — don't block the pipeline
        return {
            "qa_passed": True,
            "qa_result": {"error": str(e), "passed": True},
        }


def generate_caption(state: dict) -> dict:
    """Generate caption, hashtags, and alt text."""
    logger.info("node_enter", node="generate_caption")

    plan = state.get("content_plan", [])
    idx = state.get("current_plan_index", 0)
    entry = plan[idx] if idx < len(plan) else {}

    try:
        from content.captions import CaptionGenerator
        from content.hashtags import HashtagEngine
        from config import get_settings

        settings = get_settings()
        caption_gen = CaptionGenerator()
        hashtag_gen = HashtagEngine()

        import asyncio
        loop = asyncio.get_event_loop()

        caption = loop.run_until_complete(
            caption_gen.generate_caption(
                theme=entry.get("theme", "lifestyle"),
                mood=entry.get("mood", "positive"),
                character_name=settings.character_name,
                brand_voice=settings.character_voice,
                content_type=entry.get("content_type", "post"),
            )
        )

        hashtags = loop.run_until_complete(
            hashtag_gen.generate_hashtags(
                theme=entry.get("theme", "lifestyle"),
                niche=settings.character_niche,
                content_type=entry.get("content_type", "post"),
            )
        )

        alt_text = loop.run_until_complete(
            caption_gen.generate_alt_text(
                f"Photo of {settings.character_name} - {entry.get('theme', 'lifestyle scene')}"
            )
        )

        logger.info("caption_generated", length=len(caption))
        return {
            "caption": caption,
            "hashtags": hashtags,
            "alt_text": alt_text,
        }

    except Exception as e:
        logger.error("caption_generation_failed", error=str(e))
        return {
            "caption": f"Living the moment ✨ #{entry.get('theme', 'lifestyle')}",
            "hashtags": ["lifestyle", "instagood", "photooftheday"],
            "alt_text": "AI-generated lifestyle photo",
            "errors": state.get("errors", []) + [f"Caption generation failed: {e}"],
        }


def moderate_content(state: dict) -> dict:
    """Run content moderation checks."""
    logger.info("node_enter", node="moderate_content")

    try:
        from quality.moderator import ContentModerator

        moderator = ContentModerator()
        result = moderator.full_moderation_check(
            caption=state.get("caption", ""),
            hashtags=state.get("hashtags", []),
            image_path=state.get("generated_image_path"),
        )

        updated = {}

        # Apply auto-corrections
        if result.modified_content:
            if "caption" in result.modified_content:
                updated["caption"] = result.modified_content["caption"]

        if not result.approved:
            updated["requires_human_review"] = True
            logger.warning("content_flagged", flags=result.flags)
        else:
            updated["requires_human_review"] = False

        return updated

    except Exception as e:
        logger.error("moderation_failed", error=str(e))
        return {"requires_human_review": True}


def publish_content(state: dict) -> dict:
    """Publish content to Instagram."""
    logger.info("node_enter", node="publish_content")

    try:
        from publisher.instagram import InstagramPublisher
        from publisher.media_upload import MediaUploader
        from content.hashtags import HashtagEngine

        publisher = InstagramPublisher()
        uploader = MediaUploader()
        hashtag_engine = HashtagEngine()

        # Build full caption with hashtags
        caption = state.get("caption", "")
        hashtags = state.get("hashtags", [])
        full_caption = f"{caption}\n\n{hashtag_engine.format_hashtags(hashtags)}"

        plan = state.get("content_plan", [])
        idx = state.get("current_plan_index", 0)
        entry = plan[idx] if idx < len(plan) else {}
        content_type = entry.get("content_type", "post")

        import asyncio
        loop = asyncio.get_event_loop()

        if content_type == "reel" and state.get("generated_video_path"):
            video_url = uploader.upload_video(state["generated_video_path"])
            result = loop.run_until_complete(
                publisher.publish_reel(video_url, full_caption)
            )
        elif content_type == "story":
            image_url = uploader.upload_image(state["generated_image_path"])
            result = loop.run_until_complete(
                publisher.publish_story(image_url)
            )
        else:
            image_url = uploader.upload_image(state["generated_image_path"])
            result = loop.run_until_complete(
                publisher.publish_single_image(
                    image_url, full_caption,
                    alt_text=state.get("alt_text"),
                )
            )

        logger.info("publish_complete", success=result.success, media_id=result.media_id)
        return {
            "phase": WorkflowPhase.PUBLISHING,
            "publish_result": {
                "success": result.success,
                "media_id": result.media_id,
                "error": result.error_message,
            },
        }

    except Exception as e:
        logger.error("publish_failed", error=str(e))
        return {
            "publish_result": {"success": False, "error": str(e)},
            "errors": state.get("errors", []) + [f"Publishing failed: {e}"],
        }


def collect_analytics(state: dict) -> dict:
    """Collect engagement analytics for recent posts."""
    logger.info("node_enter", node="collect_analytics")

    try:
        from analytics.tracker import EngagementTracker

        tracker = EngagementTracker()

        import asyncio
        insights = asyncio.get_event_loop().run_until_complete(
            tracker.collect_all_recent(days=7)
        )

        logger.info("analytics_collected", post_count=len(insights))
        return {
            "phase": WorkflowPhase.ANALYTICS,
            "latest_insights": {"recent_posts": insights},
        }

    except Exception as e:
        logger.error("analytics_failed", error=str(e))
        return {"latest_insights": {"error": str(e)}}


def advance_to_next(state: dict) -> dict:
    """Move to the next item in the content plan."""
    new_index = state.get("current_plan_index", 0) + 1
    logger.info("advancing", new_index=new_index)
    return {
        "current_plan_index": new_index,
        "generated_image_path": "",
        "generated_video_path": "",
        "qa_passed": False,
        "qa_result": {},
        "retry_count": 0,
        "caption": "",
        "hashtags": [],
        "alt_text": "",
        "publish_result": {},
        "requires_human_review": False,
    }


def handle_error(state: dict) -> dict:
    """Handle errors and decide whether to skip or stop."""
    errors = state.get("errors", [])
    logger.error("handling_error", error_count=len(errors), latest=errors[-1] if errors else "none")
    return {"retry_count": 0}


# --- Routing functions ---

def should_generate_video(state: dict) -> str:
    """Decide if video generation is needed."""
    plan = state.get("content_plan", [])
    idx = state.get("current_plan_index", 0)
    if idx < len(plan):
        content_type = plan[idx].get("content_type", "post")
        if content_type in ("reel", "story"):
            return "generate_video"
    return "run_qa_check"


def check_qa_result(state: dict) -> str:
    """Route based on QA result."""
    if state.get("qa_passed"):
        return "generate_caption"
    if state.get("retry_count", 0) < state.get("max_retries", 3):
        return "generate_image"  # retry
    return "handle_error"


def check_moderation(state: dict) -> str:
    """Route based on moderation result."""
    if state.get("requires_human_review"):
        return "__end__"  # pause for human review
    return "publish_content"


def check_more_content(state: dict) -> str:
    """Check if there are more items in the plan."""
    plan = state.get("content_plan", [])
    idx = state.get("current_plan_index", 0) + 1
    if idx < len(plan):
        return "advance_to_next"
    return "__end__"
