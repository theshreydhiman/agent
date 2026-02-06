"""Celery worker tasks for async job processing."""

import structlog

from tasks.celery_app import celery_app

logger = structlog.get_logger()


@celery_app.task(name="tasks.workers.generate_daily_content", bind=True, max_retries=2)
def generate_daily_content(self):
    """Generate content for the next day's posts."""
    logger.info("task_start", task="generate_daily_content")

    try:
        from agent.graph import InfluencerAgent
        from config import get_settings

        settings = get_settings()
        agent = InfluencerAgent()

        # Run the full agent pipeline
        result = agent.run(character_id="default")

        errors = result.get("errors", [])
        if errors:
            logger.warning("generation_completed_with_errors", errors=errors)
        else:
            logger.info("generation_completed")

        return {"status": "completed", "errors": errors}

    except Exception as e:
        logger.error("generate_daily_content_failed", error=str(e))
        raise self.retry(exc=e, countdown=60 * 5)


@celery_app.task(name="tasks.workers.publish_scheduled_content", bind=True, max_retries=3)
def publish_scheduled_content(self):
    """Check for and publish any content that's due."""
    logger.info("task_start", task="publish_scheduled_content")

    try:
        from datetime import datetime, timedelta
        from config import get_settings

        settings = get_settings()

        # In production, this would query the database for approved content
        # where scheduled_time <= now and status == 'approved'
        logger.info("checking_scheduled_content")

        # Placeholder: actual DB query would go here
        # For now, just log that we checked
        return {"status": "checked", "published": 0}

    except Exception as e:
        logger.error("publish_scheduled_failed", error=str(e))
        raise self.retry(exc=e, countdown=60)


@celery_app.task(name="tasks.workers.collect_analytics")
def collect_analytics():
    """Collect engagement metrics for recent posts."""
    logger.info("task_start", task="collect_analytics")

    try:
        from analytics.tracker import EngagementTracker
        import asyncio

        tracker = EngagementTracker()
        insights = asyncio.get_event_loop().run_until_complete(
            tracker.collect_all_recent(days=1)
        )

        logger.info("analytics_collected", count=len(insights))
        return {"status": "collected", "post_count": len(insights)}

    except Exception as e:
        logger.error("collect_analytics_failed", error=str(e))
        return {"status": "failed", "error": str(e)}


@celery_app.task(name="tasks.workers.generate_single_image")
def generate_single_image(prompt: str, width: int = 1080, height: int = 1350):
    """Generate a single image on demand."""
    logger.info("task_start", task="generate_single_image")

    try:
        from generation.image.generator import ImageGenerator
        import asyncio

        generator = ImageGenerator()
        result = asyncio.get_event_loop().run_until_complete(
            generator.generate_image(prompt=prompt, width=width, height=height)
        )

        return {
            "status": "completed",
            "file_path": result.file_path,
            "seed": result.seed,
        }

    except Exception as e:
        logger.error("generate_image_failed", error=str(e))
        return {"status": "failed", "error": str(e)}


@celery_app.task(name="tasks.workers.generate_single_video")
def generate_single_video(image_path: str, prompt: str, duration: float = 5.0):
    """Generate a single video on demand."""
    logger.info("task_start", task="generate_single_video")

    try:
        from generation.video.generator import VideoGenerator
        import asyncio

        generator = VideoGenerator()
        result = asyncio.get_event_loop().run_until_complete(
            generator.generate_video(
                source_image_path=image_path, prompt=prompt, duration=duration,
            )
        )

        return {
            "status": "completed",
            "file_path": result.file_path,
            "duration": result.duration_seconds,
        }

    except Exception as e:
        logger.error("generate_video_failed", error=str(e))
        return {"status": "failed", "error": str(e)}
