"""Celery application configuration."""

from celery import Celery

from config import get_settings

settings = get_settings()

celery_app = Celery(
    "influencer_agent",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone=settings.posting_timezone,
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Beat schedule for periodic tasks
    beat_schedule={
        "generate-daily-content": {
            "task": "tasks.workers.generate_daily_content",
            "schedule": 3600.0 * 6,  # Every 6 hours
        },
        "publish-scheduled-content": {
            "task": "tasks.workers.publish_scheduled_content",
            "schedule": 300.0,  # Every 5 minutes
        },
        "collect-analytics": {
            "task": "tasks.workers.collect_analytics",
            "schedule": 3600.0,  # Every hour
        },
    },
)
