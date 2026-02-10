"""Central configuration loaded from environment variables."""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_env: str = "development"
    app_debug: bool = True
    app_secret_key: str = "change-me"

    # Database
    database_url: str = "postgresql+asyncpg://agent:agent@localhost:5432/influencer_agent"
    database_sync_url: str = "postgresql://agent:agent@localhost:5432/influencer_agent"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Instagram
    meta_app_id: str = ""
    meta_app_secret: str = ""
    instagram_account_id: str = ""
    instagram_access_token: str = ""

    # LLM (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.3:70b"

    # ComfyUI
    comfyui_base_url: str = "http://localhost:8188"
    comfyui_output_dir: str = "/output"
    comfyui_checkpoint_name: str = "flux1-dev.safetensors"

    # CogVideoX
    cogvideo_base_url: str = "http://localhost:8100"

    # S3 / MinIO
    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket_name: str = "influencer-media"
    s3_region: str = "us-east-1"

    # Character
    character_name: str = "Luna"
    character_niche: str = "lifestyle"
    character_voice: str = "friendly, witty, approachable"

    # Content schedule
    posts_per_week: int = 8
    reels_per_week: int = 3
    stories_per_week: int = 5
    posting_timezone: str = "America/New_York"

    # Quality thresholds
    face_similarity_hard_floor: float = 0.80
    face_similarity_target: float = 0.85
    face_similarity_goal: float = 0.90

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
