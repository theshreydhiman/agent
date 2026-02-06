"""Character management API endpoints."""

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

router = APIRouter()


class CharacterConfig(BaseModel):
    name: str
    niche: str
    brand_voice: str
    personality_traits: list[str] = []
    appearance: dict = {}


class CharacterResponse(BaseModel):
    id: str
    name: str
    niche: str
    brand_voice: str
    reference_image_count: int = 0


@router.get("/", response_model=CharacterResponse)
async def get_character():
    """Get current character configuration."""
    from config import get_settings
    settings = get_settings()
    return CharacterResponse(
        id="default",
        name=settings.character_name,
        niche=settings.character_niche,
        brand_voice=settings.character_voice,
    )


@router.put("/")
async def update_character(config: CharacterConfig):
    """Update character configuration."""
    # In production, this would update the database
    return {"status": "updated", "character": config.model_dump()}


@router.post("/reference-images")
async def upload_reference_image(file: UploadFile = File(...)):
    """Upload a new reference image for the character."""
    contents = await file.read()
    # In production: save to disk, extract embedding, store in DB
    return {
        "status": "uploaded",
        "filename": file.filename,
        "size": len(contents),
    }


@router.get("/reference-images")
async def list_reference_images():
    """List all reference images for the character."""
    return {"images": [], "count": 0}
