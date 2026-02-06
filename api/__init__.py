"""API routes for the influencer agent."""

from fastapi import APIRouter

from api.content import router as content_router
from api.publishing import router as publishing_router
from api.analytics import router as analytics_router
from api.character import router as character_router

router = APIRouter()
router.include_router(content_router, prefix="/content", tags=["content"])
router.include_router(publishing_router, prefix="/publishing", tags=["publishing"])
router.include_router(analytics_router, prefix="/analytics", tags=["analytics"])
router.include_router(character_router, prefix="/character", tags=["character"])
