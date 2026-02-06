"""SQLAlchemy ORM models for the AI Virtual Influencer Agent."""

import enum
from datetime import date, datetime, time
from typing import Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Date,
    DateTime,
    ForeignKey,
    LargeBinary,
    Text,
    Time,
    func,
)
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.session import Base


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ContentType(str, enum.Enum):
    image = "image"
    video = "video"
    carousel = "carousel"


class QAStatus(str, enum.Enum):
    pending = "pending"
    passed = "passed"
    failed = "failed"
    flagged = "flagged"


class PlanContentType(str, enum.Enum):
    post = "post"
    story = "story"
    reel = "reel"
    carousel = "carousel"


class PlanStatus(str, enum.Enum):
    planned = "planned"
    content_ready = "content_ready"
    approved = "approved"
    published = "published"
    failed = "failed"


class PublishStatus(str, enum.Enum):
    pending = "pending"
    uploading = "uploading"
    published = "published"
    failed = "failed"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Character(Base):
    __tablename__ = "characters"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column()
    niche: Mapped[str] = mapped_column()
    personality_traits: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    appearance_config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    brand_voice: Mapped[str] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # relationships
    reference_images: Mapped[list["ReferenceImage"]] = relationship(
        back_populates="character", cascade="all, delete-orphan"
    )
    generated_contents: Mapped[list["GeneratedContent"]] = relationship(
        back_populates="character", cascade="all, delete-orphan"
    )
    content_plans: Mapped[list["ContentPlan"]] = relationship(
        back_populates="character", cascade="all, delete-orphan"
    )


class ReferenceImage(Base):
    __tablename__ = "reference_images"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    character_id: Mapped[UUID] = mapped_column(
        ForeignKey("characters.id", ondelete="CASCADE")
    )
    file_path: Mapped[str] = mapped_column()
    s3_key: Mapped[str] = mapped_column()
    angle: Mapped[str] = mapped_column()
    expression: Mapped[str] = mapped_column()
    embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # relationships
    character: Mapped["Character"] = relationship(back_populates="reference_images")


class GeneratedContent(Base):
    __tablename__ = "generated_contents"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    character_id: Mapped[UUID] = mapped_column(
        ForeignKey("characters.id", ondelete="CASCADE")
    )
    content_type: Mapped[ContentType] = mapped_column()
    file_path: Mapped[str] = mapped_column()
    s3_key: Mapped[str] = mapped_column()
    prompt_used: Mapped[str] = mapped_column(Text)
    face_similarity_score: Mapped[Optional[float]] = mapped_column(nullable=True)
    resolution_width: Mapped[int] = mapped_column()
    resolution_height: Mapped[int] = mapped_column()
    duration_seconds: Mapped[Optional[float]] = mapped_column(nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    qa_status: Mapped[QAStatus] = mapped_column(default=QAStatus.pending)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # relationships
    character: Mapped["Character"] = relationship(back_populates="generated_contents")
    content_plans: Mapped[list["ContentPlan"]] = relationship(
        back_populates="generated_content"
    )
    qa_reviews: Mapped[list["QAReview"]] = relationship(
        back_populates="generated_content", cascade="all, delete-orphan"
    )


class ContentPlan(Base):
    __tablename__ = "content_plans"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    character_id: Mapped[UUID] = mapped_column(
        ForeignKey("characters.id", ondelete="CASCADE")
    )
    scheduled_date: Mapped[date] = mapped_column(Date)
    scheduled_time: Mapped[time] = mapped_column(Time)
    content_type: Mapped[PlanContentType] = mapped_column()
    theme: Mapped[str] = mapped_column()
    caption: Mapped[str] = mapped_column(Text)
    hashtags: Mapped[list[str]] = mapped_column(JSON, default=list)
    alt_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[PlanStatus] = mapped_column(default=PlanStatus.planned)
    generated_content_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("generated_contents.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # relationships
    character: Mapped["Character"] = relationship(back_populates="content_plans")
    generated_content: Mapped[Optional["GeneratedContent"]] = relationship(
        back_populates="content_plans"
    )
    publish_logs: Mapped[list["PublishLog"]] = relationship(
        back_populates="content_plan", cascade="all, delete-orphan"
    )


class PublishLog(Base):
    __tablename__ = "publish_logs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    content_plan_id: Mapped[UUID] = mapped_column(
        ForeignKey("content_plans.id", ondelete="CASCADE")
    )
    instagram_media_id: Mapped[Optional[str]] = mapped_column(nullable=True)
    status: Mapped[PublishStatus] = mapped_column(default=PublishStatus.pending)
    attempt_number: Mapped[int] = mapped_column(default=1)
    api_response: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    published_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # relationships
    content_plan: Mapped["ContentPlan"] = relationship(back_populates="publish_logs")
    engagement_metrics: Mapped[list["EngagementMetric"]] = relationship(
        back_populates="publish_log", cascade="all, delete-orphan"
    )


class EngagementMetric(Base):
    __tablename__ = "engagement_metrics"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    publish_log_id: Mapped[UUID] = mapped_column(
        ForeignKey("publish_logs.id", ondelete="CASCADE")
    )
    likes: Mapped[int] = mapped_column(default=0)
    comments: Mapped[int] = mapped_column(default=0)
    shares: Mapped[int] = mapped_column(default=0)
    saves: Mapped[int] = mapped_column(default=0)
    reach: Mapped[int] = mapped_column(default=0)
    impressions: Mapped[int] = mapped_column(default=0)
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    # relationships
    publish_log: Mapped["PublishLog"] = relationship(
        back_populates="engagement_metrics"
    )


class QAReview(Base):
    __tablename__ = "qa_reviews"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    generated_content_id: Mapped[UUID] = mapped_column(
        ForeignKey("generated_contents.id", ondelete="CASCADE")
    )
    face_similarity_score: Mapped[float] = mapped_column()
    resolution_check: Mapped[bool] = mapped_column()
    nsfw_check: Mapped[bool] = mapped_column()
    artifact_check: Mapped[bool] = mapped_column()
    overall_pass: Mapped[bool] = mapped_column()
    rejection_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reviewer: Mapped[str] = mapped_column()
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    # relationships
    generated_content: Mapped["GeneratedContent"] = relationship(
        back_populates="qa_reviews"
    )
