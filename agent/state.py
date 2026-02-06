"""Agent state that flows through the LangGraph workflow."""

from dataclasses import dataclass, field
from enum import Enum


class WorkflowPhase(str, Enum):
    PLANNING = "planning"
    GENERATING = "generating"
    QA_REVIEW = "qa_review"
    PUBLISHING = "publishing"
    ANALYTICS = "analytics"
    IDLE = "idle"


@dataclass
class AgentState:
    """State that flows through the agent workflow."""

    phase: WorkflowPhase = WorkflowPhase.IDLE
    character_id: str = ""

    # Planning
    content_plan: list[dict] = field(default_factory=list)
    current_plan_index: int = 0

    # Generation
    current_prompt: str = ""
    generated_image_path: str = ""
    generated_video_path: str = ""

    # QA
    qa_result: dict = field(default_factory=dict)
    qa_passed: bool = False
    retry_count: int = 0
    max_retries: int = 3

    # Publishing
    caption: str = ""
    hashtags: list[str] = field(default_factory=list)
    alt_text: str = ""
    publish_result: dict = field(default_factory=dict)

    # Analytics
    latest_insights: dict = field(default_factory=dict)

    # Control
    errors: list[str] = field(default_factory=list)
    requires_human_review: bool = False
    completed: bool = False
