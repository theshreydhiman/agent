"""LangGraph workflow definition for the influencer agent."""

import structlog
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent import nodes

logger = structlog.get_logger()


class InfluencerAgent:
    """Central orchestration agent using LangGraph."""

    def __init__(self):
        self._graph = self._build_graph()

    def _build_graph(self):
        """Construct the LangGraph workflow."""
        graph = StateGraph(dict)

        # Add nodes
        graph.add_node("plan_content", nodes.plan_content)
        graph.add_node("generate_image", nodes.generate_image)
        graph.add_node("generate_video", nodes.generate_video)
        graph.add_node("run_qa_check", nodes.run_qa_check)
        graph.add_node("generate_caption", nodes.generate_caption)
        graph.add_node("moderate_content", nodes.moderate_content)
        graph.add_node("publish_content", nodes.publish_content)
        graph.add_node("collect_analytics", nodes.collect_analytics)
        graph.add_node("advance_to_next", nodes.advance_to_next)
        graph.add_node("handle_error", nodes.handle_error)

        # Set entry point
        graph.set_entry_point("plan_content")

        # Edges
        graph.add_edge("plan_content", "generate_image")

        # After image generation, decide if we need video
        graph.add_conditional_edges(
            "generate_image",
            nodes.should_generate_video,
            {
                "generate_video": "generate_video",
                "run_qa_check": "run_qa_check",
            },
        )

        graph.add_edge("generate_video", "run_qa_check")

        # After QA, decide next step
        graph.add_conditional_edges(
            "run_qa_check",
            nodes.check_qa_result,
            {
                "generate_caption": "generate_caption",
                "generate_image": "generate_image",  # retry
                "handle_error": "handle_error",
            },
        )

        graph.add_edge("generate_caption", "moderate_content")

        # After moderation, publish or wait for human review
        graph.add_conditional_edges(
            "moderate_content",
            nodes.check_moderation,
            {
                "publish_content": "publish_content",
                "__end__": END,
            },
        )

        graph.add_edge("publish_content", "collect_analytics")

        # After analytics, check if more content to process
        graph.add_conditional_edges(
            "collect_analytics",
            nodes.check_more_content,
            {
                "advance_to_next": "advance_to_next",
                "__end__": END,
            },
        )

        graph.add_edge("advance_to_next", "generate_image")

        # Error handler — check if we can continue with next item
        graph.add_conditional_edges(
            "handle_error",
            nodes.check_more_content,
            {
                "advance_to_next": "advance_to_next",
                "__end__": END,
            },
        )

        return graph.compile()

    def run(self, character_id: str) -> dict:
        """Execute the full workflow for a character."""
        logger.info("agent_run_start", character_id=character_id)

        initial_state = {
            "phase": "idle",
            "character_id": character_id,
            "content_plan": [],
            "current_plan_index": 0,
            "current_prompt": "",
            "generated_image_path": "",
            "generated_video_path": "",
            "qa_result": {},
            "qa_passed": False,
            "retry_count": 0,
            "max_retries": 3,
            "caption": "",
            "hashtags": [],
            "alt_text": "",
            "publish_result": {},
            "latest_insights": {},
            "errors": [],
            "requires_human_review": False,
            "completed": False,
        }

        final_state = self._graph.invoke(initial_state)
        logger.info("agent_run_complete", errors=len(final_state.get("errors", [])))
        return final_state

    def run_single(self, character_id: str, plan_entry: dict) -> dict:
        """Run the workflow for a single content piece."""
        logger.info("agent_run_single", character_id=character_id)

        initial_state = {
            "phase": "idle",
            "character_id": character_id,
            "content_plan": [plan_entry],
            "current_plan_index": 0,
            "current_prompt": "",
            "generated_image_path": "",
            "generated_video_path": "",
            "qa_result": {},
            "qa_passed": False,
            "retry_count": 0,
            "max_retries": 3,
            "caption": "",
            "hashtags": [],
            "alt_text": "",
            "publish_result": {},
            "latest_insights": {},
            "errors": [],
            "requires_human_review": False,
            "completed": False,
        }

        return self._graph.invoke(initial_state)

    def resume_after_approval(self, state: dict) -> dict:
        """Resume workflow after human approval of flagged content."""
        logger.info("agent_resume", character_id=state.get("character_id"))
        state["requires_human_review"] = False
        # Continue from publish step
        from publisher.instagram import InstagramPublisher
        result = nodes.publish_content(state)
        state.update(result)
        return state
