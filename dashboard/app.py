"""Streamlit admin dashboard for the AI Influencer Agent."""

import streamlit as st
import httpx

API_BASE = "http://api:8000/api"

st.set_page_config(
    page_title="AI Influencer Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("AI Influencer Agent — Admin Dashboard")


# --- Sidebar ---
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Content Plan", "Generate", "Review Queue", "Analytics", "Settings"],
)


def api_get(endpoint: str) -> dict:
    """Make a GET request to the backend API."""
    try:
        resp = httpx.get(f"{API_BASE}{endpoint}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return {}


def api_post(endpoint: str, data: dict = None) -> dict:
    """Make a POST request to the backend API."""
    try:
        resp = httpx.post(f"{API_BASE}{endpoint}", json=data or {}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return {}


# --- Pages ---

if page == "Overview":
    st.header("Overview")

    col1, col2, col3, col4 = st.columns(4)

    character = api_get("/character/")
    if character:
        col1.metric("Character", character.get("name", "N/A"))
        col2.metric("Niche", character.get("niche", "N/A"))
        col3.metric("Reference Images", character.get("reference_image_count", 0))

    followers = api_get("/analytics/followers")
    if followers:
        col4.metric("Followers", followers.get("followers", 0))

    st.subheader("Recent Performance")
    summary = api_get("/analytics/summary?days=7")
    if summary and summary.get("total_posts", 0) > 0:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Posts (7d)", summary["total_posts"])
        col2.metric("Avg Likes", f"{summary['avg_likes']:.0f}")
        col3.metric("Avg Comments", f"{summary['avg_comments']:.0f}")
        col4.metric("Engagement Rate", f"{summary['avg_engagement_rate']:.2%}")
    else:
        st.info("No analytics data yet. Start publishing to see metrics.")


elif page == "Content Plan":
    st.header("Content Plan")

    if st.button("Generate Weekly Plan"):
        with st.spinner("Generating plan..."):
            result = api_post("/content/plan", {"days": 7})
            if result.get("plan"):
                st.success(f"Generated {result['count']} content entries")

    plan = api_get("/content/plan")
    if plan.get("plan"):
        for entry in plan["plan"]:
            with st.expander(f"{entry['date']} — {entry['content_type']} — {entry['theme']}"):
                st.write(f"**Time:** {entry.get('time_slot', 'TBD')}")
                st.write(f"**Scene:** {entry.get('scene_preset', 'N/A')}")
                st.write(f"**Outfit:** {entry.get('outfit_preset', 'N/A')}")
    else:
        st.info("No content plan yet. Click 'Generate Weekly Plan' to create one.")


elif page == "Generate":
    st.header("Generate Content")

    tab1, tab2 = st.tabs(["Image", "Video"])

    with tab1:
        prompt = st.text_area("Image Prompt", "A lifestyle photo of a young woman in a coffee shop")
        col1, col2 = st.columns(2)
        width = col1.number_input("Width", value=1080, step=1)
        height = col2.number_input("Height", value=1350, step=1)

        if st.button("Generate Image"):
            with st.spinner("Queuing image generation..."):
                result = api_post("/content/generate-image", {
                    "prompt": prompt, "width": width, "height": height,
                })
                if result.get("task_id"):
                    st.success(f"Task queued: {result['task_id']}")

    with tab2:
        image_path = st.text_input("Source Image Path")
        video_prompt = st.text_area("Video Prompt", "Subtle head movement and natural blinking")
        duration = st.slider("Duration (seconds)", 3, 15, 5)

        if st.button("Generate Video"):
            with st.spinner("Queuing video generation..."):
                result = api_post("/content/generate-video", {
                    "image_path": image_path,
                    "prompt": video_prompt,
                    "duration": duration,
                })
                if result.get("task_id"):
                    st.success(f"Task queued: {result['task_id']}")


elif page == "Review Queue":
    st.header("Content Review Queue")

    queue = api_get("/publishing/queue")
    if queue.get("queue"):
        for item in queue["queue"]:
            with st.expander(f"{item.get('content_type', 'post')} — {item.get('theme', '')}"):
                st.write(f"**Caption:** {item.get('caption', '')}")
                col1, col2 = st.columns(2)
                if col1.button("Approve", key=f"approve_{item.get('id')}"):
                    api_post(f"/publishing/approve/{item['id']}")
                    st.success("Approved!")
                if col2.button("Reject", key=f"reject_{item.get('id')}"):
                    api_post(f"/publishing/reject/{item['id']}")
                    st.warning("Rejected")
    else:
        st.info("No content pending review.")


elif page == "Analytics":
    st.header("Analytics")

    days = st.selectbox("Period", [7, 14, 30], index=0)
    summary = api_get(f"/analytics/summary?days={days}")

    if summary and summary.get("total_posts", 0) > 0:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Posts", summary["total_posts"])
        col2.metric("Avg Engagement", f"{summary['avg_engagement_rate']:.2%}")
        col3.metric("Best Hour", f"{summary.get('best_hour', 'N/A')}:00")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Likes", f"{summary['avg_likes']:.0f}")
        col2.metric("Avg Comments", f"{summary['avg_comments']:.0f}")
        col3.metric("Avg Saves", f"{summary['avg_saves']:.0f}")
        col4.metric("Avg Reach", f"{summary['avg_reach']:.0f}")
    else:
        st.info("No analytics data available yet.")

    st.subheader("Recommendations")
    recs = api_get(f"/analytics/recommendations?days={days}")
    if recs.get("recommendations"):
        for rec in recs["recommendations"]:
            st.write(f"- {rec}")


elif page == "Settings":
    st.header("Settings")

    character = api_get("/character/")

    name = st.text_input("Character Name", character.get("name", ""))
    niche = st.selectbox(
        "Niche",
        ["lifestyle", "fashion", "fitness", "travel", "beauty", "food", "tech"],
        index=0,
    )
    voice = st.text_area("Brand Voice", character.get("brand_voice", ""))

    if st.button("Save Settings"):
        result = api_post("/character/", {
            "name": name, "niche": niche, "brand_voice": voice,
        })
        if result:
            st.success("Settings saved!")

    st.subheader("Reference Images")
    uploaded = st.file_uploader("Upload Reference Image", type=["jpg", "jpeg", "png"])
    if uploaded:
        st.info(f"Upload support coming soon. File: {uploaded.name}")
