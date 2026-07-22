"""Streamlit browser for multimodal video-search results."""

from pathlib import Path

import streamlit as st
import yaml

from src.inference import search
from src.agents.orchestrator import ReActOrchestrator, TaskType


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@st.cache_resource
def load_config() -> dict:
    """Load once so inference can reuse its process-lifetime model/index cache."""
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def keyframe_path(video_id: str, frame_idx: int, config: dict) -> Path:
    frame_id = f"{video_id}_{frame_idx:06d}"
    return PROJECT_ROOT / config["data"]["keyframe_dir"] / video_id / f"{frame_id}.jpg"


def video_path(video_id: str, config: dict) -> Path:
    return PROJECT_ROOT / config["data"]["video_dir"] / f"{video_id}.mp4"


def render_result(result: dict, rank: int, config: dict) -> None:
    """Render a single result without requiring source videos/keyframes to exist."""
    frame_path = keyframe_path(result["video_id"], result["frame_idx"], config)
    source_video = video_path(result["video_id"], config)
    title = (
        f"#{rank} · {result['video_id']} · "
        f"{result['timestamp_sec']:.2f}s · score {result['score']:.4f}"
    )

    with st.expander(title, expanded=rank == 1):
        image_col, detail_col = st.columns((2, 1))
        with image_col:
            if frame_path.exists():
                st.image(str(frame_path), caption=f"Frame {result['frame_idx']}")
            else:
                st.warning(f"Keyframe is unavailable: {frame_path.relative_to(PROJECT_ROOT)}")
        with detail_col:
            st.metric("Fusion score", f"{result['score']:.4f}")
            st.write(f"**Frame:** {result['frame_idx']}")
            st.write(f"**Timestamp:** {result['timestamp_sec']:.2f} seconds")
            st.code(
                f"{result['video_id']},{result['timestamp_sec']:.2f}",
                language=None,
            )
            if source_video.exists():
                st.video(str(source_video), start_time=float(result["timestamp_sec"]))
            else:
                st.caption(f"Source video unavailable: {source_video.relative_to(PROJECT_ROOT)}")


st.set_page_config(page_title="AIC 2026 Search", page_icon="🎬", layout="wide")
st.title("🎬 AIC 2026 — Multimodal Video Search")
st.caption("SigLIP semantic retrieval + OCR/ASR BM25 search, fused by reciprocal rank.")

config = load_config()
planner = ReActOrchestrator(search)
if "conversation" not in st.session_state:
    st.session_state.conversation = []

with st.sidebar:
    st.header("Search settings")
    top_k = st.slider("Results", min_value=1, max_value=50, value=10)
    task_label = st.selectbox("Task type", ["Auto", *[task.value for task in TaskType]])
    if st.button("Clear conversation"):
        st.session_state.conversation = []
    st.caption(f"Vector index: `{config['turbovec']['index_dir']}`")
    st.caption(f"Text index: `{config['elasticsearch']['index_name']}`")

with st.form("search_form"):
    query = st.text_input(
        "Describe the moment you want to find",
        placeholder="e.g. a person reading a sign in a market",
    )
    submitted = st.form_submit_button("Search", type="primary")

if submitted:
    if not query.strip():
        st.warning("Enter a query before searching.")
    else:
        try:
            with st.spinner("Searching visual, OCR, and speech indexes..."):
                task_override = None if task_label == "Auto" else TaskType(task_label)
                response = planner.run(
                    query.strip(),
                    config,
                    top_k=top_k,
                    history=st.session_state.conversation,
                    task_override=task_override,
                )
        except Exception as exc:  # infrastructure/model failures should be actionable in the UI
            st.error(f"Search could not complete: {exc}")
            st.info(
                "Check that the SigLIP Turbovec index has been built and Elasticsearch is running."
            )
        else:
            st.session_state.conversation.append(query.strip())
            st.caption(f"{response.plan.task_type} · {' → '.join(response.plan.actions)}")
            st.write(response.summary)
            results = response.results
            if not results:
                if not response.plan.clarification:
                    st.info("No matching indexed frames were found.")
            else:
                st.success(f"Found {len(results)} ranked frame matches.")
                if response.clips:
                    with st.expander("Temporal candidate clips"):
                        for clip in response.clips:
                            st.write(
                                f"**{clip.video_id}** · {clip.start_sec:.2f}s–{clip.end_sec:.2f}s "
                                f"· {len(clip.frames)} evidence frames"
                            )
                for rank, result in enumerate(results, start=1):
                    render_result(result, rank, config)
else:
    st.info("Index videos first, then describe a scene to search across visual, OCR, and ASR evidence.")
