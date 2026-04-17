import streamlit as st
import tempfile
import cv2
import os
from pathlib import Path

# Reduces fork/atexit issues when Streamlit reloads + transformers/sklearn import chain (esp. Python 3.13).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from video_processor import VideoProcessor
from summarizer import VideoSummarizer
from ollama_summarizer import summarize_frames_with_ollama
from summary_store import append_local_summary
from summary_templates import ANALYSIS_STYLES, DEFAULT_VISION_MODEL_LABEL, style_key_from_label
from db.search_video import (
    build_search_text,
    suggest_search_terms,
    search_similar,
)
from video_storage import persist_uploaded_video
from embeddings.embedder import embed_text
from db.video_store import insert_summary

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None

def get_credentials():
    """Username -> password. From env (single user) or Streamlit secrets."""
    try:
        if "passwords" in st.secrets:
            pw = st.secrets["passwords"]
            if pw:
                return pw
    except Exception:
        pass
    user = os.getenv("LOGIN_USERNAME")
    pwd = os.getenv("LOGIN_PASSWORD")
    if user and pwd:
        return {user: pwd}
    return {}


def _login_required() -> bool:
    """If True, show login form when credentials exist. Default False = open local app."""
    return os.getenv("REQUIRE_LOGIN", "").lower() in ("1", "true", "yes")


def _apply_login_bypass() -> None:
    """Open app without password by default; set REQUIRE_LOGIN=1 to force LOGIN_* / secrets."""
    if st.session_state.logged_in:
        return
    if not _login_required():
        st.session_state.logged_in = True
        st.session_state.username = os.getenv("OPEN_MODE_LABEL", "guest")
        return
    if os.getenv("DEV_SKIP_LOGIN", "").lower() in ("1", "true", "yes"):
        st.session_state.logged_in = True
        st.session_state.username = os.getenv("DEV_LOGIN_LABEL", "local-dev")
        return
    creds = get_credentials()
    if not creds:
        # REQUIRE_LOGIN=1 but no LOGIN_* / secrets: stay logged out; UI shows warning below
        return
    # REQUIRE_LOGIN=1 and credentials exist: show login form (stay logged out)


# Page configuration
st.set_page_config(
    page_title="Video Summarizer",
    page_icon="🎥",
    layout="wide"
)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'frames' not in st.session_state:
    st.session_state.frames = None
if 'search_hints' not in st.session_state:
    st.session_state.search_hints = []

# Title and description
st.title("🎥 Video Summarizer with Cosmos AI")
_cosmos_id = os.getenv("COSMOS_MODEL", "nvidia/Cosmos-Reason2-2B")
_cosmos_label = os.getenv("COSMOS_MODEL_LABEL", DEFAULT_VISION_MODEL_LABEL)
st.markdown(
    f"Upload a video for AI summaries. Frame captions use **`{_cosmos_label}`** "
    f"(Hugging Face: `{_cosmos_id}`). Set `COSMOS_MODEL` in `.env` to override."
)

_apply_login_bypass()

if not st.session_state.logged_in:
    credentials = get_credentials()
    if _login_required() and credentials:
        with st.form("login"):
            st.subheader("Login")
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted and u and p and credentials.get(u) == p:
                st.session_state.logged_in = True
                st.session_state.username = u
                st.rerun()
            elif submitted:
                st.error("Invalid username or password.")
        st.stop()
    elif _login_required() and not credentials:
        st.warning(
            "You set **REQUIRE_LOGIN=1** but there are no credentials. "
            "Add **LOGIN_USERNAME** and **LOGIN_PASSWORD** to `.env` (next to `app.py`), "
            "or remove **REQUIRE_LOGIN** for open local access."
        )
        st.stop()

if st.session_state.logged_in:
    _creds = get_credentials()
    if _login_required() and _creds:
        st.sidebar.caption(f"Logged in as **{st.session_state.username}**")
        if st.sidebar.button("Log out"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()
    else:
        st.sidebar.caption(
            f"Local mode (**{st.session_state.username}**) — set **REQUIRE_LOGIN=1** and `LOGIN_*` in `.env` to require a password."
        )
    st.sidebar.divider()
st.sidebar.header("Configuration")
frame_interval = st.sidebar.slider(
    "Frame Sampling Interval (seconds)",
    min_value=1,
    max_value=10,
    value=2,
    help="Extract one frame every N seconds"
)

max_frames = st.sidebar.slider(
    "Maximum Frames to Analyze",
    min_value=5,
    max_value=50,
    value=20,
    help="Limit total frames to avoid overwhelming the model"
)

# Only Municipal report with Ollama is supported in this branch
summary_style_label = "Municipal report (detailed)"
style_key = "municipal_report"
summary_engine = "Ollama (local LLM)"
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")

st.sidebar.info(f"✅ **Ollama Summarizer Active**\n\nModel: `{ollama_model}`\nStyle: `{summary_style_label}`")

# MOCK_COSMOS is disabled for this production branch
_mock_cosmos = False

st.sidebar.divider()
st.sidebar.subheader("Search similar videos")
search_query = st.sidebar.text_input(
    "Search by keywords",
    placeholder="e.g. pedestrian crossing",
    help="Semantic search over saved summaries (sidebar). Results show below.",
    key="sidebar_search_query",
)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Supported formats: MP4, AVI, MOV, MKV"
    )
    
    if uploaded_file is not None:
        # Display video
        st.video(uploaded_file)
        
        # Process button
        if st.button("🚀 Generate Summary", type="primary"):
            with st.spinner("Processing video..."):
                try:
                    raw_bytes = uploaded_file.read()
                    saved_local = persist_uploaded_video(
                        raw_bytes, getattr(uploaded_file, "name", None)
                    )
                    if saved_local is not None:
                        st.caption(f"Saved local copy: `{saved_local}`")

                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(raw_bytes)
                        video_path = tmp_file.name

                    # Compute duration for DB (some schemas may require NOT NULL).
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    duration_sec = (total_frames / fps) if fps and fps > 0 else None
                    cap.release()
                    
                    # Step 1: Extract frames
                    st.info("Step 1/3: Extracting frames from video...")
                    processor = VideoProcessor()
                    frames, timestamps = processor.extract_frames(
                        video_path,
                        interval_seconds=frame_interval,
                        max_frames=max_frames
                    )
                    st.session_state.frames = frames
                    st.success(f"✓ Extracted {len(frames)} frames")
                    
                    st.info("Step 2/3: Analyzing frames with Cosmos AI...")
                    # Lazy import: avoids loading transformers/sklearn at app startup (fixes Streamlit+Py3.13 issues).
                    from model_handler import CosmosModelHandler

                    model_handler = CosmosModelHandler()
                    frame_descriptions = model_handler.analyze_frames(frames)
                    st.success(f"✓ Analyzed {len(frame_descriptions)} frames")
                    
                    # Step 3: Generate summary with Ollama
                    st.info("Step 3/3: Generating video summary with Ollama...")
                    summary = summarize_frames_with_ollama(
                        frame_descriptions,
                        timestamps,
                        style=style_key,
                        model=ollama_model.strip() or None,
                        host=os.getenv("OLLAMA_HOST") or None,
                        vision_model=_cosmos_label,
                    )
                    _engine = "ollama"
                    st.session_state.summary = summary
                    _search_text = build_search_text(summary, frame_descriptions)
                    st.session_state.search_hints = suggest_search_terms(frame_descriptions)

                    try:
                        _local_path = append_local_summary(
                            summary_text=summary,
                            filename=getattr(uploaded_file, "name", None),
                            duration_sec=duration_sec,
                            style=style_key,
                            engine=_engine,
                            vision_model=_cosmos_label,
                            search_text=_search_text,
                        )
                        if _local_path is not None:
                            st.caption(f"Stored summary locally: `{_local_path}`")
                    except Exception as _e:
                        st.caption(f"Local summary file not written: {_e}")
                    
                    # Persist summary + embedding to the vector DB
                    # (Streamlit runs on the server, so we can call Python directly.)
                    st.info("Embedding summary + frame captions for search (objects/scenes)…")
                    try:
                        embedding = embed_text(_search_text)
                        insert_summary(
                            filename=getattr(uploaded_file, "name", None),
                            duration_sec=duration_sec,
                            summary_style=style_key,
                            summary_text=summary,
                            embedding=embedding,
                        )
                        st.success("✓ Saved summary to database")
                    except Exception as e:
                        # Summary is still usable even if DB insert fails.
                        st.warning(f"Saved summary, but database insert failed: {e}")
                    st.session_state.processed = True
                    st.success("✓ Summary generated successfully!")
                    
                    # Clean up
                    os.unlink(video_path)
                    
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    st.exception(e)

with col2:
    st.header("Summary Results")
    
    if st.session_state.processed and st.session_state.summary:
        # Display summary
        st.markdown("### 📝 Video Summary")
        st.markdown(st.session_state.summary)
        
        # Download button
        st.download_button(
            label="📥 Download Summary",
            data=st.session_state.summary,
            file_name="video_summary.txt",
            mime="text/plain"
        )
        
        # Show sample frames
        if st.session_state.frames and len(st.session_state.frames) > 0:
            with st.expander("🖼️ View Sample Frames"):
                # Display first few frames
                num_to_show = min(6, len(st.session_state.frames))
                cols = st.columns(3)
                for idx in range(num_to_show):
                    with cols[idx % 3]:
                        st.image(
                            st.session_state.frames[idx],
                            caption=f"Frame {idx + 1}",
                            use_container_width=True
                        )
    else:
        st.info("Upload a video and click 'Generate Summary' to see results here.")

st.divider()
st.subheader("Search results")
if search_query and search_query.strip():
    st.caption(
        "Uses database-backed embeddings over **summary + every frame caption**."
    )
    if st.session_state.search_hints:
        with st.expander("Suggested terms from last run"):
            st.write(", ".join(st.session_state.search_hints))
    q = search_query.strip()
    with st.spinner("Searching..."):
        results: list = []
        try:
            query_embedding = embed_text(q)
            results = search_similar(query_embedding, limit=10)
        except Exception as e:
            st.warning(f"Database search failed: {e}")

    if results:
        for r in results:
            filename = r.get("filename") or "Unknown file"
            summary_text = r.get("summary_text") or ""
            distance = r.get("distance")
            sim = r.get("similarity")
            st.markdown(f"**{filename}**")
            if distance is not None:
                st.caption(f"Distance: {distance:.4f} (lower = closer)")
            elif sim is not None:
                st.caption(f"Similarity: {sim:.4f}")
            st.write(summary_text)
            st.divider()
    else:
        st.info(
            "No matches found yet. Generate a summary first, or try a different phrase."
        )
else:
    if os.getenv("SUPABASE_DB_URL"):
        st.caption("Use the search box in the sidebar to find similar saved summaries.")
    else:
        st.caption("Set `SUPABASE_DB_URL` in `.env` to enable database search.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    f"Powered by Nvidia {_cosmos_label} | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
