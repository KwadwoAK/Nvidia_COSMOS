import streamlit as st
import tempfile
import os
from pathlib import Path

from dotenv import load_dotenv

_APP_DIR = Path(__file__).resolve().parent
# Load .env next to this file (works even if Streamlit's cwd is elsewhere)
load_dotenv(_APP_DIR / ".env")
load_dotenv()
# Reduces fork/atexit issues when Streamlit reloads + transformers/sklearn import chain (esp. Python 3.13).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import cv2
from video_processor import VideoProcessor
from mock_vision import mock_analyze_frames
from summarizer import VideoSummarizer
from ollama_summarizer import summarize_frames_with_ollama
from summary_store import append_local_summary
from summary_templates import ANALYSIS_STYLES, DEFAULT_VISION_MODEL_LABEL, style_key_from_label
from vision_search import (
    build_search_text,
    search_local_summaries_semantic,
    suggest_search_terms,
)
from video_storage import persist_uploaded_video
from embeddings.embedder import embed_text
from db.video_store import insert_summary
from db.search_video import search_similar

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

summary_style_label = st.sidebar.selectbox(
    "Analysis type",
    [label for label, _ in ANALYSIS_STYLES],
    help=(
        "Bullet points: scannable lists. Concise: short. Formal: professional memo tone. "
        "Municipal report: long English incident-style record (best with Ollama)."
    ),
)
style_key = style_key_from_label(summary_style_label)

summary_engine = st.sidebar.selectbox(
    "Summary engine",
    ["Heuristic (no LLM)", "Ollama (local LLM)"],
    help="Heuristic stitches captions with rules. Ollama uses your local LLM (see instructions below).",
)
ollama_model = st.sidebar.text_input(
    "Ollama model",
    value=os.getenv("OLLAMA_MODEL", "llama3.2"),
    disabled=summary_engine != "Ollama (local LLM)",
    help="Run: ollama pull <name>",
)
if summary_engine == "Ollama (local LLM)":
    with st.sidebar.expander("How to run Ollama locally"):
        st.markdown(
            """
1. Install [Ollama](https://ollama.com/download) for Windows.  
2. Open a terminal and run: `ollama serve` (or use the tray app — it listens on port **11434**).  
3. Pull a model: `ollama pull llama3.2` (or the name you typed above).  
4. Restart this app if you change models.  

**Municipal report** uses long outputs; ensure your model has enough context (raise `OLLAMA_NUM_PREDICT_MUNICIPAL` in `.env` if output is cut off).
            """
        )

_mock_cosmos = os.getenv("MOCK_COSMOS", "").lower() in ("1", "true", "yes")
if _mock_cosmos:
    st.sidebar.warning(
        "**MOCK_COSMOS** is enabled — Cosmos weights are not loaded. "
        "Captions are placeholders; use only for UI and pipeline tests."
    )

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
                    
                    # Step 2: Analyze with Cosmos model (or mock if HF unreachable)
                    if _mock_cosmos:
                        st.info("Step 2/3: Mock vision (MOCK_COSMOS=1) — skipping Cosmos download…")
                        frame_descriptions = mock_analyze_frames(frames, timestamps)
                        st.success(f"✓ Mock captions for {len(frame_descriptions)} frames (not real vision)")
                    else:
                        st.info("Step 2/3: Analyzing frames with Cosmos AI...")
                        # Lazy import: avoids loading transformers/sklearn at app startup (fixes Streamlit+Py3.13 issues).
                        from model_handler import CosmosModelHandler

                        model_handler = CosmosModelHandler()
                        frame_descriptions = model_handler.analyze_frames(frames)
                        st.success(f"✓ Analyzed {len(frame_descriptions)} frames")
                    
                    # Step 3: Generate summary (heuristic or Ollama)
                    st.info("Step 3/3: Generating video summary...")
                    if summary_engine == "Ollama (local LLM)":
                        summary = summarize_frames_with_ollama(
                            frame_descriptions,
                            timestamps,
                            style=style_key,
                            model=ollama_model.strip() or None,
                            host=os.getenv("OLLAMA_HOST") or None,
                            vision_model=_cosmos_label,
                        )
                        _engine = "ollama"
                    else:
                        summarizer = VideoSummarizer(vision_model=_cosmos_label)
                        summary = summarizer.generate_summary(
                            frame_descriptions,
                            timestamps,
                            style=style_key,
                        )
                        _engine = "heuristic"
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
        "Uses the same text embeddings as summaries, indexed over **summary + every frame caption**."
    )
    if st.session_state.search_hints:
        with st.expander("Suggested terms from last run"):
            st.write(", ".join(st.session_state.search_hints))
    q = search_query.strip()
    with st.spinner("Searching..."):
        results: list = []
        if os.getenv("SUPABASE_DB_URL"):
            try:
                query_embedding = embed_text(q)
                results = search_similar(query_embedding, limit=10)
            except Exception as e:
                st.warning(f"Database search failed: {e}")
        else:
            try:
                results = search_local_summaries_semantic(q, limit=10)
            except Exception as e:
                st.warning(f"Local search failed: {e}")

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
            "No matches found yet. Generate a summary first, or try a different phrase "
            "(search uses captions from Cosmos or mock vision)."
        )
else:
    st.caption("Use the search box in the sidebar to find similar saved summaries.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    f"Powered by Nvidia {_cosmos_label} | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
