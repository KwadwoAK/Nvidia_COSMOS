import streamlit as st
import tempfile
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import cv2
from video_processor import VideoProcessor
from model_handler import CosmosModelHandler
from summarizer import VideoSummarizer
from embeddings.embedder import embed_text
from db.video_store import insert_summary
from db.search_video import search_similar

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = os.getenv("UI_THEME", "Night")

def get_credentials():
    """Username -> password. From env (single user) or Streamlit secrets."""
    try:
        if "passwords" in st.secrets:
            return st.secrets["passwords"]
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


def _apply_theme(theme_mode: str) -> None:
    """Apply a simple Light/Night UI skin without requiring Streamlit config files."""
    theme = (theme_mode or "Night").strip().lower()
    if theme == "light":
        colors = {
            "background": "#f6f8fc",
            "surface": "#ffffff",
            "surface_alt": "#eef3fb",
            "text": "#162033",
            "muted": "#5b6474",
            "border": "#d7dfed",
            "accent": "#2563eb",
            "accent_text": "#ffffff",
        }
    else:
        colors = {
            "background": "#0f172a",
            "surface": "#111827",
            "surface_alt": "#1f2937",
            "text": "#e5eefb",
            "muted": "#a7b4c8",
            "border": "#334155",
            "accent": "#60a5fa",
            "accent_text": "#08111f",
        }

    st.markdown(
        f"""
        <style>
        .stApp, [data-testid="stAppViewContainer"] {{
            background: {colors["background"]};
            color: {colors["text"]};
        }}
        [data-testid="stHeader"] {{
            background: {colors["background"]};
        }}
        [data-testid="stSidebar"] {{
            background: {colors["surface"]};
            border-right: 1px solid {colors["border"]};
        }}
        [data-testid="stSidebar"] * {{
            color: {colors["text"]};
        }}
        [data-testid="stAppViewContainer"] * {{
            border-color: {colors["border"]};
        }}
        h1, h2, h3, h4, h5, h6, p, li, label, span, div {{
            color: {colors["text"]};
        }}
        .stMarkdown, .stCaption {{
            color: {colors["text"]};
        }}
        .stAlert {{
            background: {colors["surface_alt"]};
            color: {colors["text"]};
            border: 1px solid {colors["border"]};
        }}
        .stExpander, .stTextInput > div > div, .stSelectbox > div > div,
        .stNumberInput > div > div, .stSlider, .stFileUploader,
        .stTextArea textarea {{
            background: {colors["surface"]};
            color: {colors["text"]};
        }}
        .stButton > button, .stDownloadButton > button {{
            background: {colors["accent"]};
            color: {colors["accent_text"]};
            border: 1px solid {colors["accent"]};
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            filter: brightness(1.05);
        }}
        code {{
            color: {colors["text"]};
            background: {colors["surface_alt"]};
        }}
        small, [data-testid="stCaptionContainer"] {{
            color: {colors["muted"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
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

_apply_login_bypass()

# Title and description
st.sidebar.header("Appearance")
theme_mode = st.sidebar.selectbox(
    "Theme",
    ["Night", "Light"],
    index=0 if st.session_state.theme_mode == "Night" else 1,
    help="Switch the UI between dark and light colors.",
)
st.session_state.theme_mode = theme_mode
_apply_theme(theme_mode)
st.sidebar.divider()

st.title("🎥 Video Summarizer with Cosmos AI")
st.markdown("Upload a video to get an AI-generated summary using Nvidia's Cosmos-reason2-8b model")

if not st.session_state.logged_in:
    credentials = get_credentials()
    if not credentials:
        st.warning("Set LOGIN_USERNAME and LOGIN_PASSWORD in the environment, or add a 'passwords' dict in Streamlit secrets.")
        st.stop()

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

if st.session_state.logged_in:
    st.sidebar.caption(f"Logged in as **{st.session_state.username}**")
    if st.sidebar.button("Log out"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()
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

summary_style = st.sidebar.selectbox(
    "Summary Style",
    ["Detailed", "Concise", "Bullet Points"],
    help="Choose how you want the summary formatted"
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
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_file.read())
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
                    
                    # Step 2: Analyze with Cosmos model
                    st.info("Step 2/3: Analyzing frames with Cosmos AI...")
                    model_handler = CosmosModelHandler()
                    frame_descriptions = model_handler.analyze_frames(frames)
                    st.success(f"✓ Analyzed {len(frame_descriptions)} frames")
                    
                    # Step 3: Generate summary
                    st.info("Step 3/3: Generating video summary...")
                    summarizer = VideoSummarizer()
                    summary = summarizer.generate_summary(
                        frame_descriptions,
                        timestamps,
                        style=summary_style.lower()
                    )
                    st.session_state.summary = summary
                    
                    # Persist summary + embedding to the vector DB
                    # (Streamlit runs on the server, so we can call Python directly.)
                    st.info("Embedding summary and saving to database...")
                    try:
                        embedding = embed_text(summary)
                        insert_summary(
                            filename=getattr(uploaded_file, "name", None),
                            duration_sec=duration_sec,
                            summary_style=summary_style.lower(),
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
    with st.spinner("Embedding query and searching..."):
        query_embedding = embed_text(search_query.strip())
        results = search_similar(query_embedding, limit=10)

    if results:
        for r in results:
            filename = r.get("filename") or "Unknown file"
            summary_text = r.get("summary_text") or ""
            distance = r.get("distance")
            st.markdown(f"**{filename}**")
            if distance is not None:
                st.caption(f"Distance: {distance:.4f}")
            st.write(summary_text)
            st.divider()
    else:
        st.info("No similar summaries found yet. Generate a summary first.")
else:
    st.caption("Use the search box in the sidebar to find similar saved summaries.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Powered by Nvidia Cosmos-reason2-8b | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
