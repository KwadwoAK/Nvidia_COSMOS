import os
import tempfile
from typing import Optional

import cv2
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

from auth import render_user_sidebar, require_login
from db.search_video import search_similar_by_text
from db.supabase_storage import (
    build_object_key,
    content_type_for_filename,
    is_storage_configured,
    try_public_video_url,
    upload_local_file_to_video_bucket,
)
from db.video_store import insert_summary
from embeddings.embedder import embed_text
from model_handler import CosmosModelHandler
from summarys.gemma_summarizer import summarize_frames_with_gemma
from summarys.summary_templates import (
    DEFAULT_VISION_MODEL_LABEL,
    parse_template_id_from_summary,
    style_key_from_label,
)
from video_processor import VideoProcessor
from vision_search import build_search_text

THEME_OPTIONS = ["Light Mode", "Dark Mode"]
_LEGACY_THEME_MAP = {
    "light": "Light Mode",
    "light mode": "Light Mode",
    "night": "Dark Mode",
    "dark": "Dark Mode",
    "dark mode": "Dark Mode",
    "navy": "Dark Mode",
    "navy blue": "Dark Mode",
}


def _normalize_theme(value: Optional[str]) -> str:
    """Map legacy theme names and free-form input to a supported option."""
    if not value:
        return "Dark Mode"
    if value in THEME_OPTIONS:
        return value
    return _LEGACY_THEME_MAP.get(value.strip().lower(), "Dark Mode")


_THEME_PALETTES = {
    "Light Mode": {
        "background": "#f4f6fb",
        "surface": "#ffffff",
        "surface_alt": "#eef3fb",
        "text": "#121a2b",
        "muted": "#5b6474",
        "border": "#d7dfed",
        "accent": "#2563eb",
        "accent_soft": "#93c5fd",
        "accent_text": "#ffffff",
        "highlight": "#7c3aed",
        "shadow": "rgba(15, 23, 42, 0.10)",
        "glow": "rgba(37, 99, 235, 0.18)",
        "bg_gradient": "radial-gradient(1200px 600px at 85% -10%, rgba(124,58,237,0.10), transparent 60%), radial-gradient(900px 500px at -10% 20%, rgba(37,99,235,0.12), transparent 55%), #f4f6fb",
    },
    "Dark Mode": {
        "background": "#0b1020",
        "surface": "#121a2e",
        "surface_alt": "#182342",
        "text": "#e8edfb",
        "muted": "#9aa6c2",
        "border": "#233158",
        "accent": "#60a5fa",
        "accent_soft": "#3b82f6",
        "accent_text": "#08111f",
        "highlight": "#a78bfa",
        "shadow": "rgba(2, 6, 23, 0.55)",
        "glow": "rgba(96, 165, 250, 0.25)",
        "bg_gradient": "radial-gradient(1100px 600px at 90% -5%, rgba(167,139,250,0.18), transparent 60%), radial-gradient(900px 500px at -10% 10%, rgba(96,165,250,0.18), transparent 55%), #0b1020",
    },
}


def _get_palette(theme_mode: str) -> dict:
    return _THEME_PALETTES.get(_normalize_theme(theme_mode), _THEME_PALETTES["Dark Mode"])


def _inject_cursor_glow(colors: dict) -> None:
    """Attach a soft cursor-following glow to the parent Streamlit document."""
    glow_color = colors["glow"]
    accent = colors["accent"]
    components.html(
        f"""
        <script>
        (function() {{
          try {{
            const doc = window.parent.document;
            if (!doc) return;
            const existing = doc.getElementById('cursor-glow');
            if (existing) existing.remove();

            const glow = doc.createElement('div');
            glow.id = 'cursor-glow';
            glow.style.cssText = [
              'position:fixed',
              'top:0',
              'left:0',
              'width:320px',
              'height:320px',
              'border-radius:50%',
              'pointer-events:none',
              'z-index:1',
              'transform:translate(-50%,-50%)',
              'background:radial-gradient(circle, {glow_color} 0%, rgba(0,0,0,0) 65%)',
              'mix-blend-mode:screen',
              'filter:blur(10px)',
              'opacity:0',
              'transition:opacity 300ms ease-out, transform 120ms ease-out'
            ].join(';');
            doc.body.appendChild(glow);

            const dot = doc.createElement('div');
            dot.id = 'cursor-dot';
            dot.style.cssText = [
              'position:fixed',
              'top:0',
              'left:0',
              'width:6px',
              'height:6px',
              'border-radius:50%',
              'pointer-events:none',
              'z-index:2',
              'transform:translate(-50%,-50%)',
              'background:{accent}',
              'box-shadow:0 0 8px {accent}',
              'opacity:0',
              'transition:opacity 200ms ease-out, transform 80ms ease-out'
            ].join(';');
            doc.body.appendChild(dot);

            let mouseX = 0, mouseY = 0;
            let glowX = 0, glowY = 0;
            function onMove(e) {{
              mouseX = e.clientX;
              mouseY = e.clientY;
              glow.style.opacity = 0.5;
              dot.style.opacity = 0.55;
              dot.style.left = mouseX + 'px';
              dot.style.top = mouseY + 'px';
            }}
            function onLeave() {{
              glow.style.opacity = 0;
              dot.style.opacity = 0;
            }}
            function tick() {{
              glowX += (mouseX - glowX) * 0.12;
              glowY += (mouseY - glowY) * 0.12;
              glow.style.left = glowX + 'px';
              glow.style.top = glowY + 'px';
              window.parent.requestAnimationFrame(tick);
            }}
            doc.addEventListener('mousemove', onMove);
            doc.addEventListener('mouseleave', onLeave);
            window.parent.requestAnimationFrame(tick);
          }} catch (err) {{
            /* silent */
          }}
        }})();
        </script>
        """,
        height=0,
    )


def _apply_theme(theme_mode: str) -> None:
    """Apply the selected UI skin with animations and a cursor glow."""
    colors = _get_palette(theme_mode)

    st.markdown(
        f"""
        <style>
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translate3d(0, 12px, 0); }}
            to   {{ opacity: 1; transform: translate3d(0, 0, 0); }}
        }}
        @keyframes floatShimmer {{
            0%   {{ background-position: 0% 50%; }}
            50%  {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        @keyframes pulseGlow {{
            0%   {{ box-shadow: 0 0 0 0 {colors["glow"]}; }}
            70%  {{ box-shadow: 0 0 0 16px rgba(0,0,0,0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(0,0,0,0); }}
        }}
        :root {{
            --accent: {colors["accent"]};
            --accent-soft: {colors["accent_soft"]};
            --highlight: {colors["highlight"]};
            --surface: {colors["surface"]};
            --surface-alt: {colors["surface_alt"]};
            --border: {colors["border"]};
            --text: {colors["text"]};
            --muted: {colors["muted"]};
            --shadow: {colors["shadow"]};
            --glow: {colors["glow"]};
        }}
        * {{
            transition: background-color 220ms ease, color 220ms ease,
                        border-color 220ms ease, box-shadow 220ms ease,
                        transform 220ms ease, filter 220ms ease;
        }}
        .block-container {{
            max-width: 1200px;
            padding-top: 2.2rem;
            padding-bottom: 2.5rem;
            animation: fadeInUp 520ms ease-out;
        }}
        .stApp, [data-testid="stAppViewContainer"] {{
            background: {colors["bg_gradient"]};
            color: {colors["text"]};
        }}
        [data-testid="stHeader"] {{
            background: transparent;
            backdrop-filter: blur(6px);
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {colors["surface"]} 0%, {colors["surface_alt"]} 100%);
            border-right: 1px solid {colors["border"]};
        }}
        [data-testid="stSidebarUserContent"] {{
            padding-top: 1rem;
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
        h1, h2, h3 {{
            letter-spacing: -0.02em;
        }}
        .stMarkdown, .stCaption {{
            color: {colors["text"]};
        }}
        .stAlert {{
            background: {colors["surface_alt"]};
            color: {colors["text"]};
            border: 1px solid {colors["border"]};
            border-radius: 16px;
            animation: fadeInUp 420ms ease-out;
        }}
        .stExpander, .stTextInput > div > div, .stSelectbox > div > div,
        .stNumberInput > div > div, .stSlider, .stFileUploader,
        .stTextArea textarea {{
            background: {colors["surface"]};
            color: {colors["text"]};
        }}
        .stTextInput > div > div, .stSelectbox > div > div {{
            border-radius: 12px;
        }}
        .stTextInput > div > div:focus-within,
        .stSelectbox > div > div:focus-within {{
            box-shadow: 0 0 0 3px {colors["glow"]};
            border-color: {colors["accent"]} !important;
        }}
        [data-testid="stVerticalBlockBorderWrapper"] {{
            border-radius: 20px;
            background: {colors["surface"]};
            border: 1px solid {colors["border"]};
            box-shadow: 0 14px 36px {colors["shadow"]};
            animation: fadeInUp 520ms ease-out both;
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 22px 48px {colors["shadow"]};
            border-color: {colors["accent_soft"]};
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background: {colors["surface"]};
            border: 1.5px dashed {colors["accent_soft"]};
            border-radius: 18px;
            padding: 1.1rem;
        }}
        [data-testid="stFileUploaderDropzone"] * {{
            color: {colors["text"]} !important;
        }}
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzoneInstructions"] small,
        [data-testid="stFileUploaderDropzoneInstructions"] span {{
            color: {colors["muted"]} !important;
        }}
        [data-testid="stFileUploaderDropzone"] svg {{
            fill: {colors["accent"]} !important;
            color: {colors["accent"]} !important;
        }}
        [data-testid="stFileUploaderDropzone"] button {{
            background: {colors["accent"]} !important;
            color: {colors["accent_text"]} !important;
            border: 1px solid {colors["accent"]} !important;
            border-radius: 999px !important;
            font-weight: 600 !important;
            padding: 0.4rem 1rem !important;
            box-shadow: 0 6px 14px {colors["shadow"]} !important;
        }}
        [data-testid="stFileUploaderDropzone"] button:hover {{
            filter: brightness(1.08) !important;
            transform: translateY(-1px);
        }}
        [data-testid="stFileUploaderDropzone"]:hover {{
            border-color: {colors["accent"]};
            box-shadow: 0 0 0 4px {colors["glow"]};
        }}
        .stButton > button, .stDownloadButton > button {{
            background: linear-gradient(135deg, {colors["accent"]} 0%, {colors["highlight"]} 100%);
            color: {colors["accent_text"]};
            border: 1px solid {colors["accent"]};
            border-radius: 999px;
            font-weight: 600;
            min-height: 2.9rem;
            padding: 0.55rem 1.15rem;
            box-shadow: 0 10px 24px {colors["shadow"]};
            background-size: 200% 200%;
            background-position: 0% 50%;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            transform: translateY(-2px) scale(1.01);
            filter: brightness(1.08);
            background-position: 100% 50%;
            animation: pulseGlow 1.2s ease-out 1;
        }}
        .stButton > button:active, .stDownloadButton > button:active {{
            transform: translateY(0) scale(0.99);
        }}
        .stButton > button {{
            width: 100%;
        }}
        [data-testid="stSlider"] [role="slider"] {{
            box-shadow: 0 0 0 4px {colors["glow"]};
        }}
        code {{
            color: {colors["text"]};
            background: {colors["surface_alt"]};
        }}
        small, [data-testid="stCaptionContainer"] {{
            color: {colors["muted"]};
        }}
        .hero-panel {{
            position: relative;
            padding: 1.9rem 2rem;
            border: 1px solid {colors["border"]};
            border-radius: 26px;
            background:
                linear-gradient(135deg, {colors["surface"]} 0%, {colors["surface_alt"]} 100%);
            box-shadow: 0 20px 44px {colors["shadow"]};
            margin-bottom: 1.1rem;
            overflow: hidden;
            animation: fadeInUp 600ms ease-out;
        }}
        .hero-panel::before {{
            content: "";
            position: absolute;
            inset: -40%;
            background: radial-gradient(ellipse at top right,
                {colors["glow"]} 0%, transparent 60%);
            pointer-events: none;
            opacity: 0.9;
        }}
        .hero-eyebrow, .section-eyebrow {{
            position: relative;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.74rem;
            font-weight: 700;
            color: {colors["accent"]};
            margin-bottom: 0.55rem;
        }}
        .hero-title {{
            position: relative;
            font-size: 2.45rem;
            font-weight: 800;
            line-height: 1.08;
            margin: 0;
            background: linear-gradient(90deg, {colors["text"]} 0%, {colors["accent"]} 60%, {colors["highlight"]} 100%);
            background-size: 200% auto;
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            animation: floatShimmer 10s ease-in-out infinite;
        }}
        .hero-copy {{
            position: relative;
            max-width: 760px;
            margin: 0.9rem 0 0;
            color: {colors["muted"]};
            font-size: 1.02rem;
            line-height: 1.65;
        }}
        .metric-card {{
            position: relative;
            border: 1px solid {colors["border"]};
            border-radius: 18px;
            background: {colors["surface"]};
            padding: 1rem 1.05rem;
            min-height: 130px;
            box-shadow: 0 12px 28px {colors["shadow"]};
            margin-bottom: 0.6rem;
            overflow: hidden;
            animation: fadeInUp 520ms ease-out both;
        }}
        .metric-card::after {{
            content: "";
            position: absolute;
            left: 0; top: 0;
            height: 3px;
            width: 100%;
            background: linear-gradient(90deg, {colors["accent"]}, {colors["highlight"]});
            opacity: 0.85;
        }}
        .metric-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 18px 40px {colors["shadow"]};
            border-color: {colors["accent_soft"]};
        }}
        .metric-card-label {{
            color: {colors["muted"]};
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 700;
        }}
        .metric-card-value {{
            font-size: 1.25rem;
            font-weight: 700;
            margin-top: 0.55rem;
        }}
        .metric-card-caption {{
            color: {colors["muted"]};
            font-size: 0.92rem;
            line-height: 1.55;
            margin-top: 0.45rem;
        }}
        .section-intro {{
            color: {colors["muted"]};
            margin-top: -0.3rem;
            margin-bottom: 1rem;
            line-height: 1.55;
        }}
        .empty-state {{
            border: 1px dashed {colors["border"]};
            background: {colors["surface_alt"]};
            border-radius: 18px;
            padding: 1.3rem 1.25rem;
            color: {colors["muted"]};
            line-height: 1.6;
            animation: fadeInUp 500ms ease-out;
        }}
        .search-result-title {{
            font-size: 1.06rem;
            font-weight: 700;
            color: {colors["accent"]};
        }}
        .search-result-meta {{
            color: {colors["muted"]};
            font-size: 0.92rem;
            margin-top: 0.25rem;
        }}
        .app-footer {{
            text-align: center;
            color: {colors["muted"]};
            font-size: 0.92rem;
            padding-top: 0.8rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    _inject_cursor_glow(colors)


def _format_filesize(num_bytes: Optional[int]) -> str:
    """Return a human-readable file size."""
    if not num_bytes:
        return "Unavailable"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _format_duration(seconds: Optional[float]) -> str:
    """Return a readable minutes:seconds string."""
    if not seconds or seconds <= 0:
        return "Not available"
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def _render_metric_card(label: str, value: str, caption: str) -> None:
    """Render a compact presentation card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-label">{label}</div>
            <div class="metric-card-value">{value}</div>
            <div class="metric-card-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Upload & summarize",
    page_icon="🎥",
    layout="wide",
)

if "processed" not in st.session_state:
    st.session_state.processed = False
if "summary" not in st.session_state:
    st.session_state.summary = None
if "frames" not in st.session_state:
    st.session_state.frames = None
if "analysis_stats" not in st.session_state:
    st.session_state.analysis_stats = None
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = _normalize_theme(os.getenv("UI_THEME"))
else:
    st.session_state.theme_mode = _normalize_theme(st.session_state.theme_mode)

require_login()
render_user_sidebar()

st.sidebar.divider()
st.sidebar.header("Appearance")
current_theme = _normalize_theme(st.session_state.theme_mode)
theme_mode = st.sidebar.radio(
    "Theme",
    THEME_OPTIONS,
    index=THEME_OPTIONS.index(current_theme),
    help="Choose between Light Mode and Dark Mode.",
)
st.session_state.theme_mode = theme_mode
_apply_theme(theme_mode)
st.sidebar.divider()

st.markdown(
    """
    <div class="hero-panel">
        <div class="hero-eyebrow">Red Light Camera Sight Selection</div>
        <div class="hero-title">Video Analysis and Summary Workspace</div>
        <div class="hero-copy">
            Submit recorded footage, generate structured AI-assisted summaries, and review archived outputs in a
            single professional interface designed for presentation and evaluation.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

overview_cols = st.columns(3)
with overview_cols[0]:
    _render_metric_card(
        "Primary model",
        "Cosmos Reason2 8B",
        "Multimodal frame analysis is performed before summary generation.",
    )
with overview_cols[1]:
    _render_metric_card(
        "Workflow",
        "3-stage pipeline",
        "Frame extraction, visual interpretation, and structured summary synthesis.",
    )
with overview_cols[2]:
    _render_metric_card(
        "Archive search",
        "Semantic retrieval",
        "Saved summaries can be searched from the sidebar to support review and comparison.",
    )

st.sidebar.header("Analysis Settings")
frame_interval = st.sidebar.slider(
    "Frame sampling interval (seconds)",
    min_value=1,
    max_value=10,
    value=2,
    help="Extract one frame at the selected interval.",
)

max_frames = st.sidebar.slider(
    "Maximum frames to analyze",
    min_value=5,
    max_value=50,
    value=20,
    help="Cap the number of frames to keep processing time predictable.",
)

summary_style = st.sidebar.selectbox(
    "Summary style",
    ["Detailed", "Concise", "Bullet Points"],
    help="Select the output structure used for the written summary.",
)

st.sidebar.divider()
st.sidebar.subheader("Archive Search")
search_query = st.sidebar.text_input(
    "Search saved summaries",
    placeholder="Example: pedestrian crossing",
    help="Run semantic search across previously saved summaries.",
    key="sidebar_search_query",
)
st.sidebar.caption("Other pages: open **Semantic search** below.")

col1, col2 = st.columns([1, 1])

with col1:
    with st.container(border=True):
        st.markdown('<div class="section-eyebrow">Input</div>', unsafe_allow_html=True)
        st.header("Video Submission")
        st.markdown(
            '<div class="section-intro">Upload a source video to begin frame extraction, visual analysis, and summary generation.</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Select a video file",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supported formats: MP4, AVI, MOV, and MKV.",
        )

        if uploaded_file is not None:
            file_cols = st.columns(3)
            with file_cols[0]:
                _render_metric_card("File name", uploaded_file.name, "Current item prepared for analysis.")
            with file_cols[1]:
                _render_metric_card("File size", _format_filesize(getattr(uploaded_file, "size", None)), "Estimated upload size.")
            with file_cols[2]:
                _render_metric_card("Output style", summary_style, "Current summary formatting selection.")

            st.video(uploaded_file)

            if st.button("Generate Summary", type="primary"):
                with st.spinner("Processing video and generating analysis..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            video_path = tmp_file.name

                        cap = cv2.VideoCapture(video_path)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        duration_sec = (total_frames / fps) if fps and fps > 0 else None
                        cap.release()

                        st.info("Step 1 of 3: Extracting representative frames from the submitted video.")
                        processor = VideoProcessor()
                        frames, timestamps = processor.extract_frames(
                            video_path,
                            interval_seconds=frame_interval,
                            max_frames=max_frames,
                        )
                        st.session_state.frames = frames
                        st.success(f"Frame extraction completed: {len(frames)} frames prepared for analysis.")

                        st.info("Step 2 of 3: Interpreting the extracted frames with the vision-language model.")
                        model_handler = CosmosModelHandler()
                        frame_descriptions = model_handler.analyze_frames(frames)
                        st.success(f"Visual analysis completed: {len(frame_descriptions)} frame descriptions generated.")

                        st.info("Step 3 of 3: Synthesizing the final written summary.")
                        style_key = style_key_from_label(summary_style)
                        summary = summarize_frames_with_gemma(
                            frame_descriptions,
                            timestamps,
                            style=style_key,
                        )
                        st.session_state.summary = summary
                        st.session_state.analysis_stats = {
                            "filename": getattr(uploaded_file, "name", "Current video"),
                            "duration_sec": duration_sec,
                            "frame_count": len(frame_descriptions),
                            "summary_style": summary_style,
                        }

                        st.info("Uploading video file (if configured) and saving to database...")
                        storage_object_path: Optional[str] = None
                        if is_storage_configured():
                            try:
                                key = build_object_key(
                                    str(st.session_state.username or "user"),
                                    getattr(uploaded_file, "name", None),
                                )
                                upload_local_file_to_video_bucket(
                                    video_path,
                                    key,
                                    content_type=content_type_for_filename(getattr(uploaded_file, "name", None)),
                                )
                                storage_object_path = key
                                st.success("Video stored in Supabase Storage.")
                            except Exception as upload_exc:
                                st.warning(f"Could not upload video to Storage: {upload_exc}")
                        else:
                            st.info(
                                "Storage upload skipped: set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
                                "to save the video file to your bucket."
                            )

                        try:
                            search_text = build_search_text(summary, frame_descriptions)
                            embedding = embed_text(search_text)
                            insert_summary(
                                filename=getattr(uploaded_file, "name", None),
                                duration_sec=duration_sec,
                                summary_style=style_key,
                                summary_text=summary,
                                embedding=embedding,
                                summary_engine="gemma4",
                                vision_model=os.getenv("COSMOS_MODEL_LABEL", DEFAULT_VISION_MODEL_LABEL),
                                template_id=parse_template_id_from_summary(summary),
                                search_text=search_text,
                                storage_object_path=storage_object_path,
                            )
                            st.success("Archive update completed successfully.")
                        except Exception as e:
                            st.warning(f"The summary was generated, but archive storage failed: {e}")
                        st.session_state.processed = True
                        st.success("Analysis completed successfully.")

                        os.unlink(video_path)

                    except Exception as e:
                        st.error(f"An error occurred while processing the video: {str(e)}")
                        st.exception(e)
        else:
            st.markdown(
                """
                <div class="empty-state">
                    No source video has been selected yet. Once a file is uploaded, a preview and submission controls
                    will appear in this panel.
                </div>
                """,
                unsafe_allow_html=True,
            )

with col2:
    with st.container(border=True):
        st.markdown('<div class="section-eyebrow">Output</div>', unsafe_allow_html=True)
        st.header("Analysis Results")
        st.markdown(
            '<div class="section-intro">Review the generated summary, download the report, and inspect representative frames from the analysis pass.</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.processed and st.session_state.summary:
            stats = st.session_state.analysis_stats or {}
            result_cols = st.columns(3)
            with result_cols[0]:
                _render_metric_card(
                    "Reviewed asset",
                    stats.get("filename", "Current submission"),
                    "The most recently processed video in this session.",
                )
            with result_cols[1]:
                _render_metric_card(
                    "Estimated duration",
                    _format_duration(stats.get("duration_sec")),
                    "Computed from the uploaded file when timing data is available.",
                )
            with result_cols[2]:
                _render_metric_card(
                    "Frames analyzed",
                    str(stats.get("frame_count", 0)),
                    f"Summary style: {stats.get('summary_style', 'Unknown')}.",
                )

            st.subheader("Generated Summary")
            st.markdown(st.session_state.summary)

            st.download_button(
                label="Download Summary (.txt)",
                data=st.session_state.summary,
                file_name="video_summary.txt",
                mime="text/plain",
            )

            if st.session_state.frames and len(st.session_state.frames) > 0:
                with st.expander("Review Sample Frames"):
                    num_to_show = min(6, len(st.session_state.frames))
                    cols = st.columns(3)
                    for idx in range(num_to_show):
                        with cols[idx % 3]:
                            st.image(
                                st.session_state.frames[idx],
                                caption=f"Frame {idx + 1}",
                                use_container_width=True,
                            )
        else:
            st.markdown(
                """
                <div class="empty-state">
                    No analysis has been generated yet. After you submit a video, the completed summary, downloadable
                    report, and sample frames will appear here.
                </div>
                """,
                unsafe_allow_html=True,
            )

st.divider()
st.markdown('<div class="section-eyebrow">Archive</div>', unsafe_allow_html=True)
st.subheader("Saved Summary Search Results")
if search_query and search_query.strip():
    st.markdown(
        '<div class="section-intro">Results below are retrieved semantically from previously saved summaries.</div>',
        unsafe_allow_html=True,
    )
    search_failed = False
    try:
        with st.spinner("Searching archived summaries..."):
            results = search_similar_by_text(search_query.strip(), limit=10)
    except Exception as e:
        search_failed = True
        results = []
        st.error(f"Search failed (check SUPABASE_DB_URL and pgvector): {e}")

    if results:
        for r in results:
            filename = r.get("filename") or "Unknown file"
            summary_text = r.get("summary_text") or ""
            distance = r.get("distance")
            storage_key = r.get("storage_object_path")
            with st.container(border=True):
                st.markdown(f'<div class="search-result-title">{filename}</div>', unsafe_allow_html=True)
                meta_text = "Similarity score unavailable"
                if distance is not None:
                    meta_text = f"Vector distance: {distance:.4f}"
                st.markdown(f'<div class="search-result-meta">{meta_text}</div>', unsafe_allow_html=True)
                video_url = try_public_video_url(storage_key) if storage_key else None
                if video_url:
                    st.video(video_url)
                st.write(summary_text)
    elif not search_failed:
        st.markdown(
            """
            <div class="empty-state">
                No related summaries were found for the current search. Try broader keywords or add more archived
                summaries to improve retrieval coverage.
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        """
        <div class="empty-state">
            Use the archive search field in the sidebar to retrieve previously saved summaries and compare them with
            the current submission.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown(
    "<div class='app-footer'>Trinity College Senior Project 2026 - Osarfo-Akoto, Sanchez, Carpe-Elias </div>",
    unsafe_allow_html=True,
)
