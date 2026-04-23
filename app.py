import streamlit as st
from dotenv import load_dotenv

from auth import render_user_sidebar, require_login
from db.supabase_storage import try_public_video_url
from services.archive_search import run_archive_search
from services.pipeline import run_generate_summary_workflow
from state.session import init_session_state
from ui.components import format_duration, format_filesize, render_metric_card
from ui.sidebar import render_sidebar
from ui.theme import apply_theme

load_dotenv()
st.set_page_config(page_title="Upload & summarize", page_icon="🎥", layout="wide")

init_session_state()
require_login()
render_user_sidebar()

sidebar_config = render_sidebar()
st.session_state.theme_mode = sidebar_config.theme_mode
apply_theme(sidebar_config.theme_mode)

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
    render_metric_card("Primary model", "Cosmos-Reason2-8B", "Multimodal frame analysis is performed before summary generation.")
with overview_cols[1]:
    render_metric_card("Workflow", "3-stage pipeline", "Frame extraction, visual interpretation, and structured summary synthesis.")
with overview_cols[2]:
    render_metric_card("Archive search", "Semantic retrieval", "Saved summaries can be searched from the sidebar to support review and comparison.")

col1, col2 = st.columns([1, 1])

with col1:
    with st.container(border=True):
        st.markdown('<div class="section-eyebrow">Input</div>', unsafe_allow_html=True)
        st.header("Video Submission")
        st.markdown('<div class="section-intro">Upload a source video to begin frame extraction, visual analysis, and summary generation.</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Select a video file", type=["mp4", "avi", "mov", "mkv"], help="Supported formats: MP4, AVI, MOV, and MKV.")
        if uploaded_file is not None:
            file_cols = st.columns(3)
            with file_cols[0]:
                render_metric_card("File name", uploaded_file.name, "Current item prepared for analysis.")
            with file_cols[1]:
                render_metric_card("File size", format_filesize(getattr(uploaded_file, "size", None)), "Estimated upload size.")
            with file_cols[2]:
                render_metric_card("Output style", sidebar_config.summary_style, "Current summary formatting selection.")
            st.video(uploaded_file)
            if st.button("Generate Summary", type="primary"):
                with st.spinner("Processing video and generating analysis..."):
                    try:
                        run_generate_summary_workflow(uploaded_file, sidebar_config.frame_interval, sidebar_config.max_frames, sidebar_config.summary_style)
                    except Exception as exc:
                        st.error(f"An error occurred while processing the video: {str(exc)}")
                        st.exception(exc)
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
        st.markdown('<div class="section-intro">Review the generated summary, download the report, and inspect representative frames from the analysis pass.</div>', unsafe_allow_html=True)
        if st.session_state.processed and st.session_state.summary:
            stats = st.session_state.analysis_stats or {}
            result_cols = st.columns(3)
            with result_cols[0]:
                render_metric_card("Reviewed asset", stats.get("filename", "Current submission"), "The most recently processed video in this session.")
            with result_cols[1]:
                render_metric_card("Estimated duration", format_duration(stats.get("duration_sec")), "Computed from the uploaded file when timing data is available.")
            with result_cols[2]:
                render_metric_card("Frames analyzed", str(stats.get("frame_count", 0)), f"Summary style: {stats.get('summary_style', 'Unknown')}.")
            st.subheader("Generated Summary")
            st.markdown(st.session_state.summary)
            st.download_button(label="Download Summary (.txt)", data=st.session_state.summary, file_name="video_summary.txt", mime="text/plain")
            if st.session_state.frames and len(st.session_state.frames) > 0:
                with st.expander("Review Sample Frames"):
                    num_to_show = min(6, len(st.session_state.frames))
                    cols = st.columns(3)
                    for idx in range(num_to_show):
                        with cols[idx % 3]:
                            st.image(st.session_state.frames[idx], caption=f"Frame {idx + 1}", use_container_width=True)
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
search_query = sidebar_config.search_query
if search_query and search_query.strip():
    st.markdown('<div class="section-intro">Results below are retrieved semantically from previously saved summaries.</div>', unsafe_allow_html=True)
    with st.spinner("Searching archived summaries..."):
        results, error_message = run_archive_search(search_query.strip(), limit=10)
    if error_message:
        st.error(f"Search failed (check SUPABASE_DB_URL and pgvector): {error_message}")
    elif results:
        for result in results:
            filename = result.get("filename") or "Unknown file"
            summary_text = result.get("summary_text") or ""
            distance = result.get("distance")
            storage_key = result.get("storage_object_path")
            with st.container(border=True):
                st.markdown(f'<div class="search-result-title">{filename}</div>', unsafe_allow_html=True)
                meta_text = f"Vector distance: {distance:.4f}" if distance is not None else "Similarity score unavailable"
                st.markdown(f'<div class="search-result-meta">{meta_text}</div>', unsafe_allow_html=True)
                video_url = try_public_video_url(storage_key) if storage_key else None
                if video_url:
                    st.video(video_url)
                st.write(summary_text)
    else:
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
st.markdown("<div class='app-footer'>Trinity College Senior Project 2026 - Osarfo-Akoto, Sanchez, Carpe-Elias </div>", unsafe_allow_html=True)
