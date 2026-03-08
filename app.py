import streamlit as st
import tempfile
import os
from pathlib import Path
from video_processor import VideoProcessor
from model_handler import CosmosModelHandler
from summarizer import VideoSummarizer

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None

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

# Title and description
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
# ... rest of your sidebar (sliders, selectbox)   

# Sidebar for configuration
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

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Powered by Nvidia Cosmos-reason2-8b | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
