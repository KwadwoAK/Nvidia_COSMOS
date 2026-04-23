import os
import tempfile
from typing import Any, Optional

import cv2
import streamlit as st

from db.supabase_storage import (
    build_object_key,
    content_type_for_filename,
    is_storage_configured,
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


def run_generate_summary_workflow(
    uploaded_file: Any,
    frame_interval: int,
    max_frames: int,
    summary_style: str,
) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    try:
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

        st.info("Step 2 of 3: Interpreting the extracted frames with the cosmos-reason2-8b model.")
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
        except Exception as exc:
            st.warning(f"The summary was generated, but archive storage failed: {exc}")

        st.session_state.processed = True
        st.success("Analysis completed successfully.")
    finally:
        os.unlink(video_path)
