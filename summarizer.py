import os
from typing import List, Dict

from summary_templates import DEFAULT_VISION_MODEL_LABEL, format_heuristic_summary
from video_processor import VideoProcessor


class VideoSummarizer:
    """Structured summaries from frame descriptions (same template as Ollama path)."""

    def __init__(self, vision_model: str | None = None):
        self.processor = VideoProcessor()
        self.vision_model = vision_model or os.getenv("COSMOS_MODEL_LABEL", DEFAULT_VISION_MODEL_LABEL)

    def generate_summary(
        self,
        frame_descriptions: List[Dict[str, str]],
        timestamps: List[float],
        style: str = "detailed",
    ) -> str:
        return format_heuristic_summary(
            frame_descriptions,
            timestamps,
            style,
            self.processor.format_timestamp,
            vision_model=self.vision_model,
        )

    def extract_key_topics(self, frame_descriptions: List[Dict[str, str]]) -> List[str]:
        """
        Extract main topics/themes from the video
        
        Args:
            frame_descriptions: Frame analysis data
            
        Returns:
            List of key topics
        """
        # This is a simplified version
        # You could use NLP techniques or another LLM call for better topic extraction
        topics = set()
        
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                       'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                       'this', 'that', 'these', 'those'}
        
        for frame_desc in frame_descriptions:
            words = frame_desc['description'].lower().split()
            # Extract potential topics (nouns, longer words)
            for word in words:
                cleaned = word.strip('.,!?;:"\'')
                if len(cleaned) > 5 and cleaned not in common_words:
                    topics.add(cleaned)
        
        return list(topics)[:10]  # Return top 10
