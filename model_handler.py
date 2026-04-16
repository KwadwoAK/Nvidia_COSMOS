import os
import base64
import json
from io import BytesIO
from PIL import Image
from typing import List, Dict, Optional

import requests

VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://10.20.1.116:8000/v1")
MODEL_ID = os.getenv("COSMOS_MODEL_ID", "nvidia/Cosmos-Reason2-8B")


def _image_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    buf = BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class CosmosModelHandler:
    """Handles interaction with Nvidia's Cosmos-Reason2-8B via a vLLM API container."""

    def __init__(
        self,
        api_base: str = VLLM_API_BASE,
        model_id: str = MODEL_ID,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        self.api_base = api_base.rstrip("/")
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._session = requests.Session()
        print(f"CosmosModelHandler connected to {self.api_base} using model {self.model_id}")

    def _chat(self, messages: list, stream: bool = False, extra_body: Optional[dict] = None) -> str:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": stream,
        }
        if extra_body:
            payload.update(extra_body)

        resp = self._session.post(
            f"{self.api_base}/chat/completions",
            json=payload,
            stream=stream,
            timeout=120,
        )
        resp.raise_for_status()

        if not stream:
            return resp.json()["choices"][0]["message"]["content"]

        # SSE streaming — accumulate and return full text
        text = ""
        for line in resp.iter_lines():
            if not line or line == b"data: [DONE]":
                continue
            if line.startswith(b"data: "):
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0].get("delta", {}).get("content", "")
                text += delta
        return text

    def analyze_single_frame(
        self,
        image: Image.Image,
        prompt: str = "Describe what is happening in this image in detail.",
    ) -> str:
        b64 = _image_to_base64(image)
        messages = [
            {"role": "system", "content": "You are a video surveillance analyst."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            },
        ]
        try:
            return self._chat(messages)
        except Exception as e:
            print(f"Error analyzing frame: {e}")
            return f"Error: Could not analyze frame - {str(e)}"

    def analyze_frames(
        self,
        frames: List[Image.Image],
        batch_size: int = 1,
    ) -> List[Dict[str, str]]:
        """Analyze a list of PIL Images one at a time via the vLLM image API."""
        descriptions = []
        for idx, frame in enumerate(frames):
            print(f"Analyzing frame {idx + 1}/{len(frames)}...")
            prompt = (
                f"You are analyzing frame {idx + 1} of a video. "
                "Describe what is happening in this image, including: "
                "1) The main subjects or objects, "
                "2) The action or activity taking place, "
                "3) The setting or environment, "
                "4) Any notable details. "
                "Be concise but informative."
            )
            description = self.analyze_single_frame(frame, prompt)
            descriptions.append({"frame_index": idx, "description": description})
        return descriptions

    def analyze_video(
        self,
        video_bytes: bytes,
        prompt: str = "Analyze this footage and describe what is happening.",
        fps: int = 4,
        mime_type: str = "video/mp4",
    ) -> str:
        """
        Send a full video clip to the model in one request.
        More efficient than per-frame calls and allows cross-frame temporal reasoning.

        Args:
            video_bytes: Raw video file bytes.
            prompt: Instruction prompt for the model.
            fps: Frames per second the model should sample from the video.
            mime_type: MIME type of the video (e.g. 'video/mp4').

        Returns:
            Model's analysis of the video.
        """
        b64 = base64.b64encode(video_bytes).decode("utf-8")
        messages = [
            {"role": "system", "content": "You are a video surveillance analyst."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:{mime_type};base64,{b64}"},
                    },
                ],
            },
        ]
        extra = {"media_io_kwargs": {"video": {"fps": fps}}}
        try:
            return self._chat(messages, extra_body=extra)
        except Exception as e:
            print(f"Error analyzing video: {e}")
            return f"Error: Could not analyze video - {str(e)}"

    def analyze_with_context(
        self,
        frames: List[Image.Image],
        previous_context: str = "",
    ) -> List[Dict[str, str]]:
        """Analyze frames sequentially, feeding each result as context for the next."""
        descriptions = []
        context = previous_context

        for idx, frame in enumerate(frames):
            print(f"Analyzing frame {idx + 1}/{len(frames)} with context...")
            if context:
                prompt = (
                    f"Previously in this video: {context}\n\n"
                    "Now, describe what is happening in this new frame. "
                    "Focus on what has changed or what is new."
                )
            else:
                prompt = "This is the first frame of a video. Describe what you see in detail."

            description = self.analyze_single_frame(frame, prompt)
            descriptions.append({"frame_index": idx, "description": description})
            context = description

        return descriptions
