"""
Placeholder frame captions when Cosmos cannot be downloaded (no Hugging Face access).

Set MOCK_COSMOS=1 in .env to test the rest of the pipeline (summaries, Ollama, UI).
"""
from __future__ import annotations

from typing import Dict, List

from PIL import Image


def mock_analyze_frames(
    frames: List[Image.Image],
    timestamps: List[float],
) -> List[Dict[str, str]]:
    """Same shape as CosmosModelHandler.analyze_frames, fake descriptions."""
    out: List[Dict[str, str]] = []
    for idx, ts in enumerate(timestamps):
        out.append(
            {
                "frame_index": idx,
                "description": (
                    f"[MOCK vision — no Cosmos model] Frame {idx + 1} at ~{ts:.2f}s. "
                    "Visual content not analyzed; use real model when HF download works."
                ),
            }
        )
    return out
