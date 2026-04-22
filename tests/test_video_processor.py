import pytest

from video_processor import VideoProcessor


def test_format_timestamp_outputs_mm_ss_and_hh_mm_ss():
    processor = VideoProcessor()

    assert processor.format_timestamp(125.5) == "02:05"
    assert processor.format_timestamp(3661) == "01:01:01"


def test_extract_frames_raises_when_video_cannot_be_opened(monkeypatch):
    class _ClosedCapture:
        def isOpened(self):
            return False

    monkeypatch.setattr("video_processor.cv2.VideoCapture", lambda _: _ClosedCapture())

    processor = VideoProcessor()
    with pytest.raises(ValueError, match="Could not open video file"):
        processor.extract_frames("missing.mp4")
