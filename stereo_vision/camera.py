"""Camera capture helpers for a side-by-side stereo USB stream.

The expected raw frame layout is [left | right] in a single image.
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class CameraConfig:
    """Runtime camera settings for OpenCV V4L2/GStreamer capture.

    Attributes:
        device: Video device node, e.g. /dev/video0.
        width: Combined frame width (left + right cameras).
        height: Frame height.
        fps: Target capture FPS.
        use_gstreamer: Use custom GStreamer pipeline when True.
        gstreamer_pipeline: Optional explicit pipeline string.
        warmup_frames: Frames to discard after open.
    """

    device: str = "/dev/video0"
    width: int = 1280
    height: int = 480
    fps: int = 30
    use_gstreamer: bool = False
    gstreamer_pipeline: Optional[str] = None
    warmup_frames: int = 5


def build_usb_gstreamer_pipeline(device: str, width: int, height: int, fps: int) -> str:
    """Build a low-latency GStreamer pipeline for V4L2 USB camera capture.

    The pipeline prefers low buffering to minimize capture latency.
    """
    return (
        f"v4l2src device={device} io-mode=2 ! "
        f"video/x-raw,format=YUY2,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert n-threads=2 ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


class StereoCamera:
    """Capture combined stereo frame and split into left/right images.

    This wrapper hides backend differences (V4L2 vs GStreamer) and always
    returns synchronized left/right images from a single captured frame.
    """

    def __init__(self, cfg: CameraConfig):
        """Initialize camera wrapper with immutable capture configuration."""
        self.cfg = cfg
        self.cap: Optional[cv2.VideoCapture] = None
        self.last_ts: float = 0.0

    def open(self) -> None:
        """Open camera stream and warm it up before first use.

        Raises:
            RuntimeError: If camera cannot be opened.
        """
        if self.cfg.use_gstreamer:
            pipeline = self.cfg.gstreamer_pipeline or build_usb_gstreamer_pipeline(
                self.cfg.device, self.cfg.width, self.cfg.height, self.cfg.fps
            )
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(self.cfg.device, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Failed to open stereo camera")

        # Drop initial frames so exposure/auto-controls stabilize.
        for _ in range(max(0, self.cfg.warmup_frames)):
            self.cap.read()

    def read(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Read one combined frame, split into left/right images, and timestamp it.

        Returns:
            (left_bgr, right_bgr, capture_timestamp_seconds).
        """
        if self.cap is None:
            raise RuntimeError("Camera is not open")

        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read camera frame")

        # Height is kept for readability and potential future validation hooks.
        h, w = frame.shape[:2]
        if w % 2 != 0:
            raise ValueError(f"Expected combined frame width to be even, got {w}")

        # Input frame is expected to be side-by-side: left half + right half.
        half = w // 2
        left = frame[:, :half]
        right = frame[:, half:]

        self.last_ts = time.perf_counter()
        return left, right, self.last_ts

    def release(self) -> None:
        """Release camera resources if a stream is currently open."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
