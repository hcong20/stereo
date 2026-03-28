"""Single stereo camera capture helpers.

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
    warmup_frames: int = 1
    fast_reopen: bool = True


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
        self._configured_once = False

    def open(self) -> None:
        """Open camera stream and warm it up before first use.

        Raises:
            RuntimeError: If camera cannot be opened.
        """
        was_configured = self._configured_once
        if self.cfg.use_gstreamer:
            pipeline = self.cfg.gstreamer_pipeline or build_usb_gstreamer_pipeline(
                self.cfg.device, self.cfg.width, self.cfg.height, self.cfg.fps
            )
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(self.cfg.device, cv2.CAP_V4L2)

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Failed to open stereo camera")

        if not self.cfg.use_gstreamer:
            # Always enforce combined stereo dimensions; otherwise some drivers
            # reopen at half width and produce visibly half-frame output.
            force_full_config = (not bool(self.cfg.fast_reopen)) or (not was_configured)
            if force_full_config:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            else:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            expected_min_w = max(2, int(self.cfg.width * 0.75))
            expected_min_h = max(2, int(self.cfg.height * 0.75))
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            if actual_w > 0 and actual_h > 0 and (actual_w < expected_min_w or actual_h < expected_min_h):
                # Retry one strict reconfiguration before failing the open.
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                time.sleep(0.01)
                actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            if actual_w > 0 and actual_h > 0 and (actual_w < expected_min_w or actual_h < expected_min_h):
                raise RuntimeError(
                    "Unexpected frame size after open "
                    f"({actual_w}x{actual_h}); expected around {self.cfg.width}x{self.cfg.height}"
                )

        self._configured_once = True

        # Drop initial frames so exposure/auto-controls stabilize.
        warmup = max(0, int(self.cfg.warmup_frames))
        if bool(self.cfg.fast_reopen) and was_configured:
            # On reopen, skipping warmup reduces switch delay significantly.
            warmup = 0
        for _ in range(warmup):
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
        min_expected_w = max(2, int(self.cfg.width * 0.75))
        if w < min_expected_w:
            raise RuntimeError(
                "Captured frame width is smaller than expected stereo width: "
                f"got {w}, expected around {self.cfg.width}"
            )
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