"""Camera capture helpers for a side-by-side stereo USB stream.

The expected raw frame layout is [left | right] in a single image.
"""

import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, List

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


class BufferedStereoCamera:
    """Continuously capture from one stereo device in a background thread.

    The latest split frame is kept in memory so consumers can fetch a fresh frame
    without blocking on camera I/O.
    """

    def __init__(self, cfg: CameraConfig, name: Optional[str] = None):
        self.cfg = cfg
        self.name = name or cfg.device
        self._camera = StereoCamera(cfg)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[Tuple[np.ndarray, np.ndarray, float, int]] = None
        self._frame_id = 0
        self._last_error: Optional[str] = None

    def start(self) -> None:
        """Start background capture loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, name=f"cam:{self.name}", daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        """Read frames continuously; auto-reopen on transient capture failures."""
        while self._running:
            try:
                if self._camera.cap is None:
                    self._camera.open()
                left, right, ts = self._camera.read()
                self._frame_id += 1
                with self._lock:
                    self._latest = (left, right, ts, self._frame_id)
                    self._last_error = None
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                self._camera.release()
                time.sleep(0.1)

    def get_latest(self) -> Optional[Tuple[np.ndarray, np.ndarray, float, int]]:
        """Return latest captured stereo pair and monotonically increasing frame id."""
        with self._lock:
            return self._latest

    def get_last_error(self) -> Optional[str]:
        """Return last capture/open error message for diagnostics."""
        with self._lock:
            return self._last_error

    def stop(self) -> None:
        """Stop background loop and release camera resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._camera.release()


class MultiStereoCamera:
    """Manage multiple pre-opened stereo inputs with fast active-source switching."""

    def __init__(
        self,
        configs: List[CameraConfig],
        single_active_mode: bool = False,
        initial_active_index: int = 0,
    ):
        if len(configs) == 0:
            raise ValueError("At least one camera config is required")
        self.sources: List[BufferedStereoCamera] = [
            BufferedStereoCamera(cfg, name=f"{idx}:{cfg.device}") for idx, cfg in enumerate(configs)
        ]
        self._active_idx = max(0, min(int(initial_active_index), len(self.sources) - 1))
        self._active_lock = threading.Lock()
        self.single_active_mode = bool(single_active_mode)

    @property
    def source_count(self) -> int:
        return len(self.sources)

    def start(self, stagger_s: float = 0.12) -> None:
        """Start all camera reader threads.

        A small stagger avoids burst-open contention on USB/V4L2 stacks.
        """
        if self.single_active_mode:
            self.sources[self.active_index()].start()
            return

        delay = max(0.0, float(stagger_s))
        for idx, src in enumerate(self.sources):
            src.start()
            if delay > 0.0 and idx < len(self.sources) - 1:
                time.sleep(delay)

    def switch_to(self, index: int) -> int:
        """Switch active source by index and return applied index."""
        idx = max(0, min(int(index), len(self.sources) - 1))
        old_idx = self.active_index()
        if self.single_active_mode and idx != old_idx:
            # Start target source first; stop previous source to avoid
            # multi-stream bandwidth contention on constrained USB links.
            self.sources[idx].start()
            with self._active_lock:
                self._active_idx = idx
            self.sources[old_idx].stop()
        else:
            with self._active_lock:
                self._active_idx = idx
        return idx

    def active_index(self) -> int:
        with self._active_lock:
            return self._active_idx

    def source_statuses(self) -> List[dict]:
        """Return diagnostic status for each configured source."""
        now = time.perf_counter()
        out: List[dict] = []
        for idx, src in enumerate(self.sources):
            latest = src.get_latest()
            err = src.get_last_error()
            has_frame = latest is not None
            frame_age_ms = None
            frame_id = -1
            if latest is not None:
                frame_age_ms = max(0.0, (now - float(latest[2])) * 1000.0)
                frame_id = int(latest[3])
            out.append(
                {
                    "index": idx,
                    "device": src.cfg.device,
                    "has_frame": has_frame,
                    "frame_age_ms": frame_age_ms,
                    "frame_id": frame_id,
                    "last_error": err,
                }
            )
        return out

    def read(
        self,
        timeout_s: float = 0.5,
        min_timestamp_s: float = 0.0,
        allow_fallback: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, float, int, int]:
        """Read latest frame from active source.

        Returns:
            left, right, timestamp, frame_id, source_index
        """
        deadline = time.perf_counter() + max(0.0, timeout_s)
        active_idx = self.active_index()
        while True:
            active_idx = self.active_index()
            latest = self.sources[active_idx].get_latest()
            if latest is not None:
                left, right, ts, frame_id = latest
                if ts >= float(min_timestamp_s):
                    return left, right, ts, frame_id, active_idx
            if time.perf_counter() >= deadline:
                break
            time.sleep(0.002)

        if allow_fallback:
            # Keep pipeline responsive: return freshest frame from any source,
            # but do not overwrite the active selection.
            freshest: Optional[Tuple[int, Tuple[np.ndarray, np.ndarray, float, int]]] = None
            for fallback_idx, src in enumerate(self.sources):
                fallback = src.get_latest()
                if fallback is None:
                    continue
                if freshest is None or fallback[2] > freshest[1][2]:
                    freshest = (fallback_idx, fallback)
            if freshest is not None:
                fallback_idx, fallback = freshest
                left, right, ts, frame_id = fallback
                return left, right, ts, frame_id, fallback_idx

        if time.perf_counter() >= deadline:
                details = []
                for i, src in enumerate(self.sources):
                    err = src.get_last_error()
                    if err:
                        details.append(f"{i}:{err}")
                detail_text = "; ".join(details) if details else "no source reports frames yet"
                raise RuntimeError(
                    "No frame available on active source "
                    f"{active_idx} within {timeout_s:.2f}s. errors={detail_text}"
                )

        raise RuntimeError("Unexpected read state")

    def release(self) -> None:
        """Stop all sources and release devices."""
        for src in self.sources:
            src.stop()
