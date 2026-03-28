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
            if (not bool(self.cfg.fast_reopen)) or (not was_configured):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self._configured_once = True
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Failed to open stereo camera")

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

    def __init__(
        self,
        cfg: CameraConfig,
        name: Optional[str] = None,
    ):
        self.cfg = cfg
        self.name = name or cfg.device
        self._camera = StereoCamera(cfg)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()
        self._lock = threading.Lock()
        self._latest: Optional[Tuple[np.ndarray, np.ndarray, float, int]] = None
        self._frame_id = 0
        self._last_error: Optional[str] = None
        self._last_open_ms: float = 0.0
        self._last_open_ts: float = 0.0
        self._open_count: int = 0

    def start(self) -> None:
        """Start background capture loop."""
        with self._state_lock:
            if self._running:
                return

            # If a previous thread is still winding down, try to reuse it by
            # cancelling the stop request. This avoids missing rapid re-switches.
            if self._thread is not None and self._thread.is_alive():
                self._running = True
                self._thread.join(timeout=0.03)
                if self._thread.is_alive():
                    return
                self._thread = None

            self._running = True
            self._thread = threading.Thread(target=self._reader_loop, name=f"cam:{self.name}", daemon=True)
            self._thread.start()

    def _reader_loop(self) -> None:
        """Read frames continuously; auto-reopen on transient capture failures."""
        while self._running:
            try:
                if self._camera.cap is None:
                    t_open0 = time.perf_counter()
                    self._camera.open()
                    open_ms = (time.perf_counter() - t_open0) * 1000.0
                    with self._lock:
                        self._last_open_ms = open_ms
                        self._last_open_ts = time.perf_counter()
                        self._open_count += 1
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

        # Release from capture thread context when a non-blocking stop is used.
        self._camera.release()

    def get_latest(self) -> Optional[Tuple[np.ndarray, np.ndarray, float, int]]:
        """Return latest captured stereo pair and monotonically increasing frame id."""
        with self._lock:
            return self._latest

    def get_last_error(self) -> Optional[str]:
        """Return last capture/open error message for diagnostics."""
        with self._lock:
            return self._last_error

    def get_open_stats(self) -> tuple[float, float, int]:
        """Return (last_open_ms, last_open_ts, open_count)."""
        with self._lock:
            return self._last_open_ms, self._last_open_ts, self._open_count

    def is_running(self) -> bool:
        """Return whether capture loop is currently requested to run."""
        with self._state_lock:
            return self._running

    def is_thread_alive(self) -> bool:
        """Return whether background thread object is alive."""
        with self._state_lock:
            return self._thread is not None and self._thread.is_alive()

    def stop(
        self,
        wait: bool = True,
        join_timeout_s: float = 1.0,
        release_device: bool = False,
    ) -> None:
        """Stop background loop.

        Args:
            wait: When False, request stop and return immediately.
            join_timeout_s: Max join wait when wait is True.
            release_device: Force releasing capture handle after stop.
        """
        with self._state_lock:
            self._running = False
            thread = self._thread

        if wait and thread is not None:
            thread.join(timeout=max(0.0, float(join_timeout_s)))
            thread_alive = thread.is_alive()
            with self._state_lock:
                if self._thread is thread and not thread_alive:
                    self._thread = None
            # Only release from caller thread when capture thread has stopped.
            # Otherwise the capture thread will release from its own context.
            if not thread_alive:
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
            BufferedStereoCamera(
                cfg,
                name=f"{idx}:{cfg.device}",
            )
            for idx, cfg in enumerate(configs)
        ]
        self._active_idx = max(0, min(int(initial_active_index), len(self.sources) - 1))
        self._active_lock = threading.Lock()
        self.single_active_mode = bool(single_active_mode)
        self._switch_lock = threading.Lock()
        self._pending_switch: Optional[dict] = None
        self._last_switch_breakdown: Optional[dict] = None

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
        t_switch = time.perf_counter()
        stop_ms = 0.0
        start_call_ms = 0.0
        target_prev_frame_id = -1
        target_open_count_before = 0
        target_latest = self.sources[idx].get_latest()
        if target_latest is not None:
            target_prev_frame_id = int(target_latest[3])
        _, _, target_open_count_before = self.sources[idx].get_open_stats()

        if self.single_active_mode and idx != old_idx:
            # On constrained USB links, stop old source first so target source
            # can acquire bus bandwidth and deliver frames reliably.
            t_stop0 = time.perf_counter()
            self.sources[old_idx].stop(wait=True, join_timeout_s=0.03, release_device=True)
            stop_ms = (time.perf_counter() - t_stop0) * 1000.0

            t_start0 = time.perf_counter()
            self.sources[idx].start()
            start_call_ms = (time.perf_counter() - t_start0) * 1000.0
            with self._active_lock:
                self._active_idx = idx
        else:
            t_start0 = time.perf_counter()
            with self._active_lock:
                self._active_idx = idx
            start_call_ms = (time.perf_counter() - t_start0) * 1000.0

        if idx != old_idx:
            with self._switch_lock:
                self._pending_switch = {
                    "request_ts": t_switch,
                    "from_idx": old_idx,
                    "to_idx": idx,
                    "stop_ms": stop_ms,
                    "start_call_ms": start_call_ms,
                    "target_prev_frame_id": target_prev_frame_id,
                    "target_open_count_before": target_open_count_before,
                }

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
                    "running": src.is_running(),
                    "thread_alive": src.is_thread_alive(),
                }
            )
        return out

    def get_last_switch_breakdown(self) -> Optional[dict]:
        """Return latest completed switch timing breakdown.

        Keys:
            from_idx, to_idx, total_ms, stop_ms, open_ms, first_frame_ms
        """
        with self._switch_lock:
            if self._last_switch_breakdown is None:
                return None
            return dict(self._last_switch_breakdown)

    def _maybe_finalize_switch_breakdown(self, source_idx: int, frame_ts: float, frame_id: int) -> None:
        """Complete pending switch metrics when first valid frame arrives."""
        with self._switch_lock:
            pending = self._pending_switch
            if pending is None:
                return
            if int(pending["to_idx"]) != int(source_idx):
                return

            request_ts = float(pending["request_ts"])
            prev_id = int(pending["target_prev_frame_id"])
            if frame_ts < request_ts or int(frame_id) <= prev_id:
                return

            open_ms, _, open_count = self.sources[source_idx].get_open_stats()
            open_count_before = int(pending["target_open_count_before"])
            if open_count <= open_count_before:
                open_ms = 0.0

            total_ms = (time.perf_counter() - request_ts) * 1000.0
            stop_ms = float(pending["stop_ms"])
            first_frame_ms = max(0.0, total_ms - stop_ms - float(open_ms))

            self._last_switch_breakdown = {
                "from_idx": int(pending["from_idx"]),
                "to_idx": int(pending["to_idx"]),
                "total_ms": total_ms,
                "stop_ms": stop_ms,
                "open_ms": float(open_ms),
                "first_frame_ms": first_frame_ms,
                "start_call_ms": float(pending["start_call_ms"]),
            }
            self._pending_switch = None

    def read(
        self,
        timeout_s: float = 0.5,
        min_timestamp_s: float = 0.0,
        allow_fallback: bool = True,
        max_fallback_age_s: float = 0.35,
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
                    self._maybe_finalize_switch_breakdown(active_idx, ts, frame_id)
                    return left, right, ts, frame_id, active_idx
            if time.perf_counter() >= deadline:
                break
            time.sleep(0.002)

        if allow_fallback:
            # Keep pipeline responsive: return freshest frame from any source,
            # but do not overwrite the active selection.
            freshest: Optional[Tuple[int, Tuple[np.ndarray, np.ndarray, float, int]]] = None
            now = time.perf_counter()
            max_age = max(0.0, float(max_fallback_age_s))
            for fallback_idx, src in enumerate(self.sources):
                fallback = src.get_latest()
                if fallback is None:
                    continue
                frame_age = now - float(fallback[2])
                if frame_age > max_age:
                    continue
                if freshest is None or fallback[2] > freshest[1][2]:
                    freshest = (fallback_idx, fallback)
            if freshest is not None:
                fallback_idx, fallback = freshest
                left, right, ts, frame_id = fallback
                self._maybe_finalize_switch_breakdown(fallback_idx, ts, frame_id)
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
            src.stop(wait=True, release_device=True)
