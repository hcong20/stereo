"""Buffered background capture for one stereo source."""

import threading
import time
from typing import Optional, Tuple

import numpy as np

from stereo_vision.camera_single import CameraConfig, StereoCamera


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
        self._latest_preview: Optional[Tuple[np.ndarray, np.ndarray, float, int]] = None
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
        consecutive_failures = 0
        reopen_backoff_s = 0.1
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
                consecutive_failures = 0
                reopen_backoff_s = 0.1
                preview_pair = self._camera.get_last_preview_pair()
                with self._lock:
                    self._latest = (left, right, ts, self._frame_id)
                    if preview_pair is None:
                        self._latest_preview = None
                    else:
                        self._latest_preview = (
                            preview_pair[0],
                            preview_pair[1],
                            ts,
                            self._frame_id,
                        )
                    self._last_error = None
            except Exception as exc:
                msg = str(exc)
                with self._lock:
                    self._last_error = msg

                consecutive_failures += 1
                hard_reopen = (
                    self._camera.cap is None
                    or "Unexpected frame size after open" in msg
                    or "Captured frame width is smaller than expected stereo width" in msg
                )
                soft_reopen = consecutive_failures >= 5

                if hard_reopen or soft_reopen:
                    self._camera.release()
                    time.sleep(reopen_backoff_s)
                    reopen_backoff_s = min(1.0, reopen_backoff_s * 1.8)
                    consecutive_failures = 0
                else:
                    # Retry transient read failures a few times before reopen.
                    time.sleep(min(0.02 * consecutive_failures, 0.1))

        # Release from capture thread context when a non-blocking stop is used.
        self._camera.release()

    def get_latest(self) -> Optional[Tuple[np.ndarray, np.ndarray, float, int]]:
        """Return latest captured stereo pair and monotonically increasing frame id."""
        with self._lock:
            return self._latest

    def get_latest_preview(self) -> Optional[Tuple[np.ndarray, np.ndarray, float, int]]:
        """Return latest preview-only BGR pair and monotonically increasing frame id."""
        with self._lock:
            return self._latest_preview

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