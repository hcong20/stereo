"""Multi-source stereo capture management and switching."""

import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from stereo_vision.camera_buffered import BufferedStereoCamera
from stereo_vision.camera_single import CameraConfig


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