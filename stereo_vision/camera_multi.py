"""Multi-source stereo capture management and switching."""

import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from stereo_vision.camera_buffered import BufferedStereoCamera
from stereo_vision.camera_multi_helpers import (
    aligned_live_indices,
    build_group_maps,
    build_pending_switch_payload,
    clamp_index,
    finalize_switch_breakdown,
    normalize_group_ids,
    pick_freshest_fallback,
)
from stereo_vision.camera_single import CameraConfig


class MultiStereoCamera:
    """Manage multiple pre-opened stereo inputs with fast active-source switching."""

    def __init__(
        self,
        configs: List[CameraConfig],
        single_active_mode: bool = False,
        initial_active_index: int = 0,
        group_ids: Optional[List[str]] = None,
        keep_one_live_per_group: bool = False,
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
        source_count = len(self.sources)
        self._active_idx = clamp_index(initial_active_index, source_count)
        self._active_lock = threading.Lock()
        self.single_active_mode = bool(single_active_mode)
        self.group_ids = normalize_group_ids(group_ids, source_count)
        self._group_to_indices, self._index_slot = build_group_maps(self.group_ids)
        self.group_live_mode = bool(self.single_active_mode and keep_one_live_per_group and group_ids is not None)
        self._group_live_index: Dict[str, int] = {}
        self._switch_lock = threading.Lock()
        self._pending_switch: Optional[dict] = None
        self._last_switch_breakdown: Optional[dict] = None

    def _group_of(self, index: int) -> str:
        idx = clamp_index(index, len(self.sources))
        return self.group_ids[idx]

    def _aligned_live_indices(self, reference_index: int) -> Dict[str, int]:
        """Pick one live source per group aligned by intra-group slot.

        Example for groups [0,0,2,2]:
            slot 0 -> (1,3)
            slot 1 -> (2,4)
        """
        return aligned_live_indices(
            group_to_indices=self._group_to_indices,
            index_slot=self._index_slot,
            reference_index=reference_index,
            source_count=len(self.sources),
        )

    @property
    def source_count(self) -> int:
        return len(self.sources)

    def start(self, stagger_s: float = 0.12) -> None:
        """Start all camera reader threads.

        A small stagger avoids burst-open contention on USB/V4L2 stacks.
        """
        if self.single_active_mode:
            if self.group_live_mode:
                grouped = self._aligned_live_indices(self.active_index())
                self._group_live_index = dict(grouped)
                for src_idx in sorted(set(grouped.values())):
                    self.sources[src_idx].start()
                return
            self.sources[self.active_index()].start()
            return

        delay = max(0.0, float(stagger_s))
        for idx, src in enumerate(self.sources):
            src.start()
            if delay > 0.0 and idx < len(self.sources) - 1:
                time.sleep(delay)

    def switch_to(self, index: int) -> int:
        """Switch active source by index and return applied index."""
        idx = clamp_index(index, len(self.sources))
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

        if self.single_active_mode and idx != old_idx and self.group_live_mode:
            target_group_live = self._aligned_live_indices(idx)

            to_stop = []
            for group, current_live in self._group_live_index.items():
                target_live = target_group_live.get(group)
                if target_live is not None and current_live != target_live:
                    to_stop.append(current_live)

            t_stop0 = time.perf_counter()
            for src_idx in sorted(set(to_stop)):
                self.sources[src_idx].stop(wait=True, join_timeout_s=0.03, release_device=True)
            stop_ms = (time.perf_counter() - t_stop0) * 1000.0

            to_start = []
            for group, target_live in target_group_live.items():
                if self._group_live_index.get(group) != target_live:
                    to_start.append(target_live)

            t_start0 = time.perf_counter()
            for src_idx in sorted(set(to_start)):
                self.sources[src_idx].start()
            start_call_ms = (time.perf_counter() - t_start0) * 1000.0

            self._group_live_index = dict(target_group_live)
            with self._active_lock:
                self._active_idx = idx
        elif self.single_active_mode and idx != old_idx:
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
                self._pending_switch = build_pending_switch_payload(
                    request_ts=t_switch,
                    from_idx=old_idx,
                    to_idx=idx,
                    stop_ms=stop_ms,
                    start_call_ms=start_call_ms,
                    target_prev_frame_id=target_prev_frame_id,
                    target_open_count_before=target_open_count_before,
                )

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
                    "group": self._group_of(idx),
                    "group_live": bool(self.group_live_mode and self._group_live_index.get(self._group_of(idx)) == idx),
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
            open_ms, _, open_count = self.sources[source_idx].get_open_stats()
            pending_next, breakdown = finalize_switch_breakdown(
                pending=pending,
                source_idx=source_idx,
                frame_ts=frame_ts,
                frame_id=frame_id,
                open_ms=open_ms,
                open_count=open_count,
                now_ts=time.perf_counter(),
            )
            if breakdown is None:
                return
            self._last_switch_breakdown = breakdown
            self._pending_switch = pending_next

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
            now = time.perf_counter()
            latest_frames = [src.get_latest() for src in self.sources]
            freshest = pick_freshest_fallback(
                latest_frames_by_index=latest_frames,
                now_ts=now,
                max_fallback_age_s=max_fallback_age_s,
            )
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

    def get_latest_preview(
        self,
        source_index: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float, int, int]]:
        """Return latest preview-only BGR frame for requested (or active) source."""
        idx = self.active_index() if source_index is None else clamp_index(source_index, len(self.sources))
        latest = self.sources[idx].get_latest_preview()
        if latest is None:
            return None
        left, right, ts, frame_id = latest
        return left, right, ts, frame_id, idx

    def release(self) -> None:
        """Stop all sources and release devices."""
        for src in self.sources:
            src.stop(wait=True, release_device=True)