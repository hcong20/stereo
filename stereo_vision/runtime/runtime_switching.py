"""Runtime helpers for multi-input switch/read flow in main loop."""

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from stereo_vision.capture.camera import CameraManger


@dataclass
class SwitchRuntimeState:
    """Track switch state, timeout policy, and recent read status.

    Attributes:
        timeout_s: Max allowed switch settle time before timeout fallback.
        read_timeout_idle_s: Frame read timeout during steady-state operation.
        read_timeout_switch_s: Frame read timeout while a switch is pending.
        fallback_max_age_s: Max age for fallback frames when allowed.
        pending: Optional pending switch payload with from/to/t0 fields.
        last_good_frame: Last successfully captured frame tuple.
        last_switch_latency_ms: Most recent completed switch latency in ms.
        last_switch_breakdown: Optional backend timing segments for last switch.
    """

    timeout_s: float
    read_timeout_idle_s: float
    read_timeout_switch_s: float
    fallback_max_age_s: float
    pending: Optional[dict] = None
    last_good_frame: Optional[tuple[np.ndarray, np.ndarray, float, int]] = None
    last_switch_latency_ms: Optional[float] = None
    last_switch_breakdown: Optional[dict] = None


def configure_switch_runtime_state(
    multi_mode: bool,
    cam: Any,
    switch_timeout_s: float,
) -> SwitchRuntimeState:
    """Build runtime switch state with mode-aware timeout settings.

    Args:
        multi_mode: Whether multi-input switching mode is enabled.
        cam: Camera backend instance.
        switch_timeout_s: User-configured switch timeout.

    Returns:
        Initialized ``SwitchRuntimeState`` with bounded read/fallback timeouts.
    """
    timeout_s = float(switch_timeout_s)
    if multi_mode and isinstance(cam, CameraManger) and cam.single_active_mode and timeout_s < 3.0:
        print(
            "[WARN] Single-active mode detected with slow camera restart characteristics; "
            f"raising runtime switch timeout from {timeout_s:.2f}s to 3.00s"
        )
        timeout_s = 3.0

    return SwitchRuntimeState(
        timeout_s=timeout_s,
        read_timeout_idle_s=min(0.05, timeout_s),
        read_timeout_switch_s=min(0.20, timeout_s),
        fallback_max_age_s=max(0.12, min(0.4, timeout_s)),
    )


def read_active_frame(
    cam: Any,
    multi_mode: bool,
    state: SwitchRuntimeState,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Read current active frame with switch-aware read constraints.

    Args:
        cam: Camera backend instance.
        multi_mode: Whether multi-input switching mode is enabled.
        state: Runtime switching state.

    Returns:
        Tuple ``(left, right, timestamp_s, active_idx)``.
    """
    if multi_mode:
        min_ts = 0.0
        timeout = state.read_timeout_idle_s
        allow_fallback = True
        if state.pending is not None:
            pending_t0 = float(state.pending["t0"])
            min_ts = pending_t0
            timeout = state.read_timeout_switch_s
            # During a pending switch, wait for target fresh frame only.
            allow_fallback = False
        l, r, ts, _, idx = cam.read(
            timeout_s=timeout,
            min_timestamp_s=min_ts,
            allow_fallback=allow_fallback,
            max_fallback_age_s=state.fallback_max_age_s,
        )
        return l, r, ts, idx

    l, r, ts = cam.read()
    return l, r, ts, 0


def update_switch_breakdown_snapshot(multi_mode: bool, cam: Any, state: SwitchRuntimeState) -> None:
    """Refresh cached switch timing breakdown from backend.

    Args:
        multi_mode: Whether multi-input switching mode is enabled.
        cam: Camera backend instance.
        state: Runtime switching state to update.

    Returns:
        None.
    """
    if multi_mode and isinstance(cam, CameraManger):
        breakdown = cam.get_last_switch_breakdown()
        if breakdown is not None:
            state.last_switch_breakdown = breakdown


def recover_frame_after_read_error(
    cam: Any,
    state: SwitchRuntimeState,
) -> Optional[tuple[np.ndarray, np.ndarray, float, int]]:
    """Handle read failure and return fallback frame when possible.

    Args:
        cam: Camera backend instance.
        state: Runtime switching state.

    Returns:
        Last known good frame tuple, or ``None`` when no fallback exists.
    """
    if state.pending is not None:
        pending_idx = int(state.pending["to_idx"])
        pending_t0 = float(state.pending["t0"])
        if (time.perf_counter() - pending_t0) >= state.timeout_s:
            statuses = cam.source_statuses()
            target = statuses[pending_idx]
            age_text = "N/A" if target["frame_age_ms"] is None else f"{target['frame_age_ms']:.0f}ms"
            err_text = target["last_error"] if target["last_error"] else "-"
            print(
                "[WARN] Switch timed out: "
                f"requested_input={pending_idx + 1}, fallback_input=unchanged, "
                f"target_has_frame={target['has_frame']}, target_age={age_text}, "
                f"target_running={target['running']}, target_thread={target['thread_alive']}, "
                f"target_err={err_text}"
            )
            state.pending = None

    return state.last_good_frame


def finalize_pending_switch(
    cam: Any,
    multi_mode: bool,
    state: SwitchRuntimeState,
    active_idx: int,
    frame_ts: float,
) -> None:
    """Finalize a pending switch or mark timeout based on fresh frame info.

    Args:
        cam: Camera backend instance.
        multi_mode: Whether multi-input switching mode is enabled.
        state: Runtime switching state.
        active_idx: Active index of latest captured frame.
        frame_ts: Timestamp of latest captured frame.

    Returns:
        None.
    """
    if state.pending is None:
        return

    pending_idx = int(state.pending["to_idx"])
    pending_from_idx = int(state.pending["from_idx"])
    pending_t0 = float(state.pending["t0"])

    if pending_idx == active_idx and frame_ts >= pending_t0:
        state.last_switch_latency_ms = (time.perf_counter() - pending_t0) * 1000.0
        update_switch_breakdown_snapshot(multi_mode, cam, state)
        if state.last_switch_breakdown is not None:
            print(
                "[INFO] Switch complete: "
                f"from_input={pending_from_idx + 1}, to_input={active_idx + 1}, "
                f"latency={state.last_switch_latency_ms:.1f}ms, "
                f"[seg_stop={float(state.last_switch_breakdown.get('stop_ms', 0.0)):.0f}ms, "
                f"seg_open={float(state.last_switch_breakdown.get('open_ms', 0.0)):.0f}ms, "
                f"seg_frame={float(state.last_switch_breakdown.get('first_frame_ms', 0.0)):.0f}ms, "
                f"seg_total={float(state.last_switch_breakdown.get('total_ms', 0.0)):.0f}ms]"
            )
        else:
            print(
                "[INFO] Switch complete: "
                f"from_input={pending_from_idx + 1}, to_input={active_idx + 1}, "
                f"latency={state.last_switch_latency_ms:.1f}ms"
            )
        state.pending = None
        return

    if (time.perf_counter() - pending_t0) >= state.timeout_s:
        # Target source did not deliver a fresh frame in time.
        # Keep showing available stream and clear pending state.
        cam.switch_to(active_idx)
        statuses = cam.source_statuses()
        target = statuses[pending_idx]
        age_text = "N/A" if target["frame_age_ms"] is None else f"{target['frame_age_ms']:.0f}ms"
        err_text = target["last_error"] if target["last_error"] else "-"
        print(
            "[WARN] Switch timed out: "
            f"requested_input={pending_idx + 1}, fallback_input={active_idx + 1}, "
            f"target_has_frame={target['has_frame']}, target_age={age_text}, "
            f"target_running={target['running']}, target_thread={target['thread_alive']}, "
            f"target_err={err_text}"
        )
        state.pending = None


def request_switch(
    cam: Any,
    from_idx: int,
    to_idx: int,
    key_label: str,
) -> dict:
    """Request camera source switch and build pending-switch payload.

    Args:
        cam: Camera backend instance.
        from_idx: Previously active source index.
        to_idx: Requested target source index.
        key_label: User action label that triggered the switch.

    Returns:
        Pending switch payload with ``from_idx``, ``to_idx``, and start timestamp.
    """
    active_applied = cam.switch_to(to_idx)
    t_sw = time.perf_counter()
    print(
        "[INFO] Switch request: "
        f"from_input={from_idx + 1}, to_input={active_applied + 1}, key={key_label}"
    )
    return {
        "from_idx": int(from_idx),
        "to_idx": int(active_applied),
        "t0": float(t_sw),
    }


def capture_active_frame_and_finalize(
    *,
    cam: Any,
    multi_mode: bool,
    state: SwitchRuntimeState,
) -> tuple[np.ndarray, np.ndarray, float, int, float]:
    """Capture frame, maintain switch state, and return capture timing.

    Args:
        cam: Camera backend instance.
        multi_mode: Whether multi-input switching mode is enabled.
        state: Runtime switching state.

    Returns:
        Tuple ``(left, right, frame_ts, active_idx, t_capture)``.
    """
    try:
        left, right, frame_ts, active_idx = read_active_frame(
            cam=cam,
            multi_mode=multi_mode,
            state=state,
        )
        state.last_good_frame = (left, right, frame_ts, active_idx)
        update_switch_breakdown_snapshot(multi_mode, cam, state)
    except RuntimeError:
        recovered = recover_frame_after_read_error(
            cam=cam,
            state=state,
        )
        if recovered is None:
            raise
        left, right, frame_ts, active_idx = recovered

    t_capture = time.perf_counter()
    finalize_pending_switch(
        cam=cam,
        multi_mode=multi_mode,
        state=state,
        active_idx=active_idx,
        frame_ts=frame_ts,
    )
    return left, right, frame_ts, active_idx, t_capture


def get_preview_pair_for_active_frame(
    *,
    cam: Any,
    multi_mode: bool,
    preview_nv12_bgr: bool,
    active_idx: int,
    frame_ts: float,
) -> Optional[tuple[Any, Any]]:
    """Get optional preview pair aligned with current active frame.

    Args:
        cam: Camera backend instance.
        multi_mode: Whether multi-input switching mode is enabled.
        preview_nv12_bgr: Whether preview conversion path is enabled.
        active_idx: Active source index for current frame.
        frame_ts: Timestamp of current main processing frame.

    Returns:
        ``(left_preview, right_preview)`` when available and aligned, else ``None``.
    """
    if not preview_nv12_bgr:
        return None

    if multi_mode and hasattr(cam, "get_latest_preview"):
        preview_packet = cam.get_latest_preview(active_idx)
        if preview_packet is not None:
            p_left, p_right, p_ts, _, p_idx = preview_packet
            if int(p_idx) == int(active_idx) and abs(float(p_ts) - float(frame_ts)) <= 0.2:
                return (p_left, p_right)
    elif hasattr(cam, "get_last_preview_pair"):
        return cam.get_last_preview_pair()

    return None