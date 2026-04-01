"""Runtime visualization builders for the stereo main loop."""

import time
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

from stereo_vision.roi import ROI
from stereo_vision.rectification import rectify_pair
from stereo_vision.visualization import colorize_disparity, draw_roi, draw_text


def build_viz_layers(
    left_rect: np.ndarray,
    disparity: np.ndarray,
    roi_scaled: ROI,
    min_disp: float,
    max_disp: float,
    distance_filtered: Optional[float],
    fps: float,
    latency_ms: float,
    active_idx: int,
    device_list: Sequence[str],
    last_switch_latency_ms: Optional[float],
    switch_pending: Optional[dict],
    last_switch_breakdown: Optional[dict],
    valid_pixels: int,
    total_pixels: int,
    valid_ratio: float,
    positive_disp_pixels: int,
    clipped_by_max_depth: int,
    distance_raw: Optional[float],
    roi_gate_note: Optional[str] = None,
    roi_tune_preset: str = "off",
    runtime_note: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build left diagnostic overlay and colorized disparity visualization."""
    disp_vis = colorize_disparity(
        disparity,
        min_disp=float(min_disp),
        max_disp=float(max_disp),
    )
    left_viz_src = left_rect
    if left_viz_src.ndim == 2 or (left_viz_src.ndim == 3 and left_viz_src.shape[2] == 1):
        left_viz_src = cv2.cvtColor(left_viz_src, cv2.COLOR_GRAY2BGR)
    left_viz = draw_roi(left_viz_src, roi_scaled)

    if distance_filtered is not None:
        left_viz = draw_text(left_viz, f"ROI Distance: {distance_filtered * 1000.0:.1f} mm", (10, 30), (0, 255, 0))
    else:
        left_viz = draw_text(left_viz, "ROI Distance: N/A", (10, 30), (0, 0, 255))

    left_viz = draw_text(left_viz, f"FPS: {fps:.1f}", (10, 60), (255, 255, 0))
    left_viz = draw_text(left_viz, f"Latency: {latency_ms:.1f} ms", (10, 90), (255, 255, 0))
    left_viz = draw_text(left_viz, f"ROI Preset: {roi_tune_preset}", (10, 105), (255, 255, 0))
    left_viz = draw_text(
        left_viz,
        f"Input: {active_idx + 1}/{len(device_list)} ({device_list[active_idx]})",
        (10, 130),
        (255, 255, 0),
    )

    if last_switch_latency_ms is not None:
        left_viz = draw_text(
            left_viz,
            f"Switch latency: {last_switch_latency_ms:.1f} ms",
            (10, 160),
            (200, 255, 200),
        )
    elif switch_pending is not None:
        pending_idx = int(switch_pending["to_idx"])
        pending_t0 = float(switch_pending["t0"])
        pending_ms = (time.perf_counter() - pending_t0) * 1000.0
        left_viz = draw_text(
            left_viz,
            f"Switching to input {pending_idx + 1}: {pending_ms:.0f} ms",
            (10, 160),
            (0, 215, 255),
        )
    else:
        left_viz = draw_text(
            left_viz,
            "Switch latency: N/A",
            (10, 160),
            (255, 255, 0),
        )

    if last_switch_breakdown is not None:
        stop_ms = float(last_switch_breakdown.get("stop_ms", 0.0))
        open_ms = float(last_switch_breakdown.get("open_ms", 0.0))
        frame_ms = float(last_switch_breakdown.get("first_frame_ms", 0.0))
        total_ms = float(last_switch_breakdown.get("total_ms", 0.0))
        left_viz = draw_text(
            left_viz,
            f"Sw seg(ms) stop/open/frame={stop_ms:.0f}/{open_ms:.0f}/{frame_ms:.0f} total={total_ms:.0f}",
            (10, 330),
            (200, 255, 200),
        )

    left_viz = draw_text(
        left_viz,
        f"ROI valid: {valid_pixels}/{total_pixels}",
        (10, 190),
        (255, 255, 0),
    )
    left_viz = draw_text(
        left_viz,
        f"ROI valid ratio: {valid_ratio * 100.0:.1f}%",
        (10, 205),
        (255, 255, 0),
    )
    left_viz = draw_text(
        left_viz,
        f"ROI disp>0: {positive_disp_pixels}/{total_pixels}",
        (10, 235),
        (255, 255, 0),
    )
    left_viz = draw_text(
        left_viz,
        f"ROI clipped(maxD): {clipped_by_max_depth}",
        (10, 265),
        (255, 255, 0),
    )

    if distance_raw is not None:
        left_viz = draw_text(left_viz, f"ROI Raw: {distance_raw * 1000.0:.1f} mm", (10, 295), (200, 255, 200))
    else:
        left_viz = draw_text(left_viz, "ROI Raw: N/A", (10, 295), (0, 0, 255))

    if roi_gate_note:
        left_viz = draw_text(left_viz, f"ROI Gate: {roi_gate_note}", (10, 325), (0, 0, 255))

    if runtime_note:
        left_viz = draw_text(left_viz, runtime_note, (10, 355), (0, 255, 0))

    return left_viz, disp_vis


def apply_click_probe_overlay(
    left_viz: np.ndarray,
    click_px: Optional[Tuple[int, int]],
    click_depth: Optional[float],
) -> np.ndarray:
    """Draw click marker and per-pixel depth probe text onto left overlay."""
    if click_px is None:
        return left_viz

    cx, cy = click_px
    if 0 <= cx < left_viz.shape[1] and 0 <= cy < left_viz.shape[0]:
        cv2.circle(left_viz, (cx, cy), 5, (0, 0, 255), -1, cv2.LINE_AA)

    if click_depth is not None:
        return draw_text(left_viz, f"Click({cx},{cy}): {click_depth * 1000.0:.1f} mm", (10, 300), (255, 200, 0))
    return draw_text(left_viz, f"Click({cx},{cy}): N/A", (10, 300), (0, 0, 255))


def compose_runtime_visualization(
    *,
    left_rect: np.ndarray,
    preview_pair_raw: Optional[Tuple[np.ndarray, np.ndarray]],
    rect: Any,
    preview_nv12_warned: bool,
    disparity: np.ndarray,
    roi_scaled: ROI,
    min_disp: float,
    max_disp: float,
    distance_filtered: Optional[float],
    fps: float,
    latency_ms: float,
    active_idx: int,
    device_list: Sequence[str],
    switch_state: Any,
    valid_pixels: int,
    total_pixels: int,
    valid_ratio: float,
    positive_disp_pixels: int,
    clipped_by_max_depth: int,
    distance_raw: Optional[float],
    roi_gate_note: Optional[str],
    roi_tune_preset: str,
    runtime_note_text: Optional[str],
    runtime_note_until: float,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Build left/disparity visualization layers for display."""
    left_viz_source = left_rect
    if preview_pair_raw is not None:
        try:
            preview_left_raw, preview_right_raw = preview_pair_raw
            left_preview_rect, _ = rectify_pair(preview_left_raw, preview_right_raw, rect)
            target_h, target_w = left_rect.shape[:2]
            if left_preview_rect.shape[:2] != (target_h, target_w):
                left_preview_rect = cv2.resize(
                    left_preview_rect,
                    (target_w, target_h),
                    interpolation=cv2.INTER_AREA,
                )
            left_viz_source = left_preview_rect
        except Exception as exc:
            if not preview_nv12_warned:
                print(
                    "[WARN] Preview-only NV12->BGR conversion failed; "
                    f"continuing with grayscale preview: {exc}"
                )
                preview_nv12_warned = True

    left_viz, disp_vis = build_viz_layers(
        left_rect=left_viz_source,
        disparity=disparity,
        roi_scaled=roi_scaled,
        min_disp=float(min_disp),
        max_disp=float(max_disp),
        distance_filtered=distance_filtered,
        fps=fps,
        latency_ms=latency_ms,
        active_idx=active_idx,
        device_list=device_list,
        last_switch_latency_ms=switch_state.last_switch_latency_ms,
        switch_pending=switch_state.pending,
        last_switch_breakdown=switch_state.last_switch_breakdown,
        valid_pixels=valid_pixels,
        total_pixels=total_pixels,
        valid_ratio=valid_ratio,
        positive_disp_pixels=positive_disp_pixels,
        clipped_by_max_depth=clipped_by_max_depth,
        distance_raw=distance_raw,
        roi_gate_note=roi_gate_note,
        roi_tune_preset=roi_tune_preset,
        runtime_note=(runtime_note_text if time.perf_counter() < runtime_note_until else None),
    )
    return left_viz, disp_vis, preview_nv12_warned