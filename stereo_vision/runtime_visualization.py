"""Runtime visualization builders for the stereo main loop."""

import time
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from stereo_vision.roi import ROI
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
    left_viz = draw_text(
        left_viz,
        f"Input: {active_idx + 1}/{len(device_list)} ({device_list[active_idx]})",
        (10, 120),
        (255, 255, 0),
    )

    if last_switch_latency_ms is not None:
        left_viz = draw_text(
            left_viz,
            f"Switch latency: {last_switch_latency_ms:.1f} ms",
            (10, 150),
            (200, 255, 200),
        )
    elif switch_pending is not None:
        pending_idx = int(switch_pending["to_idx"])
        pending_t0 = float(switch_pending["t0"])
        pending_ms = (time.perf_counter() - pending_t0) * 1000.0
        left_viz = draw_text(
            left_viz,
            f"Switching to input {pending_idx + 1}: {pending_ms:.0f} ms",
            (10, 150),
            (0, 215, 255),
        )
    else:
        left_viz = draw_text(
            left_viz,
            "Switch latency: N/A",
            (10, 150),
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
        (10, 180),
        (255, 255, 0),
    )
    left_viz = draw_text(
        left_viz,
        f"ROI valid ratio: {valid_ratio * 100.0:.1f}%",
        (10, 195),
        (255, 255, 0),
    )
    left_viz = draw_text(
        left_viz,
        f"ROI disp>0: {positive_disp_pixels}/{total_pixels}",
        (10, 225),
        (255, 255, 0),
    )
    left_viz = draw_text(
        left_viz,
        f"ROI clipped(maxD): {clipped_by_max_depth}",
        (10, 255),
        (255, 255, 0),
    )

    if distance_raw is not None:
        left_viz = draw_text(left_viz, f"ROI Raw: {distance_raw * 1000.0:.1f} mm", (10, 285), (200, 255, 200))
    else:
        left_viz = draw_text(left_viz, "ROI Raw: N/A", (10, 285), (0, 0, 255))

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