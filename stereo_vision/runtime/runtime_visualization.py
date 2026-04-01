"""Runtime visualization builders for the stereo main loop."""

import time
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

from stereo_vision.core.rectification import rectify_pair
from stereo_vision.core.roi import ROI
from stereo_vision.ui.visualization import colorize_disparity, draw_roi, draw_text


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
    """Build the per-frame left overlay and colorized disparity image.

    This function renders the on-screen diagnostics (input source, performance,
    ROI quality metrics, and distance readouts) on top of the rectified left
    frame, and creates the colorized disparity panel used for side-by-side view.

    Args:
        left_rect: Rectified left frame used as overlay background.
        disparity: Disparity map for current frame.
        roi_scaled: Active ROI in processed-frame coordinates.
        min_disp: Lower bound for disparity visualization color scale.
        max_disp: Upper bound for disparity visualization color scale.
        distance_filtered: Smoothed ROI distance in meters.
        fps: Running frames-per-second estimate.
        latency_ms: Per-frame processing latency in milliseconds.
        active_idx: Zero-based active input index.
        device_list: Available input device paths.
        last_switch_latency_ms: Last measured input-switch latency (unused in overlay).
        switch_pending: Pending switch state (unused in overlay).
        last_switch_breakdown: Switch timing breakdown (unused in overlay).
        valid_pixels: Number of finite depth pixels in ROI.
        total_pixels: Total number of ROI pixels.
        valid_ratio: Fraction of finite depth pixels in ROI.
        positive_disp_pixels: Number of ROI pixels with disparity > 0.
        clipped_by_max_depth: Number of pixels clipped by max-depth threshold.
        distance_raw: Raw (unfiltered) ROI distance in meters.
        roi_gate_note: Optional gating/rejection note to show on overlay.
        roi_tune_preset: Active ROI tuning preset label.
        runtime_note: Optional transient runtime message.

    Returns:
        Tuple of (left overlay BGR frame, colorized disparity visualization).
    """
    disp_vis = colorize_disparity(
        disparity,
        min_disp=float(min_disp),
        max_disp=float(max_disp),
    )
    left_viz_src = left_rect
    if left_viz_src.ndim == 2 or (left_viz_src.ndim == 3 and left_viz_src.shape[2] == 1):
        left_viz_src = cv2.cvtColor(left_viz_src, cv2.COLOR_GRAY2BGR)
    left_viz = draw_roi(left_viz_src, roi_scaled)

    # Legacy switching metrics are intentionally hidden from the overlay.
    _ = (last_switch_latency_ms, switch_pending, last_switch_breakdown)

    valid_ratio_pct = valid_ratio * 100.0
    if distance_filtered is not None:
        dist_line = f"ROI Distance: {distance_filtered * 1000.0:.1f} mm"
        dist_color = (0, 255, 0)
    else:
        dist_line = "ROI Distance: N/A"
        dist_color = (0, 0, 255)

    # Instantaneous robust ROI distance from current frame before temporal smoothing.
    if distance_raw is not None:
        raw_line = f"ROI Raw: {distance_raw * 1000.0:.1f} mm"
        raw_color = (200, 255, 200)
    else:
        raw_line = "ROI Raw: N/A"
        raw_color = (0, 0, 255)

    # ROI Distance is the temporally smoothed estimate used as the stable user-facing readout.
    # ROI Raw is the current-frame robust estimate before temporal smoothing.
    dist_raw_line = f"{dist_line} | {raw_line}"
    dist_raw_color = dist_color if distance_filtered is not None else raw_color

    # Active stereo source index and device path currently used for capture.
    metric_input = f"Input: {active_idx + 1}/{len(device_list)} ({device_list[active_idx]})"
    # Throughput and per-frame end-to-end processing time for the current loop iteration.
    metric_perf = f"FPS: {fps:.1f} | Latency: {latency_ms:.1f} ms"
    # Runtime preset profile controlling ROI gating/smoothing behavior.
    metric_preset = f"ROI Preset: {roi_tune_preset}"
    # Depth validity in ROI: finite depth count and its percentage of all ROI pixels.
    metric_valid = f"ROI valid: {valid_pixels}/{total_pixels} ({valid_ratio_pct:.1f}%)"
    # Disparity health in ROI: positive disparity count and max-depth-clipped depth count.
    metric_disp_clip = f"ROI disp>0/clipped: {positive_disp_pixels}/{total_pixels} | {clipped_by_max_depth}"

    overlay_lines: list[tuple[str, tuple[int, int, int]]] = [
        (metric_input, (255, 255, 0)),
        (metric_perf, (255, 255, 0)),
        (dist_raw_line, dist_raw_color),
        (metric_preset, (255, 255, 0)),
        (metric_valid, (255, 255, 0)),
        (metric_disp_clip, (255, 255, 0)),
    ]
    if roi_gate_note:
        overlay_lines.append((f"ROI Gate: {roi_gate_note}", (0, 0, 255)))
    if runtime_note:
        overlay_lines.append((runtime_note, (0, 255, 0)))

    x0 = 10
    y = 28
    line_step = 22
    for text, color in overlay_lines:
        left_viz = draw_text(left_viz, text, (x0, y), color)
        y += line_step

    return left_viz, disp_vis


def apply_click_probe_overlay(
    left_viz: np.ndarray,
    click_px: Optional[Tuple[int, int]],
    click_depth: Optional[float],
) -> np.ndarray:
    """Draw click marker and sampled depth text for the last user click.

    The click marker is drawn at the clicked pixel location, and the sampled
    metric depth value is shown near the bottom of the frame to avoid overlap
    with the main metrics block.

    Args:
        left_viz: Left visualization frame to annotate.
        click_px: Last clicked pixel coordinate as (x, y), or None.
        click_depth: Sampled depth value at click location in meters, or None.

    Returns:
        Annotated left visualization frame.
    """
    if click_px is None:
        return left_viz

    cx, cy = click_px
    if 0 <= cx < left_viz.shape[1] and 0 <= cy < left_viz.shape[0]:
        cv2.circle(left_viz, (cx, cy), 5, (0, 0, 255), -1, cv2.LINE_AA)

    text_y = max(20, int(left_viz.shape[0]) - 20)
    if click_depth is not None:
        return draw_text(left_viz, f"Click({cx},{cy}): {click_depth * 1000.0:.1f} mm", (10, text_y), (255, 200, 0))
    return draw_text(left_viz, f"Click({cx},{cy}): N/A", (10, text_y), (0, 0, 255))


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
    """Compose final visualization layers used by the runtime display loop.

    Optionally uses preview-only BGR frames (when available) for better visual
    quality, falls back to the standard rectified image on conversion issues,
    then delegates overlay rendering to ``build_viz_layers``.

    Args:
        left_rect: Rectified left frame used by processing path.
        preview_pair_raw: Optional raw preview pair for BGR-first visualization.
        rect: Rectification model/parameters.
        preview_nv12_warned: Whether preview conversion warning was already logged.
        disparity: Current disparity map.
        roi_scaled: Active ROI in processed-frame coordinates.
        min_disp: Lower bound for disparity visualization color scale.
        max_disp: Upper bound for disparity visualization color scale.
        distance_filtered: Smoothed ROI distance in meters.
        fps: Running frames-per-second estimate.
        latency_ms: Per-frame processing latency in milliseconds.
        active_idx: Zero-based active input index.
        device_list: Available input device paths.
        switch_state: Runtime switch state object.
        valid_pixels: Number of finite depth pixels in ROI.
        total_pixels: Total number of ROI pixels.
        valid_ratio: Fraction of finite depth pixels in ROI.
        positive_disp_pixels: Number of ROI pixels with disparity > 0.
        clipped_by_max_depth: Number of pixels clipped by max-depth threshold.
        distance_raw: Raw (unfiltered) ROI distance in meters.
        roi_gate_note: Optional gating/rejection note.
        roi_tune_preset: Active ROI tuning preset label.
        runtime_note_text: Optional transient runtime message text.
        runtime_note_until: Monotonic timestamp until runtime note stays visible.

    Returns:
        Tuple of (left overlay frame, colorized disparity frame, preview warning flag).
    """
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