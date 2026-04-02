"""Main runtime frame loop for stereo processing and visualization."""

from __future__ import annotations

import time
from datetime import datetime
from argparse import Namespace
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from stereo_vision.app_cli import PerfStats, get_screen_size
from stereo_vision.core.depth import DepthEstimator
from stereo_vision.core.disparity import StereoDisparityEstimator
from stereo_vision.core.rectification import rectify_pair
from stereo_vision.core.roi import ROI
from stereo_vision.pipeline.optimization import RuntimeOptimizationConfig
from stereo_vision.pipeline.preprocess import FramePreprocessor
from stereo_vision.runtime.runtime_controls import process_runtime_key_events
from stereo_vision.runtime.runtime_processing import (
    compute_depth_and_distance,
    compute_disparity,
    compute_runtime_roi,
)
from stereo_vision.runtime.runtime_profile import StageProfiler
from stereo_vision.runtime.runtime_switching import (
    capture_active_frame_and_finalize,
    configure_switch_runtime_state,
    get_preview_pair_for_active_frame,
)
from stereo_vision.runtime.runtime_visualization import (
    apply_click_probe_overlay,
    compose_runtime_visualization,
)
from stereo_vision.runtime.runtime_tuning import RoiTuneController
from stereo_vision.ui.visualization import VizState, register_click


@dataclass(frozen=True)
class RuntimeLoopConfig:
    """Immutable runtime loop dependencies assembled at startup.

    Attributes:
        cam: Camera manager/capture backend used by the runtime loop.
        multi_mode: Whether multi-input switching mode is enabled.
        switch_timeout_s: Timeout waiting for first frame after input switch.
        active_idx: Initial active input index (0-based).
        rect: Stereo rectification model/parameters.
        runtime_cfg: Runtime optimization toggles (scale, ROI mode, etc.).
        preprocessor: Frame preprocessor (resize/grayscale pipeline).
        depth_estimator: Converts disparity to metric depth.
        disp_estimator: Stereo matcher used to compute disparity.
        requested_num_disp: Requested disparity search range from CLI.
        effective_num_disp: Adjusted/safe disparity search range in use.
        baseline_m: Camera baseline in meters.
        focal_px: Effective focal length in pixels.
        roi: Base ROI definition in processed-frame coordinates.
        roi_phys_w_m: Physical ROI width in meters.
        roi_phys_h_m: Physical ROI height in meters.
        roi_center_mode: Physical ROI centering policy.
        roi_depth_ref_m: Depth reference for physical ROI projection.
        device_list: Ordered list of configured capture device paths.
        roi_tuning: Runtime ROI tuning/filter controller.
        preview_nv12_bgr: Whether preview path should request BGR from NV12.
        profiler: Optional stage timing profiler used each frame.
    """

    cam: Any
    multi_mode: bool
    switch_timeout_s: float
    active_idx: int
    rect: Any
    runtime_cfg: RuntimeOptimizationConfig
    preprocessor: FramePreprocessor
    depth_estimator: DepthEstimator
    disp_estimator: StereoDisparityEstimator
    requested_num_disp: int
    effective_num_disp: int
    baseline_m: float
    focal_px: float
    roi: ROI
    roi_phys_w_m: float
    roi_phys_h_m: float
    roi_center_mode: str
    roi_depth_ref_m: float
    device_list: list[str]
    roi_tuning: RoiTuneController
    preview_nv12_bgr: bool
    profiler: StageProfiler


def run_runtime_loop(
    *,
    args: Namespace,
    cfg: RuntimeLoopConfig,
) -> None:
    """Run the per-frame processing loop until quit.

    Args:
        args: Parsed CLI/runtime arguments controlling processing and UI behavior.
        cfg: Prebuilt immutable runtime dependencies and calibration values.

    Returns:
        None. The function runs until user exit and updates UI in real time.
    """
    cam = cfg.cam
    multi_mode = cfg.multi_mode
    switch_timeout_s = cfg.switch_timeout_s
    active_idx = cfg.active_idx
    rect = cfg.rect
    runtime_cfg = cfg.runtime_cfg
    preprocessor = cfg.preprocessor
    depth_estimator = cfg.depth_estimator
    disp_estimator = cfg.disp_estimator
    requested_num_disp = cfg.requested_num_disp
    effective_num_disp = cfg.effective_num_disp
    baseline_m = cfg.baseline_m
    focal_px = cfg.focal_px
    roi = cfg.roi
    roi_phys_w_m = cfg.roi_phys_w_m
    roi_phys_h_m = cfg.roi_phys_h_m
    roi_center_mode = cfg.roi_center_mode
    roi_depth_ref_m = cfg.roi_depth_ref_m
    device_list = cfg.device_list
    roi_tuning = cfg.roi_tuning
    preview_nv12_bgr = cfg.preview_nv12_bgr
    profiler = cfg.profiler

    swap_lr = bool(args.swap_lr)
    preview_nv12_warned = False
    runtime_note_text: str | None = None
    runtime_note_until = 0.0

    show_display = not bool(getattr(args, "no_display", False))
    win = "stereo_distance"
    if show_display:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    viz_state = VizState()
    if show_display:
        register_click(win, viz_state)
        print("[INFO] Keyboard control is captured by the video window. Click the window before pressing 1-4/n/s/z/x/c/v/q.")
    else:
        print("[INFO] Display disabled (--no-display). Running in log-only headless mode.")

    perf = PerfStats()
    window_sized = False
    screen_size = get_screen_size() if show_display else None
    switch_state = configure_switch_runtime_state(
        multi_mode=multi_mode,
        cam=cam,
        switch_timeout_s=switch_timeout_s,
    )
    log_measurements = bool(getattr(args, "log_measurements", False))
    log_interval_s = max(0.0, float(getattr(args, "log_interval_ms", 250.0)) / 1000.0)
    log_file_path = str(getattr(args, "log_file", "") or "").strip()
    last_log_t = 0.0
    log_file = None
    if log_measurements:
        print(f"[INFO] Measurement logging enabled (interval={log_interval_s * 1000.0:.0f}ms)")
        if log_file_path:
            log_file = open(log_file_path, "a", encoding="utf-8", buffering=1)
            if log_file.tell() == 0:
                log_file.write(
                    "timestamp,distance_m,distance_raw_m,fps,latency_ms,input_idx,probe_x,probe_y,probe_depth_m\n"
                )
            print(f"[INFO] Measurement CSV output: {log_file_path}")

    # Main real-time loop: capture -> rectify -> preprocess -> disparity -> depth -> visualize.
    try:
        while True:
            t0 = time.perf_counter()
            # 1) Capture and optional channel swap.
            try:
                left, right, frame_ts, active_idx, t_capture = capture_active_frame_and_finalize(
                    cam=cam,
                    multi_mode=multi_mode,
                    state=switch_state,
                )
            except RuntimeError:
                time.sleep(0.01)
                continue

            preview_pair_raw = get_preview_pair_for_active_frame(
                cam=cam,
                multi_mode=multi_mode,
                preview_nv12_bgr=preview_nv12_bgr,
                active_idx=active_idx,
                frame_ts=frame_ts,
            )

            if swap_lr:
                left, right = right, left
                if preview_pair_raw is not None:
                    preview_pair_raw = (preview_pair_raw[1], preview_pair_raw[0])

            # 2) Geometric rectification: align epipolar lines for valid stereo matching.
            left_rect, right_rect = rectify_pair(left, right, rect)
            t_rectify = time.perf_counter()

            # 3) Runtime preprocess (resize + grayscale) to match disparity input expectations.
            left_rect, right_rect, gray_l, gray_r = preprocessor.process(left_rect, right_rect)
            t_preprocess = time.perf_counter()

            # Derive frame-valid ROI (optionally projected from physical-size target window).
            roi_scaled = compute_runtime_roi(
                gray_shape=gray_l.shape[:2],
                roi=roi,
                runtime_cfg=runtime_cfg,
                roi_center_mode=roi_center_mode,
                roi_phys_w_m=roi_phys_w_m,
                roi_phys_h_m=roi_phys_h_m,
                roi_depth_ref_m=roi_depth_ref_m,
                focal_px=focal_px,
            )

            # 4) Disparity estimation either on full frame or ROI-constrained mode.
            disparity, disp_estimator, _ = compute_disparity(
                gray_l=gray_l,
                gray_r=gray_r,
                roi_scaled=roi_scaled,
                runtime_cfg=runtime_cfg,
                args=args,
                requested_num_disp=requested_num_disp,
                disp_estimator=disp_estimator,
            )
            t_disparity = time.perf_counter()

            # 5) Convert disparity to metric depth and gather ROI quality/readout metrics.
            (
                depth_map,
                valid_pixels,
                total_pixels,
                valid_ratio,
                positive_disp_pixels,
                clipped_by_max_depth,
                distance_raw,
                distance_filtered,
                roi_gate_note,
            ) = compute_depth_and_distance(
                disparity=disparity,
                depth_estimator=depth_estimator,
                roi_scaled=roi_scaled,
                args=args,
                focal_px=focal_px,
                baseline_m=baseline_m,
                roi_tuning=roi_tuning,
            )

            # Update ROI projection depth from the current-frame robust estimate first.
            # The filtered value is kept for display, but using it here can freeze ROI
            # size when the filter rejects a real scene change as a jump.
            if distance_raw is not None and np.isfinite(distance_raw) and distance_raw > 0:
                roi_depth_ref_m = float(distance_raw)
            elif distance_filtered is not None and np.isfinite(distance_filtered) and distance_filtered > 0:
                roi_depth_ref_m = float(distance_filtered)

            if viz_state.clicked_px is None:
                # Default probe to image center until user selects another pixel.
                viz_state.clicked_px = (int(gray_l.shape[1] // 2), int(gray_l.shape[0] // 2))

            click_px = viz_state.clicked_px
            click_depth = None
            if click_px is not None:
                click_depth = depth_estimator.depth_at(depth_map, click_px[0], click_px[1])

            fps = perf.update_fps()
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # Emit periodic timestamped distance/FPS samples for external observation.
            if log_measurements and (t0 - last_log_t) >= log_interval_s:
                ts_text = datetime.now().isoformat(timespec="milliseconds")
                distance_m = "nan"
                if distance_filtered is not None and np.isfinite(distance_filtered):
                    distance_m = f"{float(distance_filtered):.6f}"
                distance_raw_m = "nan"
                if distance_raw is not None and np.isfinite(distance_raw):
                    distance_raw_m = f"{float(distance_raw):.6f}"
                probe_x = "-1"
                probe_y = "-1"
                if click_px is not None:
                    probe_x = str(int(click_px[0]))
                    probe_y = str(int(click_px[1]))
                probe_depth_m = "nan"
                if click_depth is not None and np.isfinite(click_depth):
                    probe_depth_m = f"{float(click_depth):.6f}"
                sample_line = (
                    f"{ts_text},fps={fps:.2f},latency_ms={latency_ms:.2f},input_idx={active_idx + 1},"
                    f"distance_m={distance_m},distance_raw_m={distance_raw_m},"
                    f"probe=({probe_x},{probe_y}),probe_depth_m={probe_depth_m}"
                )
                print(f"[MEASURE] {sample_line}")
                if log_file is not None:
                    log_file.write(
                        f"{ts_text},{active_idx + 1},{fps:.2f},{latency_ms:.2f},{distance_m},{distance_raw_m},{probe_x},{probe_y},{probe_depth_m}\n"
                    )
                last_log_t = t0

            # 7) Build visualization layers and on-screen diagnostics.
            t_depth = time.perf_counter()
            t_viz = t_depth
            if show_display:
                left_viz, disp_vis, preview_nv12_warned = compose_runtime_visualization(
                    left_rect=left_rect,
                    preview_pair_raw=preview_pair_raw,
                    rect=rect,
                    preview_nv12_warned=preview_nv12_warned,
                    disparity=disparity,
                    roi_scaled=roi_scaled,
                    min_disp=float(args.min_disp),
                    max_disp=float(args.min_disp + effective_num_disp),
                    distance_filtered=distance_filtered,
                    fps=fps,
                    latency_ms=latency_ms,
                    active_idx=active_idx,
                    device_list=device_list,
                    switch_state=switch_state,
                    valid_pixels=valid_pixels,
                    total_pixels=total_pixels,
                    valid_ratio=valid_ratio,
                    positive_disp_pixels=positive_disp_pixels,
                    clipped_by_max_depth=clipped_by_max_depth,
                    distance_raw=distance_raw,
                    roi_gate_note=roi_gate_note,
                    roi_tune_preset=roi_tuning.roi_tune_preset,
                    runtime_note_text=runtime_note_text,
                    runtime_note_until=runtime_note_until,
                )
                t_depth = time.perf_counter()

                # Optional per-pixel inspection from last mouse click.
                left_viz = apply_click_probe_overlay(
                    left_viz,
                    click_px=click_px,
                    click_depth=click_depth,
                )

                stack = np.hstack([left_viz, disp_vis])
                if not window_sized:
                    win_w, win_h = int(stack.shape[1]), int(stack.shape[0])
                    cv2.resizeWindow(win, win_w, win_h)
                    if screen_size is not None:
                        screen_w, screen_h = screen_size
                        pos_x = max(0, (screen_w - win_w) // 2)
                        pos_y = max(0, (screen_h - win_h) // 2)
                        cv2.moveWindow(win, pos_x, pos_y)
                    window_sized = True
                cv2.imshow(win, stack)
                t_viz = time.perf_counter()

            # Persist per-stage timings and emit a periodic profile line when enabled.
            profile_line = profiler.record(
                t0=t0,
                t_capture=t_capture,
                t_rectify=t_rectify,
                t_preprocess=t_preprocess,
                t_disparity=t_disparity,
                t_depth=t_depth,
                t_viz=t_viz,
            )
            if profile_line is not None:
                print(profile_line)

            # Handle keyboard controls for source switching, presets, and shutdown.
            if show_display:
                key_raw = cv2.waitKeyEx(1)
                swap_lr, key_note_text, key_note_until, should_exit = process_runtime_key_events(
                    key_raw=key_raw,
                    swap_lr=swap_lr,
                    multi_mode=multi_mode,
                    device_list=device_list,
                    active_idx=active_idx,
                    cam=cam,
                    switch_state=switch_state,
                    apply_roi_tune_preset=roi_tuning.apply_preset,
                )
                if key_note_text is not None:
                    runtime_note_text = key_note_text
                    runtime_note_until = key_note_until

                if should_exit:
                    break
    finally:
        if log_file is not None:
            log_file.close()
