#!/usr/bin/env python3
"""Entry point for real-time stereo distance measurement and visualization.

This script wires together capture, rectification, disparity estimation,
depth conversion, robust ROI distance extraction, temporal smoothing,
and live OpenCV visualization.
"""

import time
from typing import Optional

import cv2
import numpy as np

from stereo_vision.app_cli import (
    PerfStats,
    decode_switch_index,
    fourcc_to_str,
    get_screen_size,
    parse_args,
    parse_roi,
    resolve_baseline_m,
    safe_num_disparities_for_roi,
)
from stereo_vision.calibration import load_stereo_calibration
from stereo_vision.camera import MultiStereoCamera
from stereo_vision.depth import DepthConfig, DepthEstimator
from stereo_vision.disparity import SGBMConfig, StereoDisparityEstimator
from stereo_vision.filters import DistanceFilter, TemporalFilterConfig
from stereo_vision.optimization import RuntimeOptimizationConfig, crop_for_disparity
from stereo_vision.preprocess import FramePreprocessor, PreprocessConfig
from stereo_vision.rectification import build_rectification_maps, rectify_pair
from stereo_vision.roi import ROI, robust_roi_distance
from stereo_vision.startup import initialize_capture
from stereo_vision.visualization import VizState, colorize_disparity, draw_roi, draw_text, register_click


def _quiet_opencv_logs() -> None:
    """Lower OpenCV logging verbosity in a version-compatible way."""
    try:
        if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
            level = getattr(cv2.utils.logging, "LOG_LEVEL_ERROR", None)
            if level is not None:
                cv2.utils.logging.setLogLevel(level)
                return
        if hasattr(cv2, "setLogLevel"):
            level = getattr(cv2, "LOG_LEVEL_ERROR", None)
            if level is None:
                level = 2
            cv2.setLogLevel(level)
    except Exception:
        # Logging control differs between OpenCV builds; ignore unsupported APIs.
        return


def main() -> None:
    """Run the full stereo pipeline loop and render diagnostic overlays.

    The loop sequence is:
    capture -> rectify -> resize -> disparity -> depth -> ROI measurement
    -> temporal filtering -> visualization.
    """
    args = parse_args()
    if bool(getattr(args, "quiet_opencv_log", False)):
        _quiet_opencv_logs()

    device_list = [d.strip() for d in str(args.devices).split(",") if d.strip()]
    if len(device_list) == 0:
        device_list = [str(args.device)]
    if len(device_list) != 4:
        print(
            f"[WARN] Expected 4 stereo inputs for deployment goal, got {len(device_list)}. "
            "System will run with available inputs."
        )

    startup = initialize_capture(args, device_list)
    cam = startup.cam
    multi_mode = startup.multi_mode
    switch_timeout_s = startup.switch_timeout_s
    active_idx = startup.active_idx
    left0 = startup.left0
    right0 = startup.right0

    calib = load_stereo_calibration(args.calib)

    # Prime the camera once to discover frame geometry and stream properties.
    image_size = (left0.shape[1], left0.shape[0])

    if multi_mode:
        print(
            f"[INFO] Active input device: idx={active_idx + 1}, device={device_list[active_idx]}"
        )
    else:
        cap = cam.cap
        if cap is not None:
            stream_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            stream_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            stream_fps = float(cap.get(cv2.CAP_PROP_FPS))
            stream_fourcc = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
            print(
                f"[INFO] Video stream: {stream_w}x{stream_h}, "
                f"FPS={stream_fps:.2f}, FOURCC={stream_fourcc}"
            )

    # Build remap tables once; remapping is then cheap per frame.
    rect = build_rectification_maps(
        calib,
        image_size,
        use_precomputed=bool(args.use_precomputed_rect),
    )

    scale = float(args.scale)
    if not 0.25 <= scale <= 1.0:
        raise ValueError("--scale should be in [0.25, 1.0]")

    # Use rectified focal length and runtime scale for depth conversion.
    focal_px = float(rect.p1[0, 0]) * scale
    baseline_raw = float(calib.baseline_m)
    baseline_m = resolve_baseline_m(baseline_raw, str(args.baseline_unit))
    print(f"[INFO] baseline_raw={baseline_raw:.6f}, baseline_m={baseline_m:.6f}, unit={args.baseline_unit}")

    # Keep both original ROI (CLI scale) and scaled ROI (runtime scale).
    roi = parse_roi(args.roi)
    scaled_w = max(1, int(left0.shape[1] * scale))
    scaled_h = max(1, int(left0.shape[0] * scale))
    roi_scaled_init = ROI(
        x=int(roi.x * scale),
        y=int(roi.y * scale),
        w=int(roi.w * scale),
        h=int(roi.h * scale),
    ).clamp(scaled_w, scaled_h)

    requested_num_disp = int(args.num_disp)
    effective_num_disp = requested_num_disp
    if bool(args.roi_disparity_only):
        effective_num_disp = safe_num_disparities_for_roi(requested_num_disp, roi_scaled_init.w)
        if effective_num_disp != requested_num_disp:
            print(
                "[INFO] Adjusted num-disp for ROI mode: "
                f"requested={requested_num_disp}, effective={effective_num_disp}, roi_w={roi_scaled_init.w}"
            )

    # Configure depth, disparity, and filtering stages.
    depth_estimator = DepthEstimator(
        DepthConfig(
            focal_px=focal_px,
            baseline_m=baseline_m,
            min_disparity=max(0.01, float(args.depth_min_disp)),
            max_depth_m=float(args.max_depth),
        )
    )

    disp_estimator = StereoDisparityEstimator(
        SGBMConfig(
            min_disparity=int(args.min_disp),
            num_disparities=effective_num_disp,
            block_size=int(args.block_size),
            p1=8 * 1 * int(args.block_size) * int(args.block_size),
            p2=32 * 1 * int(args.block_size) * int(args.block_size),
            uniqueness_ratio=10,
            speckle_window_size=80,
            speckle_range=2,
        )
    )

    distance_filter = DistanceFilter(
        TemporalFilterConfig(
            ema_alpha=float(args.ema_alpha),
            max_jump_m=float(args.max_jump),
            window=int(args.filter_window),
        )
    )

    runtime_cfg = RuntimeOptimizationConfig(
        scale=scale,
        disparity_roi_only=bool(args.roi_disparity_only),
    )
    preprocess_backend = str(args.preprocess_backend)
    if multi_mode and preprocess_backend != "cpu" and not bool(args.allow_unsafe_rga_multicam):
        print(
            "[WARN] Multi-camera mode detected: forcing preprocess backend to CPU "
            "to avoid RGA switch instability. Use --allow-unsafe-rga-multicam to override."
        )
        preprocess_backend = "cpu"

    preprocessor = FramePreprocessor(
        PreprocessConfig(
            scale=runtime_cfg.scale,
            backend=preprocess_backend,
            rga_module=str(args.rga_module),
        )
    )
    print(f"[INFO] preprocess_backend={preprocessor.backend_name}")
    print(f"[INFO] preprocess_detail={preprocessor.backend_reason}")

    swap_lr = bool(args.swap_lr)
    viz_state = VizState()

    win = "stereo_distance"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    register_click(win, viz_state)
    print("[INFO] Keyboard control is captured by the video window. Click the window before pressing 1-4/n/q.")

    perf = PerfStats()
    window_sized = False
    screen_size = get_screen_size()
    profile_stages = bool(getattr(args, "profile_stages", False))
    profile_interval = max(1, int(getattr(args, "profile_interval", 60)))
    stage_acc = {
        "capture": 0.0,
        "rectify": 0.0,
        "preprocess": 0.0,
        "disparity": 0.0,
        "depth": 0.0,
        "viz": 0.0,
        "total": 0.0,
    }
    stage_count = 0
    last_switch_latency_ms: Optional[float] = None
    last_switch_breakdown: Optional[dict] = None
    switch_pending: Optional[dict] = None
    last_good_frame: Optional[tuple[np.ndarray, np.ndarray, float, int]] = None
    switch_runtime_timeout_s = switch_timeout_s
    if multi_mode and isinstance(cam, MultiStereoCamera) and cam.single_active_mode and switch_runtime_timeout_s < 3.0:
        print(
            "[WARN] Single-active mode detected with slow camera restart characteristics; "
            f"raising runtime switch timeout from {switch_runtime_timeout_s:.2f}s to 3.00s"
        )
        switch_runtime_timeout_s = 3.0

    read_timeout_idle_s = min(0.05, switch_runtime_timeout_s)
    read_timeout_switch_s = min(0.20, switch_runtime_timeout_s)
    fallback_max_age_s = max(0.12, min(0.4, switch_runtime_timeout_s))

    def read_active_frame() -> tuple[np.ndarray, np.ndarray, float, int]:
        """Read current active source frame and return source index for overlays."""
        if multi_mode:
            min_ts = 0.0
            timeout = read_timeout_idle_s
            allow_fallback = True
            if switch_pending is not None:
                pending_t0 = float(switch_pending["t0"])
                min_ts = pending_t0
                timeout = read_timeout_switch_s
                # During a pending switch, wait for target fresh frame only.
                allow_fallback = False
            l, r, ts, _, idx = cam.read(
                timeout_s=timeout,
                min_timestamp_s=min_ts,
                allow_fallback=allow_fallback,
                max_fallback_age_s=fallback_max_age_s,
            )
            return l, r, ts, idx
        l, r, ts = cam.read()
        return l, r, ts, 0

    try:
        while True:
            t0 = time.perf_counter()
            # 1) Capture and optional channel swap.
            try:
                left, right, frame_ts, active_idx = read_active_frame()
                last_good_frame = (left, right, frame_ts, active_idx)
                if multi_mode and isinstance(cam, MultiStereoCamera):
                    breakdown = cam.get_last_switch_breakdown()
                    if breakdown is not None:
                        last_switch_breakdown = breakdown
            except RuntimeError:
                now = time.perf_counter()

                if switch_pending is not None:
                    pending_idx = int(switch_pending["to_idx"])
                    pending_t0 = float(switch_pending["t0"])
                    if (now - pending_t0) >= switch_runtime_timeout_s:
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
                        switch_pending = None

                if last_good_frame is None:
                    time.sleep(0.01)
                    continue
                left, right, frame_ts, active_idx = last_good_frame
            t_capture = time.perf_counter()

            if switch_pending is not None:
                pending_idx = int(switch_pending["to_idx"])
                pending_from_idx = int(switch_pending["from_idx"])
                pending_t0 = float(switch_pending["t0"])
                if pending_idx == active_idx and frame_ts >= pending_t0:
                    last_switch_latency_ms = (time.perf_counter() - pending_t0) * 1000.0
                    if multi_mode and isinstance(cam, MultiStereoCamera):
                        breakdown = cam.get_last_switch_breakdown()
                        if breakdown is not None:
                            last_switch_breakdown = breakdown
                    if last_switch_breakdown is not None:
                        print(
                            "[INFO] Switch complete: "
                            f"from_input={pending_from_idx + 1}, to_input={active_idx + 1}, "
                            f"latency={last_switch_latency_ms:.1f}ms, "
                            f"seg_stop={float(last_switch_breakdown.get('stop_ms', 0.0)):.0f}ms, "
                            f"seg_open={float(last_switch_breakdown.get('open_ms', 0.0)):.0f}ms, "
                            f"seg_frame={float(last_switch_breakdown.get('first_frame_ms', 0.0)):.0f}ms, "
                            f"seg_total={float(last_switch_breakdown.get('total_ms', 0.0)):.0f}ms"
                        )
                    else:
                        print(
                            "[INFO] Switch complete: "
                            f"from_input={pending_from_idx + 1}, to_input={active_idx + 1}, "
                            f"latency={last_switch_latency_ms:.1f}ms"
                        )
                    switch_pending = None
                elif (time.perf_counter() - pending_t0) >= switch_runtime_timeout_s:
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
                    switch_pending = None
            if swap_lr:
                left, right = right, left

            # 2) Geometric rectification.
            left_rect, right_rect = rectify_pair(left, right, rect)
            t_rectify = time.perf_counter()

            # 3) Runtime preprocess (resize + grayscale), optionally RGA-backed.
            left_rect, right_rect, gray_l, gray_r = preprocessor.process(left_rect, right_rect)
            t_preprocess = time.perf_counter()

            roi_scaled = ROI(
                x=int(roi.x * runtime_cfg.scale),
                y=int(roi.y * runtime_cfg.scale),
                w=int(roi.w * runtime_cfg.scale),
                h=int(roi.h * runtime_cfg.scale),
            ).clamp(gray_l.shape[1], gray_l.shape[0])

            # 4) Disparity either on full frame or constrained ROI.
            if runtime_cfg.disparity_roi_only:
                crop_l, crop_r, roi_used = crop_for_disparity(gray_l, gray_r, roi_scaled)
                try:
                    disp_crop = disp_estimator.compute(crop_l, crop_r)
                except cv2.error as exc:
                    raise RuntimeError(
                        "StereoSGBM failed in ROI mode. Increase ROI width or lower --num-disp. "
                        f"Current ROI width={roi_used.w}, effective num-disp={effective_num_disp}."
                    ) from exc
                # Keep full-frame shape so downstream ROI/depth code stays identical.
                disparity = np.full(gray_l.shape, -1.0, dtype=np.float32)
                ys, xs = roi_used.as_slice()
                disparity[ys, xs] = disp_crop
            else:
                disparity = disp_estimator.compute(gray_l, gray_r)
            t_disparity = time.perf_counter()

            # 5) Convert disparity to metric depth and gather ROI stats.
            depth_map = depth_estimator.disparity_to_depth(disparity)
            ys, xs = roi_scaled.as_slice()
            roi_disp = disparity[ys, xs]
            roi_depth = depth_map[ys, xs]
            positive_disp_pixels = int((roi_disp > 0.0).sum())
            valid_pixels = int(np.isfinite(roi_depth).sum())
            total_pixels = int(roi_depth.size)

            roi_depth_raw = np.full(roi_disp.shape, np.nan, dtype=np.float32)
            raw_valid = roi_disp > max(0.01, float(args.depth_min_disp))
            roi_depth_raw[raw_valid] = (focal_px * baseline_m) / roi_disp[raw_valid]
            clipped_by_max_depth = 0
            if float(args.max_depth) > 0:
                # Difference between raw-valid and final-valid indicates depth clipping.
                clipped_by_max_depth = int(
                    np.isfinite(roi_depth_raw).sum() - np.isfinite(roi_depth).sum()
                )

            # 6) Robust distance estimate + temporal smoothing.
            distance_raw = robust_roi_distance(
                depth_map,
                roi_scaled,
                min_valid_pixels=max(1, int(args.min_valid_pixels)),
            )
            distance_filtered = distance_filter.update(distance_raw)

            fps = perf.update_fps()
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # 7) Build visualization layers and on-screen diagnostics.
            disp_vis = colorize_disparity(
                disparity,
                min_disp=float(args.min_disp),
                max_disp=float(args.min_disp + effective_num_disp),
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
                f"ROI disp>0: {positive_disp_pixels}/{total_pixels}",
                (10, 210),
                (255, 255, 0),
            )
            left_viz = draw_text(
                left_viz,
                f"ROI clipped(maxD): {clipped_by_max_depth}",
                (10, 240),
                (255, 255, 0),
            )

            if distance_raw is not None:
                left_viz = draw_text(left_viz, f"ROI Raw: {distance_raw * 1000.0:.1f} mm", (10, 270), (200, 255, 200))
            else:
                left_viz = draw_text(left_viz, "ROI Raw: N/A", (10, 270), (0, 0, 255))
            t_depth = time.perf_counter()

            # Optional per-pixel inspection from last mouse click.
            if viz_state.clicked_px is not None:
                cx, cy = viz_state.clicked_px
                if 0 <= cx < left_viz.shape[1] and 0 <= cy < left_viz.shape[0]:
                    cv2.circle(left_viz, (cx, cy), 5, (0, 0, 255), -1, cv2.LINE_AA)
                click_depth: Optional[float] = depth_estimator.depth_at(depth_map, cx, cy)
                if click_depth is not None:
                    left_viz = draw_text(left_viz, f"Click({cx},{cy}): {click_depth * 1000.0:.1f} mm", (10, 300), (255, 200, 0))
                else:
                    left_viz = draw_text(left_viz, f"Click({cx},{cy}): N/A", (10, 300), (0, 0, 255))

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

            if profile_stages:
                stage_acc["capture"] += (t_capture - t0) * 1000.0
                stage_acc["rectify"] += (t_rectify - t_capture) * 1000.0
                stage_acc["preprocess"] += (t_preprocess - t_rectify) * 1000.0
                stage_acc["disparity"] += (t_disparity - t_preprocess) * 1000.0
                stage_acc["depth"] += (t_depth - t_disparity) * 1000.0
                stage_acc["viz"] += (t_viz - t_depth) * 1000.0
                stage_acc["total"] += (t_viz - t0) * 1000.0
                stage_count += 1
                if stage_count >= profile_interval:
                    print(
                        "[PROFILE] avg_ms "
                        f"capture={stage_acc['capture'] / stage_count:.2f} "
                        f"rectify={stage_acc['rectify'] / stage_count:.2f} "
                        f"preprocess={stage_acc['preprocess'] / stage_count:.2f} "
                        f"disparity={stage_acc['disparity'] / stage_count:.2f} "
                        f"depth={stage_acc['depth'] / stage_count:.2f} "
                        f"viz={stage_acc['viz'] / stage_count:.2f} "
                        f"total={stage_acc['total'] / stage_count:.2f}"
                    )
                    stage_acc = {k: 0.0 for k in stage_acc}
                    stage_count = 0

            # Keyboard shortcuts: s toggles left/right swap, 1-9 switches input, n selects next input, q/esc exits.
            key_raw = cv2.waitKeyEx(1)
            key = key_raw & 0xFF
            if key == ord("s"):
                swap_lr = not swap_lr

            if multi_mode:
                req_idx = decode_switch_index(key_raw, len(device_list))
                if req_idx is not None:
                    if req_idx != active_idx:
                        active_applied = cam.switch_to(req_idx)
                        t_sw = time.perf_counter()
                        switch_pending = {
                            "from_idx": int(active_idx),
                            "to_idx": int(active_applied),
                            "t0": float(t_sw),
                        }
                        print(
                            "[INFO] Switch request: "
                            f"from_input={active_idx + 1}, to_input={active_applied + 1}, key=number"
                        )

            if multi_mode and key == ord("n"):
                next_idx = (active_idx + 1) % len(device_list)
                active_applied = cam.switch_to(next_idx)
                t_sw = time.perf_counter()
                switch_pending = {
                    "from_idx": int(active_idx),
                    "to_idx": int(active_applied),
                    "t0": float(t_sw),
                }
                print(
                    "[INFO] Switch request: "
                    f"from_input={active_idx + 1}, to_input={active_applied + 1}, key=next"
                )
            if key == 27 or key == ord("q"):
                break

    finally:
        if cam is not None:
            cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
