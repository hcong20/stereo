#!/usr/bin/env python3
"""Entry point for real-time stereo distance measurement and visualization.

This script wires together capture, rectification, disparity estimation,
depth conversion, robust ROI distance extraction, temporal smoothing,
and live OpenCV visualization.
"""

import time

import cv2
import numpy as np

from stereo_vision.app_cli import (
    PerfStats,
    decode_switch_index,
    fourcc_to_str,
    get_screen_size,
    parse_physical_size_mm,
    parse_args,
    parse_roi,
    resolve_baseline_m,
    safe_num_disparities_for_roi,
)
from stereo_vision.calibration import load_stereo_calibration
from stereo_vision.depth import DepthConfig, DepthEstimator
from stereo_vision.disparity import SGBMConfig, StereoDisparityEstimator
from stereo_vision.filters import DistanceFilter, TemporalFilterConfig
from stereo_vision.optimization import RuntimeOptimizationConfig, crop_for_disparity
from stereo_vision.preprocess import FramePreprocessor, PreprocessConfig
from stereo_vision.rectification import build_rectification_maps, rectify_pair
from stereo_vision.roi import ROI, robust_roi_distance, roi_from_physical_size
from stereo_vision.runtime_profile import StageProfiler
from stereo_vision.runtime_switching import (
    configure_switch_runtime_state,
    finalize_pending_switch,
    read_active_frame,
    recover_frame_after_read_error,
    request_switch,
    update_switch_breakdown_snapshot,
)
from stereo_vision.runtime_visualization import apply_click_probe_overlay, build_viz_layers
from stereo_vision.startup import initialize_capture
from stereo_vision.visualization import VizState, register_click


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

    # Keep a static ROI for compatibility and optional center reference.
    roi = parse_roi(args.roi)
    roi_phys_w_m, roi_phys_h_m = parse_physical_size_mm(str(args.roi_physical_size_mm))
    roi_center_mode = str(args.roi_physical_center)
    roi_depth_ref_m = 1.0
    print(
        "[INFO] Physical ROI enabled: "
        f"{roi_phys_w_m * 100.0:.1f}cm x {roi_phys_h_m * 100.0:.1f}cm, center={roi_center_mode}"
    )

    scaled_w = max(1, int(left0.shape[1] * scale))
    scaled_h = max(1, int(left0.shape[0] * scale))
    roi_scaled_static = ROI(
        x=int(roi.x * scale),
        y=int(roi.y * scale),
        w=int(roi.w * scale),
        h=int(roi.h * scale),
    ).clamp(scaled_w, scaled_h)

    if roi_center_mode == "image-center":
        init_cx = scaled_w // 2
        init_cy = scaled_h // 2
    else:
        init_cx = roi_scaled_static.x + (roi_scaled_static.w // 2)
        init_cy = roi_scaled_static.y + (roi_scaled_static.h // 2)

    roi_scaled_init = roi_from_physical_size(
        img_w=scaled_w,
        img_h=scaled_h,
        center_x=init_cx,
        center_y=init_cy,
        width_m=roi_phys_w_m,
        height_m=roi_phys_h_m,
        depth_m=roi_depth_ref_m,
        fx=focal_px,
    )

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
    active_num_disp = effective_num_disp

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

    preprocessor = FramePreprocessor(
        PreprocessConfig(
            scale=runtime_cfg.scale,
        )
    )
    print(f"[INFO] preprocess_backend={preprocessor.backend_name}")
    print(f"[INFO] preprocess_detail={preprocessor.backend_reason}")
    print(
        "[INFO] ROI distance fusion: "
        f"valid_ratio_min={float(args.roi_valid_ratio_min):.2f}, "
        f"p10_weight={float(args.roi_p10_weight):.2f}, "
        f"min_weight={float(args.roi_min_weight):.2f}"
    )
    preview_nv12_bgr = bool(getattr(args, "nv12_preview_bgr", False))
    if preview_nv12_bgr:
        print("[INFO] NV12 preview color mode enabled (preview-only BGR conversion)")
    preview_nv12_warned = False

    swap_lr = bool(args.swap_lr)
    viz_state = VizState()

    win = "stereo_distance"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    register_click(win, viz_state)
    print("[INFO] Keyboard control is captured by the video window. Click the window before pressing 1-4/n/q.")

    perf = PerfStats()
    window_sized = False
    screen_size = get_screen_size()
    profiler = StageProfiler(
        enabled=bool(getattr(args, "profile_stages", False)),
        interval=max(1, int(getattr(args, "profile_interval", 60))),
    )
    switch_state = configure_switch_runtime_state(
        multi_mode=multi_mode,
        cam=cam,
        switch_timeout_s=switch_timeout_s,
    )

    try:
        while True:
            t0 = time.perf_counter()
            # 1) Capture and optional channel swap.
            try:
                left, right, frame_ts, active_idx = read_active_frame(
                    cam=cam,
                    multi_mode=multi_mode,
                    state=switch_state,
                )
                switch_state.last_good_frame = (left, right, frame_ts, active_idx)
                update_switch_breakdown_snapshot(multi_mode, cam, switch_state)
            except RuntimeError:
                recovered = recover_frame_after_read_error(
                    cam=cam,
                    state=switch_state,
                )
                if recovered is None:
                    time.sleep(0.01)
                    continue
                left, right, frame_ts, active_idx = recovered
            t_capture = time.perf_counter()
            finalize_pending_switch(
                cam=cam,
                multi_mode=multi_mode,
                state=switch_state,
                active_idx=active_idx,
                frame_ts=frame_ts,
            )

            preview_pair_raw = None
            if preview_nv12_bgr:
                if multi_mode and hasattr(cam, "get_latest_preview"):
                    preview_packet = cam.get_latest_preview(active_idx)
                    if preview_packet is not None:
                        p_left, p_right, p_ts, _, p_idx = preview_packet
                        if int(p_idx) == int(active_idx) and abs(float(p_ts) - float(frame_ts)) <= 0.2:
                            preview_pair_raw = (p_left, p_right)
                elif hasattr(cam, "get_last_preview_pair"):
                    preview_pair_raw = cam.get_last_preview_pair()

            if swap_lr:
                left, right = right, left
                if preview_pair_raw is not None:
                    preview_pair_raw = (preview_pair_raw[1], preview_pair_raw[0])

            # 2) Geometric rectification.
            left_rect, right_rect = rectify_pair(left, right, rect)
            t_rectify = time.perf_counter()

            # 3) Runtime preprocess (resize + grayscale).
            left_rect, right_rect, gray_l, gray_r = preprocessor.process(left_rect, right_rect)
            t_preprocess = time.perf_counter()

            roi_scaled_static = ROI(
                x=int(roi.x * runtime_cfg.scale),
                y=int(roi.y * runtime_cfg.scale),
                w=int(roi.w * runtime_cfg.scale),
                h=int(roi.h * runtime_cfg.scale),
            ).clamp(gray_l.shape[1], gray_l.shape[0])

            if roi_center_mode == "image-center":
                center_x = gray_l.shape[1] // 2
                center_y = gray_l.shape[0] // 2
            else:
                center_x = roi_scaled_static.x + (roi_scaled_static.w // 2)
                center_y = roi_scaled_static.y + (roi_scaled_static.h // 2)

            roi_scaled = roi_from_physical_size(
                img_w=gray_l.shape[1],
                img_h=gray_l.shape[0],
                center_x=center_x,
                center_y=center_y,
                width_m=roi_phys_w_m,
                height_m=roi_phys_h_m,
                depth_m=roi_depth_ref_m,
                fx=focal_px,
            )

            if runtime_cfg.disparity_roi_only and roi_scaled.w < 32:
                # Keep SGBM ROI mode valid even when physical projection gets too narrow.
                expand = 32 - roi_scaled.w
                roi_scaled = ROI(
                    x=roi_scaled.x - (expand // 2),
                    y=roi_scaled.y,
                    w=32,
                    h=roi_scaled.h,
                ).clamp(gray_l.shape[1], gray_l.shape[0])

            # 4) Disparity either on full frame or constrained ROI.
            if not runtime_cfg.disparity_roi_only:
                max_safe_full = ((max(0, int(gray_l.shape[1])) // 16) - 1) * 16
                if max_safe_full >= 16 and active_num_disp > max_safe_full:
                    old_num_disp = active_num_disp
                    active_num_disp = max_safe_full
                    disp_estimator = StereoDisparityEstimator(
                        SGBMConfig(
                            min_disparity=int(args.min_disp),
                            num_disparities=active_num_disp,
                            block_size=int(args.block_size),
                            p1=8 * 1 * int(args.block_size) * int(args.block_size),
                            p2=32 * 1 * int(args.block_size) * int(args.block_size),
                            uniqueness_ratio=10,
                            speckle_window_size=80,
                            speckle_range=2,
                        )
                    )
                    print(
                        "[WARN] Runtime frame width is small for current --num-disp; "
                        f"adjusting num-disp from {old_num_disp} to {active_num_disp} "
                        f"for width={gray_l.shape[1]}"
                    )

            if runtime_cfg.disparity_roi_only:
                target_num_disp = safe_num_disparities_for_roi(requested_num_disp, roi_scaled.w)
                if target_num_disp != active_num_disp:
                    active_num_disp = target_num_disp
                    disp_estimator = StereoDisparityEstimator(
                        SGBMConfig(
                            min_disparity=int(args.min_disp),
                            num_disparities=active_num_disp,
                            block_size=int(args.block_size),
                            p1=8 * 1 * int(args.block_size) * int(args.block_size),
                            p2=32 * 1 * int(args.block_size) * int(args.block_size),
                            uniqueness_ratio=10,
                            speckle_window_size=80,
                            speckle_range=2,
                        )
                    )
                crop_l, crop_r, roi_used = crop_for_disparity(gray_l, gray_r, roi_scaled)
                try:
                    disp_crop = disp_estimator.compute(crop_l, crop_r)
                except cv2.error as exc:
                    raise RuntimeError(
                        "StereoSGBM failed in ROI mode. Increase ROI width or lower --num-disp. "
                        f"Current ROI width={roi_used.w}, effective num-disp={active_num_disp}."
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
            valid_ratio = (float(valid_pixels) / float(total_pixels)) if total_pixels > 0 else 0.0

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
                min_valid_ratio=max(0.0, float(args.roi_valid_ratio_min)),
                p10_weight=float(args.roi_p10_weight),
                min_weight=float(args.roi_min_weight),
            )
            distance_filtered = distance_filter.update(distance_raw)
            if distance_filtered is not None and np.isfinite(distance_filtered) and distance_filtered > 0:
                roi_depth_ref_m = float(distance_filtered)
            elif distance_raw is not None and np.isfinite(distance_raw) and distance_raw > 0:
                roi_depth_ref_m = float(distance_raw)

            fps = perf.update_fps()
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # 7) Build visualization layers and on-screen diagnostics.
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
                min_disp=float(args.min_disp),
                max_disp=float(args.min_disp + effective_num_disp),
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
            )
            t_depth = time.perf_counter()

            # Optional per-pixel inspection from last mouse click.
            click_px = viz_state.clicked_px
            click_depth = None
            if click_px is not None:
                click_depth = depth_estimator.depth_at(depth_map, click_px[0], click_px[1])
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

            # Keyboard shortcuts: s toggles left/right swap, 1-9 switches input, n selects next input, q/esc exits.
            key_raw = cv2.waitKeyEx(1)
            key = key_raw & 0xFF
            if key == ord("s"):
                swap_lr = not swap_lr

            if multi_mode:
                req_idx = decode_switch_index(key_raw, len(device_list))
                if req_idx is not None:
                    if req_idx != active_idx:
                        switch_state.pending = request_switch(
                            cam=cam,
                            from_idx=active_idx,
                            to_idx=req_idx,
                            key_label="number",
                        )

            if multi_mode and key == ord("n"):
                next_idx = (active_idx + 1) % len(device_list)
                switch_state.pending = request_switch(
                    cam=cam,
                    from_idx=active_idx,
                    to_idx=next_idx,
                    key_label="next",
                )
            if key == 27 or key == ord("q"):
                break

    finally:
        if cam is not None:
            cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
