"""Runtime assembly helpers for startup/config composition."""

from __future__ import annotations

from argparse import Namespace

import cv2

from src.app_cli import (
    fourcc_to_str,
    parse_physical_size_mm,
    parse_roi,
    safe_num_disparities_for_roi,
)
from src.config.calibration_paths import (
    calibration_path_for_device,
    calibration_path_for_direction,
)
from src.core.calibration import load_stereo_calibration
from src.core.depth import DepthConfig, DepthEstimator
from src.core.disparity import SGBMConfig, StereoDisparityEstimator
from src.core.rectification import build_rectification_maps
from src.core.roi import ROI, roi_from_physical_size
from src.pipeline.optimization import RuntimeOptimizationConfig
from src.pipeline.preprocess import FramePreprocessor, PreprocessConfig
from src.runtime.runtime_loop import RuntimeLoopConfig
from src.runtime.runtime_profile import StageProfiler
from src.runtime.runtime_tuning import RoiTuneController
from src.capture.startup import initialize_capture


def build_runtime_context(args: Namespace) -> RuntimeLoopConfig:
    """Build and wire runtime components from parsed CLI args."""
    device_list = [d.strip() for d in str(args.devices).split(",") if d.strip()]
    if len(device_list) == 0:
        device_list = [str(args.device)]
    direction_list = [d.strip() for d in str(getattr(args, "directions", "")).split(",") if d.strip()]
    if len(device_list) != 4:
        print(
            f"[WARN] Expected 4 stereo inputs for deployment goal, got {len(device_list)}. "
            "System will run with available inputs."
        )
    if direction_list and len(direction_list) != len(device_list):
        print(
            f"[WARN] --directions count does not match --devices count; "
            f"ignoring direction labels ({len(direction_list)} vs {len(device_list)})"
        )

    startup = initialize_capture(args, device_list)
    cam = startup.cam
    multi_mode = startup.multi_mode
    switch_timeout_s = startup.switch_timeout_s
    active_idx = startup.active_idx
    left0 = startup.left0
    active_device = device_list[active_idx]
    direction_list = [d.strip() for d in str(getattr(args, "directions", "")).split(",") if d.strip()]
    active_direction = direction_list[active_idx] if len(direction_list) == len(device_list) else ""

    if str(args.calib).strip():
        calib_path = str(args.calib).strip()
    elif active_direction:
        direction_path = calibration_path_for_direction(active_direction)
        calib_path = str(direction_path if direction_path.is_file() else calibration_path_for_device(active_device))
    else:
        calib_path = str(calibration_path_for_device(active_device))
    if not str(args.calib).strip():
        print(f"[INFO] Using calibration file: {calib_path}")
    calib = load_stereo_calibration(calib_path)

    # Prime the camera once to discover frame geometry and stream properties.
    image_size = (left0.shape[1], left0.shape[0])

    if multi_mode:
        direction_text = f", direction={active_direction}" if active_direction else ""
        print(
            f"[INFO] Active input device: idx={active_idx + 1}{direction_text}, device={active_device}"
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

    rect = build_rectification_maps(
        calib,
        image_size,
        use_precomputed=bool(args.use_precomputed_rect),
    )

    scale = float(args.scale)
    if not 0.25 <= scale <= 1.0:
        raise ValueError("--scale should be in [0.25, 1.0]")

    focal_px = float(rect.p1[0, 0]) * scale
    baseline_raw = float(calib.baseline_mm)
    # Baseline stored in calibration is mm; convert to meters.
    baseline_m = float(baseline_raw) / 1000.0
    print(f"[INFO] baseline_raw={baseline_raw:.6f} mm, baseline_m={baseline_m:.6f} m")

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

    roi_tuning = RoiTuneController(args)
    if roi_tuning.roi_tune_preset != "off":
        roi_tuning.apply_preset(roi_tuning.roi_tune_preset)

    runtime_cfg = RuntimeOptimizationConfig(
        scale=scale,
        disparity_roi_only=bool(args.roi_disparity_only),
    )

    preprocessor = FramePreprocessor(
        PreprocessConfig(
            scale=runtime_cfg.scale,
            crop_height_ratio=float(args.crop_height_ratio),
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

    profiler = StageProfiler(
        enabled=bool(getattr(args, "profile_stages", False)),
        interval=max(1, int(getattr(args, "profile_interval", 60))),
    )

    return RuntimeLoopConfig(
        cam=cam,
        multi_mode=multi_mode,
        switch_timeout_s=switch_timeout_s,
        active_idx=active_idx,
        rect=rect,
        runtime_cfg=runtime_cfg,
        preprocessor=preprocessor,
        depth_estimator=depth_estimator,
        disp_estimator=disp_estimator,
        requested_num_disp=requested_num_disp,
        effective_num_disp=effective_num_disp,
        baseline_m=baseline_m,
        focal_px=focal_px,
        roi=roi,
        roi_phys_w_m=roi_phys_w_m,
        roi_phys_h_m=roi_phys_h_m,
        roi_center_mode=roi_center_mode,
        roi_depth_ref_m=roi_depth_ref_m,
        device_list=device_list,
        roi_tuning=roi_tuning,
        preview_nv12_bgr=preview_nv12_bgr,
        profiler=profiler,
    )
