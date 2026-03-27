#!/usr/bin/env python3
"""Entry point for real-time stereo distance measurement and visualization.

This script wires together capture, rectification, disparity estimation,
depth conversion, robust ROI distance extraction, temporal smoothing,
and live OpenCV visualization.
"""

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from stereo_vision.calibration import load_stereo_calibration
from stereo_vision.camera import CameraConfig, StereoCamera
from stereo_vision.depth import DepthConfig, DepthEstimator
from stereo_vision.disparity import SGBMConfig, StereoDisparityEstimator
from stereo_vision.filters import DistanceFilter, TemporalFilterConfig
from stereo_vision.optimization import RuntimeOptimizationConfig, crop_for_disparity
from stereo_vision.preprocess import FramePreprocessor, PreprocessConfig
from stereo_vision.rectification import build_rectification_maps, rectify_pair
from stereo_vision.roi import ROI, robust_roi_distance
from stereo_vision.visualization import VizState, colorize_disparity, draw_roi, draw_text, register_click


@dataclass
class PerfStats:
    """Track runtime throughput for on-screen FPS reporting.

    The class computes average FPS from process start instead of instantaneous
    FPS to keep the overlay stable and easy to interpret.
    """

    frame_count: int = 0
    start: float = time.perf_counter()

    def update_fps(self) -> float:
        """Update frame counter and return average FPS since start.

        Returns:
            Average frames per second over the elapsed runtime.
        """
        self.frame_count += 1
        elapsed = max(1e-6, time.perf_counter() - self.start)
        return self.frame_count / elapsed


def parse_args() -> argparse.Namespace:
    """Parse CLI options for camera, disparity, depth, and filtering.

    Returns:
        Parsed command-line namespace used to configure the whole pipeline.
    """
    parser = argparse.ArgumentParser(description="RK3588 Stereo Distance Measurement")

    parser.add_argument("--device", default="/dev/video20")
    parser.add_argument("--calib", default="stereo_calib_params.npz")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--gstreamer", action="store_true")
    parser.add_argument("--swap-lr", action="store_true", help="Swap left/right camera halves")
    parser.add_argument(
        "--use-precomputed-rect",
        action="store_true",
        help="Use R1/R2/P1/P2/Q from calibration if available",
    )

    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--preprocess-backend",
        choices=["auto", "cpu", "rga"],
        default="auto",
        help="Frame preprocess backend (RGA requires external module)",
    )
    parser.add_argument(
        "--rga-module",
        type=str,
        default="rga_helper",
        help="Python module name providing RGA preprocess_pair_bgr_to_gray",
    )
    parser.add_argument("--roi", type=str, default="270,175,100,70", help="x,y,w,h")
    parser.add_argument("--roi-disparity-only", action="store_true")

    parser.add_argument("--num-disp", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--min-disp", type=int, default=0)
    parser.add_argument("--depth-min-disp", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument(
        "--baseline-unit",
        choices=["auto", "m", "mm"],
        default="auto",
        help="Unit of T baseline stored in calibration",
    )

    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--max-jump", type=float, default=1.0)
    parser.add_argument("--filter-window", type=int, default=5)
    parser.add_argument("--min-valid-pixels", type=int, default=10)

    return parser.parse_args()


def parse_roi(text: str) -> ROI:
    """Parse ROI text in x,y,w,h format into an ROI object.

    Args:
        text: ROI string such as "270,175,100,70".

    Returns:
        Parsed ROI instance.
    """
    vals = [int(v.strip()) for v in text.split(",")]
    if len(vals) != 4:
        raise ValueError("ROI must be x,y,w,h")
    return ROI(*vals)


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale, or pass through if already gray.

    Args:
        img: Input image in either grayscale or BGR format.

    Returns:
        2D grayscale image.
    """
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def get_screen_size() -> tuple[int, int] | None:
    """Best-effort screen size query for window centering.

    Returns:
        (width, height) in pixels when available, else None.
    """
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = int(root.winfo_screenwidth())
        height = int(root.winfo_screenheight())
        root.destroy()
        if width > 0 and height > 0:
            return width, height
    except Exception:
        return None
    return None


def resolve_baseline_m(raw_baseline: float, baseline_unit: str) -> float:
    """Convert baseline value to meters using explicit or inferred unit.

    Args:
        raw_baseline: Baseline value loaded from calibration data.
        baseline_unit: One of "m", "mm", or "auto".

    Returns:
        Baseline value in meters.
    """
    if baseline_unit == "m":
        return raw_baseline
    if baseline_unit == "mm":
        return raw_baseline / 1000.0
    # Auto: most stereo rigs are in cm-scale baseline; values >2 are likely mm.
    if raw_baseline > 2.0:
        return raw_baseline / 1000.0
    return raw_baseline


def safe_num_disparities_for_roi(requested: int, roi_width: int) -> int:
    """Choose safe SGBM disparities for ROI mode.

    OpenCV SGBM can fail when numDisparities is too large relative to the
    horizontal extent of the ROI. This helper clamps to a conservative value.

    Args:
        requested: User requested numDisparities.
        roi_width: ROI width in pixels after scaling.

    Returns:
        A safe numDisparities value (multiple of 16).
    """
    req = max(16, (int(requested) // 16) * 16)
    max_safe = ((max(0, int(roi_width)) // 16) - 1) * 16
    if max_safe < 16:
        raise ValueError(
            f"ROI width={roi_width} is too small for SGBM ROI mode. Increase ROI width to at least 32 pixels."
        )
    return min(req, max_safe)


def fourcc_to_str(fourcc_value: float) -> str:
    """Decode OpenCV FOURCC numeric code into a readable 4-char string."""
    code = int(fourcc_value)
    return "".join([chr((code >> (8 * i)) & 0xFF) for i in range(4)])


def main() -> None:
    """Run the full stereo pipeline loop and render diagnostic overlays.

    The loop sequence is:
    capture -> rectify -> resize -> disparity -> depth -> ROI measurement
    -> temporal filtering -> visualization.
    """
    args = parse_args()

    cam = StereoCamera(
        CameraConfig(
            device=args.device,
            width=args.width,
            height=args.height,
            fps=args.fps,
            use_gstreamer=args.gstreamer,
        )
    )
    calib = load_stereo_calibration(args.calib)

    # Prime the camera once to discover frame geometry and stream properties.
    cam.open()
    left0, right0, _ = cam.read()
    image_size = (left0.shape[1], left0.shape[0])

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
    preprocessor = FramePreprocessor(
        PreprocessConfig(
            scale=runtime_cfg.scale,
            backend=str(args.preprocess_backend),
            rga_module=str(args.rga_module),
        )
    )
    print(f"[INFO] preprocess_backend={preprocessor.backend_name}")

    swap_lr = bool(args.swap_lr)
    viz_state = VizState()

    win = "stereo_distance"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    register_click(win, viz_state)

    perf = PerfStats()
    window_sized = False
    screen_size = get_screen_size()

    try:
        while True:
            t0 = time.perf_counter()
            # 1) Capture and optional channel swap.
            left, right, _ = cam.read()
            if swap_lr:
                left, right = right, left

            # 2) Geometric rectification.
            left_rect, right_rect = rectify_pair(left, right, rect)

            # 3) Runtime preprocess (resize + grayscale), optionally RGA-backed.
            left_rect, right_rect, gray_l, gray_r = preprocessor.process(left_rect, right_rect)

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
            left_viz = draw_roi(left_rect, roi_scaled)

            if distance_filtered is not None:
                left_viz = draw_text(left_viz, f"ROI Distance: {distance_filtered * 1000.0:.1f} mm", (10, 30), (0, 255, 0))
            else:
                left_viz = draw_text(left_viz, "ROI Distance: N/A", (10, 30), (0, 0, 255))

            left_viz = draw_text(left_viz, f"FPS: {fps:.1f}", (10, 60), (255, 255, 0))
            left_viz = draw_text(left_viz, f"Latency: {latency_ms:.1f} ms", (10, 90), (255, 255, 0))
            left_viz = draw_text(
                left_viz,
                f"ROI valid: {valid_pixels}/{total_pixels}",
                (10, 120),
                (255, 255, 0),
            )
            left_viz = draw_text(
                left_viz,
                f"ROI disp>0: {positive_disp_pixels}/{total_pixels}",
                (10, 150),
                (255, 255, 0),
            )
            left_viz = draw_text(
                left_viz,
                f"ROI clipped(maxD): {clipped_by_max_depth}",
                (10, 180),
                (255, 255, 0),
            )

            if distance_raw is not None:
                left_viz = draw_text(left_viz, f"ROI Raw: {distance_raw * 1000.0:.1f} mm", (10, 210), (200, 255, 200))
            else:
                left_viz = draw_text(left_viz, "ROI Raw: N/A", (10, 210), (0, 0, 255))

            # Optional per-pixel inspection from last mouse click.
            if viz_state.clicked_px is not None:
                cx, cy = viz_state.clicked_px
                if 0 <= cx < left_viz.shape[1] and 0 <= cy < left_viz.shape[0]:
                    cv2.circle(left_viz, (cx, cy), 5, (0, 0, 255), -1, cv2.LINE_AA)
                click_depth: Optional[float] = depth_estimator.depth_at(depth_map, cx, cy)
                if click_depth is not None:
                    left_viz = draw_text(left_viz, f"Click({cx},{cy}): {click_depth * 1000.0:.1f} mm", (10, 240), (255, 200, 0))
                else:
                    left_viz = draw_text(left_viz, f"Click({cx},{cy}): N/A", (10, 240), (0, 0, 255))

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

            # Keyboard shortcuts: s toggles left/right swap, q/esc exits.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                swap_lr = not swap_lr
            if key == 27 or key == ord("q"):
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
