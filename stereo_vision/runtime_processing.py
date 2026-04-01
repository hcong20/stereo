"""Per-frame ROI, disparity, and depth processing helpers."""

from __future__ import annotations

from argparse import Namespace

import cv2
import numpy as np

from stereo_vision.app_cli import safe_num_disparities_for_roi
from stereo_vision.depth import DepthEstimator
from stereo_vision.disparity import SGBMConfig, StereoDisparityEstimator
from stereo_vision.optimization import RuntimeOptimizationConfig, crop_for_disparity
from stereo_vision.roi import ROI, robust_roi_distance, roi_from_physical_size
from stereo_vision.runtime_tuning import RoiTuneController


def compute_runtime_roi(
    *,
    gray_shape: tuple[int, int],
    roi: ROI,
    runtime_cfg: RuntimeOptimizationConfig,
    roi_center_mode: str,
    roi_phys_w_m: float,
    roi_phys_h_m: float,
    roi_depth_ref_m: float,
    focal_px: float,
) -> ROI:
    """Project the physical ROI onto the current processed frame."""
    gray_h, gray_w = gray_shape
    roi_scaled_static = ROI(
        x=int(roi.x * runtime_cfg.scale),
        y=int(roi.y * runtime_cfg.scale),
        w=int(roi.w * runtime_cfg.scale),
        h=int(roi.h * runtime_cfg.scale),
    ).clamp(gray_w, gray_h)

    if roi_center_mode == "image-center":
        center_x = gray_w // 2
        center_y = gray_h // 2
    else:
        center_x = roi_scaled_static.x + (roi_scaled_static.w // 2)
        center_y = roi_scaled_static.y + (roi_scaled_static.h // 2)

    roi_scaled = roi_from_physical_size(
        img_w=gray_w,
        img_h=gray_h,
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
        ).clamp(gray_w, gray_h)

    return roi_scaled


def compute_disparity(
    *,
    gray_l: np.ndarray,
    gray_r: np.ndarray,
    roi_scaled: ROI,
    runtime_cfg: RuntimeOptimizationConfig,
    args: Namespace,
    requested_num_disp: int,
    disp_estimator: StereoDisparityEstimator,
) -> tuple[np.ndarray, StereoDisparityEstimator, int]:
    """Compute disparity on full frame or constrained ROI depending on runtime mode."""
    # Current estimator stores parameters on `cfg`; keep a fallback for older variants.
    estimator_cfg = getattr(disp_estimator, "cfg", getattr(disp_estimator, "config", None))
    if estimator_cfg is None:
        raise AttributeError("StereoDisparityEstimator must expose cfg/config with num_disparities")
    active_num_disp = int(estimator_cfg.num_disparities)

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

    return disparity, disp_estimator, active_num_disp


def compute_depth_and_distance(
    *,
    disparity: np.ndarray,
    depth_estimator: DepthEstimator,
    roi_scaled: ROI,
    args: Namespace,
    focal_px: float,
    baseline_m: float,
    roi_tuning: RoiTuneController,
) -> tuple[
    np.ndarray,
    int,
    int,
    float,
    int,
    int,
    float | None,
    float | None,
    str | None,
]:
    """Compute depth map, ROI metrics, and filtered distance."""
    depth_map = depth_estimator.disparity_to_depth(disparity)
    ys, xs = roi_scaled.as_slice()
    roi_disp = disparity[ys, xs]
    roi_depth = depth_map[ys, xs]

    positive_disp_pixels = int((roi_disp > 0.0).sum())
    valid_pixels = int(np.isfinite(roi_depth).sum())
    total_pixels = int(roi_depth.size)
    valid_ratio = (float(valid_pixels) / float(total_pixels)) if total_pixels > 0 else 0.0
    min_valid_pixels = max(1, int(args.min_valid_pixels))
    min_valid_ratio = max(0.0, float(args.roi_valid_ratio_min))

    roi_gate_note = None
    if valid_pixels < min_valid_pixels:
        roi_gate_note = f"gate: valid pixels {valid_pixels} < {min_valid_pixels}"
    elif valid_ratio < min_valid_ratio:
        roi_gate_note = (
            f"gate: valid ratio {valid_ratio * 100.0:.1f}% < {min_valid_ratio * 100.0:.1f}%"
        )

    roi_depth_raw = np.full(roi_disp.shape, np.nan, dtype=np.float32)
    raw_valid = roi_disp > max(0.01, float(args.depth_min_disp))
    roi_depth_raw[raw_valid] = (focal_px * baseline_m) / roi_disp[raw_valid]
    clipped_by_max_depth = 0
    if float(args.max_depth) > 0:
        clipped_by_max_depth = int(
            np.isfinite(roi_depth_raw).sum() - np.isfinite(roi_depth).sum()
        )

    distance_raw = robust_roi_distance(
        depth_map,
        roi_scaled,
        min_valid_pixels=min_valid_pixels,
        min_valid_ratio=min_valid_ratio,
        p10_weight=float(args.roi_p10_weight),
        min_weight=float(args.roi_min_weight),
    )
    distance_filtered = roi_tuning.distance_filter.update(distance_raw)

    return (
        depth_map,
        valid_pixels,
        total_pixels,
        valid_ratio,
        positive_disp_pixels,
        clipped_by_max_depth,
        distance_raw,
        distance_filtered,
        roi_gate_note,
    )
