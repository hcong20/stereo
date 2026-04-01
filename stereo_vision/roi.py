"""ROI primitives and robust depth aggregation utilities.

Provides a bounded ROI helper and robust depth summarization for noisy maps.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ROI:
    """Axis-aligned rectangular region of interest in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

    def clamp(self, img_w: int, img_h: int) -> "ROI":
        """Clamp ROI bounds so they stay fully inside image dimensions.

        Returns:
            New ROI guaranteed to be valid for an image of size img_w x img_h.
        """
        x = max(0, min(self.x, img_w - 1))
        y = max(0, min(self.y, img_h - 1))
        w = max(1, min(self.w, img_w - x))
        h = max(1, min(self.h, img_h - y))
        return ROI(x, y, w, h)

    def as_slice(self) -> Tuple[slice, slice]:
        """Return numpy slice objects selecting this ROI."""
        return slice(self.y, self.y + self.h), slice(self.x, self.x + self.w)


def robust_roi_distance(
    depth_map: np.ndarray,
    roi: ROI,
    min_valid_pixels: int = 10,
    min_valid_ratio: float = 0.0,
    p10_weight: float = 0.7,
    min_weight: float = 0.1,
) -> Optional[float]:
    """Estimate robust ROI depth using IQR filtering and weighted statistics.

    Args:
        depth_map: Depth image in meters (NaN for invalid pixels).
        roi: Region from which to compute a representative distance.
        min_valid_pixels: Minimum finite pixels required for a valid estimate.
        min_valid_ratio: Minimum finite-depth ratio required in ROI.
        p10_weight: Weight of P10 vs median in the base robust estimate.
        min_weight: Extra weight of minimum depth fused after base estimate.

    Returns:
        Robust median depth in meters, or None if too few valid samples.
    """
    h, w = depth_map.shape[:2]
    roi = roi.clamp(w, h)
    ys, xs = roi.as_slice()
    patch = depth_map[ys, xs]

    valid = patch[np.isfinite(patch)]
    if valid.size < min_valid_pixels:
        return None

    total = int(patch.size)
    valid_ratio = (float(valid.size) / float(total)) if total > 0 else 0.0
    if valid_ratio < max(0.0, float(min_valid_ratio)):
        return None

    # Interquartile statistics reduce influence from heavy-tail mismatch noise.
    q1 = np.percentile(valid, 25)
    q3 = np.percentile(valid, 75)
    iqr = q3 - q1
    if iqr <= 0:
        return float(np.median(valid))

    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    # Standard Tukey IQR fence to suppress outliers from mismatch noise.
    filtered = valid[(valid >= low) & (valid <= high)]
    if filtered.size == 0:
        # Keep a stable value even when IQR clipping becomes too aggressive.
        filtered = valid

    # Blend robust low-percentile and median so we remain stable yet responsive.
    p10 = float(np.percentile(filtered, 10))
    p50 = float(np.percentile(filtered, 50))
    dmin = float(np.min(filtered))

    p10_w = float(np.clip(p10_weight, 0.0, 1.0))
    min_w = float(np.clip(min_weight, 0.0, 1.0))
    base = p10_w * p10 + (1.0 - p10_w) * p50
    return float((1.0 - min_w) * base + min_w * dmin)


def roi_from_physical_size(
    img_w: int,
    img_h: int,
    center_x: int,
    center_y: int,
    width_m: float,
    height_m: float,
    depth_m: float,
    fx: float,
    fy: Optional[float] = None,
) -> ROI:
    """Convert a fixed physical-size window to pixel ROI at a given depth."""
    fy_eff = float(fx if fy is None else fy)
    z = max(1e-3, float(depth_m))
    w_px = max(1, int(round((float(width_m) * float(fx)) / z)))
    h_px = max(1, int(round((float(height_m) * fy_eff) / z)))
    x = int(round(center_x - w_px / 2.0))
    y = int(round(center_y - h_px / 2.0))
    return ROI(x=x, y=y, w=w_px, h=h_px).clamp(int(img_w), int(img_h))
