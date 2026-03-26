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


def robust_roi_distance(depth_map: np.ndarray, roi: ROI, min_valid_pixels: int = 10) -> Optional[float]:
    """Estimate robust ROI depth using median after IQR outlier rejection.

    Args:
        depth_map: Depth image in meters (NaN for invalid pixels).
        roi: Region from which to compute a representative distance.
        min_valid_pixels: Minimum finite pixels required for a valid estimate.

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
        return float(np.median(valid))

    return float(np.median(filtered))
