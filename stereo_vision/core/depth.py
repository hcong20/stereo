"""Depth conversion utilities from disparity maps.

Uses the pinhole stereo relation: depth = focal_px * baseline_m / disparity.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DepthConfig:
    """Parameters controlling disparity-to-depth conversion and clipping.

    min_disparity filters unreliable near-zero disparities.
    max_depth_m can mask far values as NaN for cleaner downstream statistics.
    """

    focal_px: float
    baseline_m: float
    min_disparity: float = 0.5
    max_depth_m: float = 30.0


class DepthEstimator:
    """Convert disparity values to metric depth estimates."""

    def __init__(self, cfg: DepthConfig):
        """Store depth conversion configuration."""
        self.cfg = cfg

    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """Compute depth map in meters while preserving invalid pixels as NaN.

        Args:
            disparity: Disparity map in pixel units.

        Returns:
            Depth map in meters with invalid entries as NaN.
        """
        depth = np.full(disparity.shape, np.nan, dtype=np.float32)
        valid = disparity > self.cfg.min_disparity
        depth[valid] = (self.cfg.focal_px * self.cfg.baseline_m) / disparity[valid]

        if self.cfg.max_depth_m > 0:
            depth[depth > self.cfg.max_depth_m] = np.nan

        return depth

    def depth_at(self, depth_map: np.ndarray, x: int, y: int) -> Optional[float]:
        """Return depth at a pixel, or None when out of bounds/invalid."""
        if x < 0 or y < 0 or y >= depth_map.shape[0] or x >= depth_map.shape[1]:
            return None
        value = depth_map[y, x]
        if np.isnan(value):
            return None
        return float(value)
