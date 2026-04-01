"""Stereo disparity estimation based on OpenCV SGBM.

Wraps matcher configuration and normalizes OpenCV fixed-point output.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SGBMConfig:
    """Tunable parameters for cv2.StereoSGBM matcher creation.

    num_disparities must be divisible by 16; block_size must be odd.
    """

    min_disparity: int = 0
    num_disparities: int = 128
    block_size: int = 5
    p1: int = 8 * 1 * 5 * 5
    p2: int = 32 * 1 * 5 * 5
    disp12_max_diff: int = 1
    pre_filter_cap: int = 31
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2
    mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY


class StereoDisparityEstimator:
    """Wrapper around OpenCV SGBM with validated configuration."""

    def __init__(self, cfg: SGBMConfig):
        """Create SGBM matcher instance after sanity checks.

        Raises:
            ValueError: If SGBM constraints are violated.
        """
        if cfg.num_disparities % 16 != 0:
            raise ValueError("num_disparities must be divisible by 16")
        if cfg.block_size % 2 == 0:
            raise ValueError("block_size must be odd")

        self.cfg = cfg
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=cfg.min_disparity,
            numDisparities=cfg.num_disparities,
            blockSize=cfg.block_size,
            P1=cfg.p1,
            P2=cfg.p2,
            disp12MaxDiff=cfg.disp12_max_diff,
            preFilterCap=cfg.pre_filter_cap,
            uniquenessRatio=cfg.uniqueness_ratio,
            speckleWindowSize=cfg.speckle_window_size,
            speckleRange=cfg.speckle_range,
            mode=cfg.mode,
        )

    def compute(self, left_gray: np.ndarray, right_gray: np.ndarray) -> np.ndarray:
        """Compute disparity map in pixel units (OpenCV fixed-point / 16).

        Args:
            left_gray: Rectified left grayscale image.
            right_gray: Rectified right grayscale image.

        Returns:
            Float32 disparity image measured in pixels.
        """
        raw = self.matcher.compute(left_gray, right_gray).astype(np.float32)
        return raw / 16.0
