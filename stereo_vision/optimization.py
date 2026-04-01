"""Lightweight runtime optimizations for stereo processing.

These helpers reduce compute cost without changing stereo math semantics.
"""

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from stereo_vision.roi import ROI


@dataclass
class RuntimeOptimizationConfig:
    """Runtime toggles for scaling and ROI-constrained disparity."""

    scale: float = 1.0
    disparity_roi_only: bool = False


def fast_resize(frame: np.ndarray, scale: float) -> np.ndarray:
    """Resize frame by scale using area interpolation for downsampling.

    Args:
        frame: Input image.
        scale: Positive scale factor where 1.0 keeps original size.

    Returns:
        Resized image.
    """
    if scale == 1.0:
        return frame
    h, w = frame.shape[:2]
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def crop_for_disparity(gray_l: np.ndarray, gray_r: np.ndarray, roi: ROI) -> Tuple[np.ndarray, np.ndarray, ROI]:
    """Crop synchronized left/right grayscale images to a valid ROI.

    ROI is clamped to image bounds before slicing to avoid index errors.
    """
    h, w = gray_l.shape[:2]
    roi = roi.clamp(w, h)
    ys, xs = roi.as_slice()
    return gray_l[ys, xs], gray_r[ys, xs], roi


