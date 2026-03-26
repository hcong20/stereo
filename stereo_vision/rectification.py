"""Stereo rectification map construction and image remapping.

Rectification aligns epipolar lines horizontally so stereo matching can be
performed along scanlines with reduced search ambiguity.
"""

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from stereo_vision.calibration import StereoCalibration


@dataclass
class RectificationData:
    """Precomputed rectification maps and projection metadata.

    Includes map pairs for both cameras and Q/P matrices for depth geometry.
    """

    map1_l: np.ndarray
    map2_l: np.ndarray
    map1_r: np.ndarray
    map2_r: np.ndarray
    q: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    roi_l: Tuple[int, int, int, int]
    roi_r: Tuple[int, int, int, int]


def build_rectification_maps(
    calib: StereoCalibration, image_size: Tuple[int, int], use_precomputed: bool = False
) -> RectificationData:
    """Build rectification maps from calibration.

    Args:
        calib: Stereo calibration container.
        image_size: Input image size as (width, height).
        use_precomputed: Reuse R1/R2/P1/P2/Q from calibration when available.

    Returns:
        RectificationData with left/right remap tables and metadata.
    """
    if (
        use_precomputed
        and
        calib.r1 is not None
        and calib.r2 is not None
        and calib.p1 is not None
        and calib.p2 is not None
        and calib.q is not None
    ):
        # Use cached calibration outputs directly when they are available.
        r1 = calib.r1
        r2 = calib.r2
        p1 = calib.p1
        p2 = calib.p2
        q = calib.q
        roi_l = (0, 0, image_size[0], image_size[1])
        roi_r = (0, 0, image_size[0], image_size[1])
    else:
        # Recompute rectification from raw intrinsics/extrinsics.
        r1, r2, p1, p2, q, roi_l, roi_r = cv2.stereoRectify(
            calib.k_left,
            calib.d_left,
            calib.k_right,
            calib.d_right,
            image_size,
            calib.r,
            calib.t,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )

    # CV_16SC2 maps are faster for remap and sufficient for this application.
    map1_l, map2_l = cv2.initUndistortRectifyMap(
        calib.k_left, calib.d_left, r1, p1, image_size, cv2.CV_16SC2
    )
    map1_r, map2_r = cv2.initUndistortRectifyMap(
        calib.k_right, calib.d_right, r2, p2, image_size, cv2.CV_16SC2
    )

    return RectificationData(
        map1_l=map1_l,
        map2_l=map2_l,
        map1_r=map1_r,
        map2_r=map2_r,
        q=q,
        p1=p1,
        p2=p2,
        roi_l=roi_l,
        roi_r=roi_r,
    )


def rectify_pair(
    left: np.ndarray, right: np.ndarray, rect: RectificationData
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply precomputed remap tables to rectify a stereo pair.

    Args:
        left: Raw left image.
        right: Raw right image.
        rect: Precomputed rectification maps.

    Returns:
        Tuple of rectified (left, right) images.
    """
    left_rect = cv2.remap(left, rect.map1_l, rect.map2_l, interpolation=cv2.INTER_LINEAR)
    right_rect = cv2.remap(right, rect.map1_r, rect.map2_r, interpolation=cv2.INTER_LINEAR)
    return left_rect, right_rect
