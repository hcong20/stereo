"""Frame-format helpers for GStreamer stereo capture."""

from typing import Optional, Tuple

import cv2
import numpy as np


def convert_gstreamer_frame_if_needed(
    frame: np.ndarray,
    use_gstreamer: bool,
    selected_pipeline: Optional[str],
) -> np.ndarray:
    """Convert NV12 appsink frames to BGR when OpenCV does not auto-convert."""
    if not use_gstreamer:
        return frame
    if not selected_pipeline or "format=NV12" not in selected_pipeline:
        return frame

    # Some OpenCV builds already return BGR even if appsink requested NV12.
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame

    in_frame = frame
    if in_frame.ndim == 3 and in_frame.shape[2] == 1:
        in_frame = in_frame[:, :, 0]

    if in_frame.ndim != 2:
        raise RuntimeError(
            "Unexpected NV12 frame layout from GStreamer appsink: "
            f"shape={frame.shape}. Try --gst-output bgr"
        )

    h2, _ = in_frame.shape[:2]
    if h2 % 3 != 0:
        raise RuntimeError(
            "Unexpected NV12 frame height from GStreamer appsink: "
            f"shape={frame.shape}. Try --gst-output bgr"
        )

    return cv2.cvtColor(in_frame, cv2.COLOR_YUV2BGR_NV12)


def split_nv12_stereo_to_gray(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split side-by-side NV12 frame into left/right grayscale images.

    For NV12, the first H rows are the Y plane, which is already grayscale.
    Using Y directly avoids per-frame full BGR conversion.
    """
    in_frame = frame
    if in_frame.ndim == 3 and in_frame.shape[2] == 1:
        in_frame = in_frame[:, :, 0]

    # Some OpenCV/GStreamer combinations may still return BGR.
    if in_frame.ndim == 3 and in_frame.shape[2] == 3:
        bgr_w = in_frame.shape[1]
        if bgr_w % 2 != 0:
            raise ValueError(f"Expected combined frame width to be even, got {bgr_w}")
        half = bgr_w // 2
        left_gray = cv2.cvtColor(in_frame[:, :half], cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(in_frame[:, half:], cv2.COLOR_BGR2GRAY)
        return left_gray, right_gray

    if in_frame.ndim != 2:
        raise RuntimeError(
            "Unexpected NV12 frame layout from GStreamer appsink: "
            f"shape={frame.shape}. Try --gst-output bgr"
        )

    h3_2, w = in_frame.shape[:2]
    if h3_2 % 3 != 0:
        raise RuntimeError(
            "Unexpected NV12 frame height from GStreamer appsink: "
            f"shape={frame.shape}. Try --gst-output bgr"
        )
    if w % 2 != 0:
        raise ValueError(f"Expected combined frame width to be even, got {w}")

    h = (h3_2 * 2) // 3
    y_plane = in_frame[:h, :]
    half = w // 2
    left_gray = y_plane[:, :half]
    right_gray = y_plane[:, half:]
    return left_gray, right_gray