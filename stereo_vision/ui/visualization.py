"""Visualization helpers for disparity and distance overlays.

Functions in this module are intentionally stateless except for VizState,
which stores UI interaction state shared with OpenCV mouse callbacks.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from stereo_vision.core.roi import ROI


@dataclass
class VizState:
    """Mutable UI state shared with mouse callback handlers."""

    clicked_px: Optional[Tuple[int, int]] = None


def colorize_disparity(disparity: np.ndarray, min_disp: float, max_disp: float) -> np.ndarray:
    """Map disparity values into a clipped pseudo-color image.

    Values outside [min_disp, max_disp] are clipped for stable contrast.
    """
    clipped = np.clip(disparity, min_disp, max_disp)
    norm = ((clipped - min_disp) / max(1e-6, max_disp - min_disp) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def draw_roi(frame: np.ndarray, roi: ROI, color=(0, 255, 0)) -> np.ndarray:
    """Draw an ROI rectangle on a copied frame.

    Copying preserves caller ownership and avoids accidental in-place edits.
    """
    out = frame.copy()
    cv2.rectangle(out, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), color, 2)
    return out


def draw_text(frame: np.ndarray, text: str, origin=(10, 30), color=(255, 255, 255)) -> np.ndarray:
    """Render high-contrast text with a dark outline for readability.

    Two-pass rendering (dark then light) keeps text visible on bright backgrounds.
    """
    out = frame.copy()
    cv2.putText(out, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)
    return out


def register_click(window: str, state: VizState) -> None:
    """Attach left-click handler that records the last clicked pixel.

    Args:
        window: OpenCV window name already created by cv2.namedWindow.
        state: Shared visualization state to update on click events.
    """
    def on_mouse(event, x, y, flags, userdata):
        # Keep latest click only; main loop consumes this position every frame.
        if event == cv2.EVENT_LBUTTONDOWN:
            state.clicked_px = (x, y)

    cv2.setMouseCallback(window, on_mouse)
