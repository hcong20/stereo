"""RK3588 RGA backend adapter for stereo preprocessing.

This module is optional. It exposes a stable API used by
`stereo_vision.preprocess.FramePreprocessor`.

To use hardware acceleration, implement/install a backend module that provides
an RGA-accelerated function and adapt it in `_rga_preprocess_impl`.
"""

from __future__ import annotations

import cv2
import numpy as np


def _try_import_rga_backend():
    """Try importing known RGA python bindings.

    Returns:
        Imported backend module, or None if unavailable.
    """
    candidates = [
        "rockchip_rga",
        "rga",
        "pyrga",
    ]
    for name in candidates:
        try:
            mod = __import__(name)
            return mod
        except Exception:
            continue
    return None


_RGA_BACKEND = _try_import_rga_backend()


def is_available() -> bool:
    """Return whether a compatible RGA Python backend is available."""
    return _RGA_BACKEND is not None


def _cpu_fallback(
    left_bgr: np.ndarray, right_bgr: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if scale == 1.0:
        left_resized = left_bgr
        right_resized = right_bgr
    else:
        h, w = left_bgr.shape[:2]
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        left_resized = cv2.resize(left_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        right_resized = cv2.resize(right_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    left_gray = cv2.cvtColor(left_resized, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_resized, cv2.COLOR_BGR2GRAY)
    return left_resized, right_resized, left_gray, right_gray


def _rga_preprocess_impl(
    left_bgr: np.ndarray, right_bgr: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Call real RGA backend implementation.

    This is intentionally explicit: if a backend is importable but no adapter
    is implemented yet for that backend API, fail with a clear message.
    """
    if _RGA_BACKEND is None:
        raise RuntimeError("RGA backend module is not available")
    fn = getattr(_RGA_BACKEND, "preprocess_pair_bgr_to_gray", None)
    if not callable(fn):
        raise RuntimeError(
            "RGA backend module does not expose preprocess_pair_bgr_to_gray()"
        )
    return fn(left_bgr, right_bgr, float(scale))


def preprocess_pair_bgr_to_gray(
    left_bgr: np.ndarray, right_bgr: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess stereo frames with RGA when available.

    Contract:
        Returns (left_resized_bgr, right_resized_bgr, left_gray, right_gray).
    """
    if not is_available():
        # If unavailable, fallback keeps local development usable.
        return _cpu_fallback(left_bgr, right_bgr, float(scale))
    return _rga_preprocess_impl(left_bgr, right_bgr, float(scale))
