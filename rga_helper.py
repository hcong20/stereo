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


def _to_gray_u8_2d(frame: np.ndarray) -> np.ndarray:
    """Normalize frame into contiguous uint8 HxW grayscale."""
    out = frame
    if out.dtype != np.uint8:
        out = out.astype(np.uint8, copy=False)
    if out.ndim == 2:
        pass
    elif out.ndim == 3 and out.shape[2] == 1:
        out = out[:, :, 0]
    elif out.ndim == 3 and out.shape[2] == 3:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    elif out.ndim == 3 and out.shape[2] == 4:
        out = cv2.cvtColor(out, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported grayscale input shape: {out.shape}")
    if not out.flags.c_contiguous:
        out = np.ascontiguousarray(out)
    return out


def _cpu_fallback_gray(
    left_gray: np.ndarray, right_gray: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CPU grayscale resize fallback matching gray-direct contract."""
    left_in = _to_gray_u8_2d(left_gray)
    right_in = _to_gray_u8_2d(right_gray)
    if scale == 1.0:
        left_resized = left_in
        right_resized = right_in
    else:
        h, w = left_in.shape[:2]
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        left_resized = cv2.resize(left_in, (nw, nh), interpolation=cv2.INTER_AREA)
        right_resized = cv2.resize(right_in, (nw, nh), interpolation=cv2.INTER_AREA)
    return left_resized, right_resized, left_resized, right_resized


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


def _rga_preprocess_gray_impl(
    left_gray: np.ndarray, right_gray: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Call native gray-direct path, or adapt through BGR path when missing."""
    if _RGA_BACKEND is None:
        raise RuntimeError("RGA backend module is not available")

    fn = getattr(_RGA_BACKEND, "preprocess_pair_gray_to_gray", None)
    if callable(fn):
        return fn(left_gray, right_gray, float(scale))

    left_bgr = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
    right_bgr = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR)
    _, _, left_out_gray, right_out_gray = _rga_preprocess_impl(left_bgr, right_bgr, float(scale))
    return left_out_gray, right_out_gray, left_out_gray, right_out_gray


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


def preprocess_pair_gray_to_gray(
    left_gray: np.ndarray, right_gray: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess grayscale stereo frames with RGA when available.

    Contract:
        Returns (left_resized_gray, right_resized_gray, left_gray, right_gray).
    """
    left_in = _to_gray_u8_2d(left_gray)
    right_in = _to_gray_u8_2d(right_gray)
    if not is_available():
        return _cpu_fallback_gray(left_in, right_in, float(scale))
    return _rga_preprocess_gray_impl(left_in, right_in, float(scale))
