"""Preprocessing backends for stereo frames.

This module provides a runtime-selectable preprocessing stage for RK3588
projects where RGA offload may be available through an external Python module.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Optional, Protocol

import cv2
import numpy as np


class RGABackendProtocol(Protocol):
    """Expected API for an optional RGA Python backend module."""

    def preprocess_pair_bgr_to_gray(
        self, left_bgr: np.ndarray, right_bgr: np.ndarray, scale: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return resized BGR pairs and grayscale pairs.

        Returns:
            (left_resized_bgr, right_resized_bgr, left_gray, right_gray)
        """


@dataclass
class PreprocessConfig:
    """Configuration for runtime preprocessing backend selection."""

    scale: float = 1.0
    backend: str = "auto"
    rga_module: str = "rga_helper"


class FramePreprocessor:
    """Preprocess rectified stereo frames using CPU or optional RGA backend."""

    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self._rga_backend: Optional[RGABackendProtocol] = None
        self._backend_reason: str = ""
        self._resolved_backend = self._resolve_backend()

    @property
    def backend_name(self) -> str:
        """Return the resolved backend name in use."""
        return self._resolved_backend

    @property
    def backend_reason(self) -> str:
        """Return explanatory detail about backend resolution."""
        return self._backend_reason

    def _resolve_backend(self) -> str:
        backend = str(self.cfg.backend).lower().strip()
        if backend not in {"auto", "cpu", "rga"}:
            raise ValueError("preprocess backend must be one of: auto, cpu, rga")

        if backend == "cpu":
            self._backend_reason = "forced by --preprocess-backend cpu"
            return "cpu"

        if backend == "rga":
            self._rga_backend = self._load_rga_backend(raise_on_failure=True)
            self._backend_reason = f"using module '{self.cfg.rga_module}'"
            return "rga"

        # auto mode: try RGA first, then fallback to CPU.
        self._rga_backend = self._load_rga_backend(raise_on_failure=False)
        if self._rga_backend is not None:
            self._backend_reason = f"auto-selected module '{self.cfg.rga_module}'"
            return "rga"
        self._backend_reason = (
            f"auto fallback to CPU; RGA module '{self.cfg.rga_module}' is missing or unavailable"
        )
        return "cpu"

    def _load_rga_backend(self, raise_on_failure: bool) -> Optional[RGABackendProtocol]:
        try:
            mod = import_module(self.cfg.rga_module)
            fn = getattr(mod, "preprocess_pair_bgr_to_gray", None)
            if fn is None:
                raise AttributeError(
                    f"Module '{self.cfg.rga_module}' missing preprocess_pair_bgr_to_gray()"
                )
            is_available_fn = getattr(mod, "is_available", None)
            if callable(is_available_fn) and not bool(is_available_fn()):
                raise RuntimeError(f"Module '{self.cfg.rga_module}' reports RGA unavailable")
            return mod
        except Exception as exc:  # pragma: no cover - backend is optional
            if raise_on_failure:
                raise RuntimeError(
                    "RGA backend requested but unavailable. "
                    f"Install/load Python module '{self.cfg.rga_module}' with "
                    "preprocess_pair_bgr_to_gray(left,right,scale)."
                ) from exc
            return None

    @staticmethod
    def _cpu_resize(frame: np.ndarray, scale: float) -> np.ndarray:
        if scale == 1.0:
            return frame
        h, w = frame.shape[:2]
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

    def process(
        self, left_bgr: np.ndarray, right_bgr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Resize and convert stereo pair to grayscale for disparity.

        Returns:
            (left_resized_bgr, right_resized_bgr, left_gray, right_gray)
        """
        if self._resolved_backend == "rga" and self._rga_backend is not None:
            return self._rga_backend.preprocess_pair_bgr_to_gray(
                left_bgr, right_bgr, float(self.cfg.scale)
            )

        left_resized = self._cpu_resize(left_bgr, float(self.cfg.scale))
        right_resized = self._cpu_resize(right_bgr, float(self.cfg.scale))
        left_gray = cv2.cvtColor(left_resized, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_resized, cv2.COLOR_BGR2GRAY)
        return left_resized, right_resized, left_gray, right_gray
