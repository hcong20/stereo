"""CPU preprocessing for stereo frames."""

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    """Configuration for CPU preprocessing."""

    scale: float = 1.0
    crop_height_ratio: float = 0.5


class FramePreprocessor:
    """Preprocess rectified stereo frames using CPU only."""

    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self._backend_reason = "cpu-only preprocess path"

    @property
    def backend_name(self) -> str:
        """Return active preprocess backend name."""
        return "cpu"

    @property
    def backend_reason(self) -> str:
        """Return explanatory detail about backend selection."""
        return self._backend_reason

    @staticmethod
    def _cpu_resize(frame: np.ndarray, scale: float) -> np.ndarray:
        if scale == 1.0:
            return frame
        h, w = frame.shape[:2]
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _center_crop_height(frame: np.ndarray, ratio: float) -> np.ndarray:
        """Center-crop frame vertically by ratio while preserving width."""
        if ratio >= 1.0:
            return frame
        if ratio <= 0.0:
            raise ValueError(f"crop_height_ratio must be > 0, got {ratio}")
        h, _ = frame.shape[:2]
        target_h = max(1, int(h * ratio))
        if target_h >= h:
            return frame
        y0 = (h - target_h) // 2
        y1 = y0 + target_h
        return frame[y0:y1, :]


    @staticmethod
    def _to_gray_if_needed(frame: np.ndarray) -> np.ndarray:
        """Return grayscale view of input frame with minimal conversion."""
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3 and frame.shape[2] == 1:
            return frame[:, :, 0]
        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        raise ValueError(f"Unsupported frame shape for grayscale conversion: {frame.shape}")

    def process(
        self, left_img: np.ndarray, right_img: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Resize/crop stereo pair and produce grayscale images for disparity.

        Returns:
            (left_resized, right_resized, left_gray, right_gray)
        """
        left_resized = self._cpu_resize(left_img, float(self.cfg.scale))
        right_resized = self._cpu_resize(right_img, float(self.cfg.scale))
        left_processed = self._center_crop_height(left_resized, float(self.cfg.crop_height_ratio))
        right_processed = self._center_crop_height(right_resized, float(self.cfg.crop_height_ratio))
        left_gray = self._to_gray_if_needed(left_processed)
        right_gray = self._to_gray_if_needed(right_processed)
        return left_processed, right_processed, left_gray, right_gray
