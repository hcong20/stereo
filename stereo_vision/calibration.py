"""Load stereo camera calibration parameters from .npz files.

This module normalizes key names from different calibration toolchains into a
single strongly-typed dataclass used by the rest of the stereo pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np


@dataclass
class StereoCalibration:
    """Container for intrinsic/extrinsic stereo calibration matrices.

    Attributes follow OpenCV stereo conventions (K, D, R, T, P, Q).
    Optional fields are populated when precomputed rectification data is stored.
    """

    k_left: np.ndarray
    d_left: np.ndarray
    k_right: np.ndarray
    d_right: np.ndarray
    r: np.ndarray
    t: np.ndarray
    r1: np.ndarray | None = None
    r2: np.ndarray | None = None
    p1: np.ndarray | None = None
    p2: np.ndarray | None = None
    q: np.ndarray | None = None

    @property
    def baseline_m(self) -> float:
        """Return stereo baseline magnitude from translation vector T.

        Returns:
            Euclidean norm of translation vector in calibration units.
        """
        return float(np.linalg.norm(self.t))


def _get_any(data: Dict[str, Any], keys: list[str]) -> Any:
    """Return first present key from aliases, or raise if none exist.

    Args:
        data: Loaded npz dictionary.
        keys: Candidate aliases for the same logical field.
    """
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"Missing keys. Tried: {keys}")


def _get_any_optional(data: Dict[str, Any], keys: list[str]) -> Any | None:
    """Return first present key from aliases, or None when absent."""
    for key in keys:
        if key in data:
            return data[key]
    return None


def load_stereo_calibration(calib_path: str) -> StereoCalibration:
    """Load and normalize stereo calibration arrays from a .npz file.

    Args:
        calib_path: Path to calibration archive.

    Returns:
        Parsed StereoCalibration with float64 matrices.
    """
    path = Path(calib_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    if path.suffix.lower() != ".npz":
        raise ValueError("Calibration file must be a .npz file")

    with np.load(path, allow_pickle=False) as npz_data:
        data: Dict[str, Any] = {k: npz_data[k] for k in npz_data.files}

    # Accept multiple naming conventions produced by different calibration tools.
    k_left = np.array(
        _get_any(data, ["K1", "K_left", "K_l", "camera_matrix_left", "M1"]),
        dtype=np.float64,
    )
    d_left = np.array(
        _get_any(data, ["D1", "D_left", "dist_l", "dist_coeffs_left"]),
        dtype=np.float64,
    )
    k_right = np.array(
        _get_any(data, ["K2", "K_right", "K_r", "camera_matrix_right", "M2"]),
        dtype=np.float64,
    )
    d_right = np.array(
        _get_any(data, ["D2", "D_right", "dist_r", "dist_coeffs_right"]),
        dtype=np.float64,
    )
    # Keep matrix precision high to avoid accumulating geometric errors.
    r = np.array(_get_any(data, ["R", "R_stereo", "rotation_matrix"]), dtype=np.float64)
    t = np.array(_get_any(data, ["T", "T_stereo", "translation_vector"]), dtype=np.float64).reshape(3, 1)
    r1 = _get_any_optional(data, ["R1"])
    r2 = _get_any_optional(data, ["R2"])
    p1 = _get_any_optional(data, ["P1"])
    p2 = _get_any_optional(data, ["P2"])
    q = _get_any_optional(data, ["Q"])

    return StereoCalibration(
        k_left=k_left,
        d_left=d_left,
        k_right=k_right,
        d_right=d_right,
        r=r,
        t=t,
        r1=None if r1 is None else np.array(r1, dtype=np.float64),
        r2=None if r2 is None else np.array(r2, dtype=np.float64),
        p1=None if p1 is None else np.array(p1, dtype=np.float64),
        p2=None if p2 is None else np.array(p2, dtype=np.float64),
        q=None if q is None else np.array(q, dtype=np.float64),
    )
