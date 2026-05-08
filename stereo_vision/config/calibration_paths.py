"""Helpers for resolving per-device stereo calibration file paths."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


_DEFAULT_CALIB_FILENAME = "stereo_calib_params.npz"


def _slugify_component(text: str) -> str:
    """Convert a device identifier into a filename-safe slug."""
    slug_parts: list[str] = []
    last_was_separator = False

    for char in str(text).strip():
        if char.isalnum():
            slug_parts.append(char.lower())
            last_was_separator = False
        elif not last_was_separator:
            slug_parts.append("_")
            last_was_separator = True

    slug = "".join(slug_parts).strip("_")
    return slug or "device"


def calibration_filename_for_devices(devices: Sequence[str]) -> str:
    """Build a stable calibration filename for the provided devices."""
    cleaned = [str(device).strip() for device in devices if str(device).strip()]
    if not cleaned:
        return _DEFAULT_CALIB_FILENAME

    device_slug = "__".join(_slugify_component(device) for device in cleaned)
    return f"stereo_calib_{device_slug}.npz"


def calibration_filename_for_direction(direction: str) -> str:
    """Build the calibration filename for a single named direction."""
    return f"stereo_calib_{_slugify_component(direction)}.npz"


def calibration_path_for_devices(devices: Sequence[str], base_dir: str | Path | None = None) -> Path:
    """Return the default save path for a given device set."""
    directory = Path(base_dir) if base_dir is not None else Path.cwd()
    return directory / calibration_filename_for_devices(devices)


def calibration_path_for_device(device: str, base_dir: str | Path | None = None) -> Path:
    """Return the default save path for a single camera/device."""
    return calibration_path_for_devices([device], base_dir=base_dir)


def calibration_path_for_direction(direction: str, base_dir: str | Path | None = None) -> Path:
    """Return the default save path for a single direction label."""
    directory = Path(base_dir) if base_dir is not None else Path.cwd()
    return directory / calibration_filename_for_direction(direction)


def resolve_existing_calibration_path(
    calib_arg: str | None,
    devices: Sequence[str],
    search_dirs: Sequence[str | Path] | None = None,
) -> Path:
    """Resolve the calibration archive to load, preferring device-specific files."""
    if calib_arg and str(calib_arg).strip():
        return Path(calib_arg).expanduser()

    candidates: list[Path] = [calibration_path_for_devices(devices)]

    cwd_default = Path.cwd() / _DEFAULT_CALIB_FILENAME
    if cwd_default not in candidates:
        candidates.append(cwd_default)

    for directory in search_dirs or []:
        candidate = Path(directory) / _DEFAULT_CALIB_FILENAME
        if candidate not in candidates:
            candidates.append(candidate)

    package_default = Path(__file__).resolve().parents[2] / _DEFAULT_CALIB_FILENAME
    if package_default not in candidates:
        candidates.append(package_default)

    for path in candidates:
        if path.is_file():
            return path

    return candidates[0]
