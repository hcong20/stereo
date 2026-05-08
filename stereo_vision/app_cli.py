"""CLI and lightweight runtime helpers for stereo app entrypoints."""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from stereo_vision.core.roi import ROI


_CONFIG_FILENAME = "device_profiles.json"


def _extract_config_arg(argv: list[str]) -> Optional[str]:
    """Return the raw --config value from argv if present."""
    for idx, arg in enumerate(argv):
        if arg == "--config" and idx + 1 < len(argv):
            return argv[idx + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


def _detect_os_profile() -> str:
    """Return the most specific OS profile name available."""
    if sys.platform == "darwin":
        return "macos"

    if sys.platform.startswith("linux"):
        try:
            os_release: dict[str, str] = {}
            with open("/etc/os-release", encoding="utf-8") as handle:
                for line in handle:
                    if "=" not in line:
                        continue
                    key, value = line.rstrip().split("=", 1)
                    os_release[key.strip().lower()] = value.strip().strip('"')

            distro_id = os_release.get("id", "").strip().lower()
            if distro_id in {"ubuntu", "debian"}:
                return distro_id

            distro_like = os_release.get("id_like", "").lower().split()
            for candidate in ("ubuntu", "debian"):
                if candidate in distro_like:
                    return candidate
        except OSError:
            pass

        return "linux"

    return "default"


def _default_config_path() -> Path:
    """Return the in-tree config file location."""
    return Path(__file__).resolve().parent / "config" / _CONFIG_FILENAME


def _resolve_config_path(explicit_path: Optional[str]) -> Optional[Path]:
    """Resolve the config file path from CLI, env, or bundled defaults."""
    if explicit_path:
        explicit = Path(explicit_path).expanduser()
        if explicit.is_file():
            return explicit
        raise FileNotFoundError(f"Config file not found: {explicit_path}")

    candidates: list[Path] = []

    env_path = os.environ.get("STEREO_VISION_CONFIG", "").strip()
    if env_path:
        env_config = Path(env_path).expanduser()
        if env_config.is_file():
            return env_config
        raise FileNotFoundError(f"Config file not found from STEREO_VISION_CONFIG: {env_path}")

    candidates.append(_default_config_path())

    cwd_path = Path.cwd() / _CONFIG_FILENAME
    if cwd_path not in candidates:
        candidates.append(cwd_path)

    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_config_file(path: Path) -> dict[str, object]:
    """Load a JSON config file containing default values and OS profiles."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return data


def _config_defaults_from_data(config_data: dict[str, object], profile_name: str) -> dict[str, object]:
    """Merge default and OS-specific config values."""
    merged: dict[str, object] = {}

    default_section = config_data.get("default")
    if isinstance(default_section, dict):
        merged.update(default_section)

    profiles = config_data.get("profiles")
    if isinstance(profiles, dict):
        linux_section = profiles.get("linux")
        if profile_name in {"ubuntu", "debian", "linux"} and isinstance(linux_section, dict):
            merged.update(linux_section)

        profile_section = profiles.get(profile_name)
        if isinstance(profile_section, dict):
            merged.update(profile_section)

    source_map = merged.get("source_map")
    if isinstance(source_map, list):
        devices: list[str] = []
        directions: list[str] = []
        bus_groups: list[str] = []
        has_bus_groups = False

        for entry in source_map:
            if not isinstance(entry, dict):
                continue
            device = str(entry.get("device", "")).strip()
            direction = str(entry.get("direction", "")).strip()
            bus_group = str(entry.get("usb_bus_group")).strip()
            if device:
                devices.append(device)
            if direction:
                directions.append(direction)
            if bus_group:
                bus_groups.append(bus_group)
                has_bus_groups = True

        if devices:
            merged["devices"] = ",".join(devices)
            merged["device"] = devices[0]
        if directions:
            merged["directions"] = ",".join(directions)
        if has_bus_groups:
            merged["usb_bus_groups"] = ",".join(bus_groups)

    return merged


def _load_runtime_defaults(argv: Optional[list[str]] = None) -> tuple[dict[str, object], Optional[Path], str]:
    """Load config-backed defaults for the current OS profile."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    explicit_config_arg = _extract_config_arg(args_list)
    config_path = _resolve_config_path(explicit_config_arg)
    profile_name = _detect_os_profile()
    defaults: dict[str, object] = {}

    if config_path is not None:
        config_data = _load_config_file(config_path)
        defaults = _config_defaults_from_data(config_data, profile_name)

    return defaults, config_path, profile_name


@dataclass
class PerfStats:
    """Track runtime throughput for on-screen FPS reporting."""

    frame_count: int = 0
    start: float = time.perf_counter()

    def update_fps(self) -> float:
        """Update frame counter and return average FPS since start."""
        self.frame_count += 1
        elapsed = max(1e-6, time.perf_counter() - self.start)
        return self.frame_count / elapsed


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI options for camera, disparity, depth, and filtering."""
    config_defaults, config_path, profile_name = _load_runtime_defaults(argv)
    parser = argparse.ArgumentParser(description="RK3588 Stereo Distance Measurement")

    parser.add_argument(
        "--config",
        default=str(config_path) if config_path is not None else "",
        help="Optional JSON config file with OS-specific defaults",
    )
    parser.add_argument("--device", default=str(config_defaults.get("device", "0")))
    parser.add_argument(
        "--devices",
        default=str(config_defaults.get("devices", "")),
        help="Comma-separated stereo input devices, e.g. /dev/video20,/dev/video22,/dev/video24,/dev/video26",
    )
    parser.add_argument(
        "--directions",
        default=str(config_defaults.get("directions", "")),
        help="Optional override for direction labels; normally derived from config source_map",
    )
    parser.add_argument(
        "--active-input",
        type=int,
        default=int(config_defaults.get("active_input", 1)),
        help="1-based input index selected at startup when --devices is used",
    )
    parser.add_argument(
        "--switch-timeout-ms",
        type=float,
        default=float(config_defaults.get("switch_timeout_ms", 2000.0)),
        help="Max wait for frame from selected input after switching",
    )
    parser.add_argument(
        "--calib",
        default=str(config_defaults.get("calib", "")),
        help="Optional calibration archive path; leave blank to auto-select a per-device file",
    )
    parser.add_argument("--width", type=int, default=int(config_defaults.get("width", 1280)))
    parser.add_argument("--height", type=int, default=int(config_defaults.get("height", 480)))
    parser.add_argument("--fps", type=int, default=int(config_defaults.get("fps", 30)))
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=int(config_defaults.get("warmup_frames", 1)),
        help="Frames discarded after camera open; lower values reduce switch latency",
    )
    parser.add_argument(
        "--usb-bus-groups",
        dest="usb_bus_groups",
        default=str(config_defaults.get("usb_bus_groups")),
        help=(
            "Comma-separated USB bus-group labels for each input device, "
            "e.g. front_right,front_right,back_left,back_left for 4 inputs. "
            "Inputs with the same label share one bus path and must not be started at the same time."
        ),
    )
    parser.add_argument(
        "--gstreamer",
        dest="gstreamer",
        action="store_true",
        default=bool(config_defaults.get("gstreamer", False)),
        help="Use OpenCV CAP_GSTREAMER backend for camera capture (default: disabled)",
    )
    parser.add_argument(
        "--gstreamer-pipeline",
        "--gst-pipeline",
        dest="gstreamer_pipeline",
        default=str(config_defaults.get("gstreamer_pipeline", "")),
        help=(
            "Optional custom GStreamer pipeline template. "
            "Supports placeholders {device}, {width}, {height}, {fps}."
        ),
    )
    parser.add_argument(
        "--gst-decode",
        choices=["auto", "hw", "sw"],
        default=str(config_defaults.get("gst_decode", "auto")),
        help=(
            "GStreamer MJPEG decode path: auto prefers RK3588 hardware decode (mppjpegdec) "
            "with software fallback"
        ),
    )
    parser.add_argument(
        "--gst-output",
        choices=["auto", "nv12", "bgr"],
        default=str(config_defaults.get("gst_output", "auto")),
        help=(
            "GStreamer output format: auto prefers NV12 (lower CPU path) with BGR fallback"
        ),
    )
    parser.add_argument(
        "--nv12-preview-bgr",
        action="store_true",
        default=bool(config_defaults.get("nv12_preview_bgr", False)),
        help=(
            "When using NV12 GStreamer output, convert frames to BGR for preview only "
            "while keeping grayscale matching path"
        ),
    )
    parser.add_argument(
        "--quiet-opencv-log",
        dest="quiet_opencv_log",
        action="store_true",
        help="Reduce OpenCV runtime log level to ERROR (hides non-fatal GStreamer WARN messages)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run in headless mode without OpenCV window display (logs/metrics only)",
    )
    parser.add_argument("--swap-lr", action="store_true", help="Swap left/right camera halves")
    parser.add_argument(
        "--use-precomputed-rect",
        action="store_true",
        help="Use R1/R2/P1/P2/Q from calibration if available",
    )

    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--crop-height-ratio",
        type=float,
        default=1,
        help=(
            "Vertical center-crop ratio applied after resize and before disparity. "
            "Must be > 0; 1.0 disables crop"
        ),
    )
    parser.add_argument("--roi", type=str, default="270,175,100,70", help="x,y,w,h")
    parser.add_argument(
        "--roi-physical-size-mm",
        type=str,
        default="800,100",
        help="Physical ROI size in millimeters as w,h (default: 800,100)",
    )
    parser.add_argument(
        "--roi-physical-center",
        choices=["image-center", "static-roi-center"],
        default="image-center",
        help="Physical ROI center reference",
    )
    parser.add_argument("--roi-disparity-only", action="store_true")

    parser.add_argument("--num-disp", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--min-disp", type=int, default=0)
    parser.add_argument("--depth-min-disp", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--max-jump", type=float, default=1.0)
    parser.add_argument("--filter-window", type=int, default=5)
    parser.add_argument("--min-valid-pixels", type=int, default=10)
    parser.add_argument(
        "--roi-valid-ratio-min",
        type=float,
        default=0.15,
        help="Minimum finite-depth ratio inside ROI required to accept a raw measurement",
    )
    parser.add_argument(
        "--roi-p10-weight",
        type=float,
        default=0.70,
        help="Weight of P10 in blended robust distance (remaining weight goes to median)",
    )
    parser.add_argument(
        "--roi-min-weight",
        type=float,
        default=0.10,
        help="Extra blend weight for minimum depth after P10/median fusion",
    )
    parser.add_argument(
        "--roi-tune-preset",
        choices=["off", "near", "mid", "far"],
        default="off",
        help=(
            "Apply tested presets for ROI distance gating/smoothing. "
            "off keeps manual values; near/mid/far are field-tuning shortcuts"
        ),
    )
    parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="Print per-stage average latency (capture/rectify/preprocess/disparity/depth/viz)",
    )
    parser.add_argument(
        "--profile-interval",
        type=int,
        default=60,
        help="Frames per profiling report when --profile-stages is enabled",
    )
    parser.add_argument(
        "--log-measurements",
        action="store_true",
        help=(
            "Log timestamped distance and FPS measurements during runtime. "
            "Output is printed to stdout and can optionally be written to CSV."
        ),
    )
    parser.add_argument(
        "--log-interval-ms",
        type=float,
        default=250.0,
        help="Measurement logging interval in milliseconds when --log-measurements is enabled",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="",
        help="Optional CSV file path for measurement logs",
    )

    args = parser.parse_args(argv)
    if config_path is not None:
        print(f"[INFO] Loaded config defaults from {config_path} (profile={profile_name})")
    return args


def parse_roi(text: str) -> ROI:
    """Parse ROI text in x,y,w,h format into an ROI object."""
    vals = [int(v.strip()) for v in text.split(",")]
    if len(vals) != 4:
        raise ValueError("ROI must be x,y,w,h")
    return ROI(*vals)


def parse_physical_size_mm(text: str) -> tuple[float, float]:
    """Parse physical size text in w,h millimeters and return meters."""
    vals = [float(v.strip()) for v in text.split(",")]
    if len(vals) != 2:
        raise ValueError("Physical ROI size must be w,h in millimeters")
    w_m = max(1e-3, vals[0] / 1000.0)
    h_m = max(1e-3, vals[1] / 1000.0)
    return w_m, h_m


def get_screen_size() -> tuple[int, int] | None:
    """Best-effort screen size query for window centering."""
    if sys.platform == "darwin":
        # Tkinter screen probing can abort on some macOS/Python builds.
        return None
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = int(root.winfo_screenwidth())
        height = int(root.winfo_screenheight())
        root.destroy()
        if width > 0 and height > 0:
            return width, height
    except Exception:
        return None
    return None


def safe_num_disparities_for_roi(requested: int, roi_width: int) -> int:
    """Choose safe SGBM disparities for ROI mode."""
    req = max(16, (int(requested) // 16) * 16)
    max_safe = ((max(0, int(roi_width)) // 16) - 1) * 16
    if max_safe < 16:
        raise ValueError(
            f"ROI width={roi_width} is too small for SGBM ROI mode. Increase ROI width to at least 32 pixels."
        )
    return min(req, max_safe)


def fourcc_to_str(fourcc_value: float) -> str:
    """Decode OpenCV FOURCC numeric code into a readable 4-char string."""
    code = int(fourcc_value)
    return "".join([chr((code >> (8 * i)) & 0xFF) for i in range(4)])


def decode_switch_index(key_raw: int, source_count: int) -> Optional[int]:
    """Decode input-switch index from OpenCV key code."""
    if source_count <= 0 or key_raw < 0:
        return None

    # Standard ASCII digits from top keyboard row.
    low = key_raw & 0xFF
    if ord("1") <= low <= ord("9"):
        idx = int(low - ord("1"))
        return idx if idx < source_count else None

    # X11 keypad keys (Linux): XK_KP_1..XK_KP_9 (0xFFB1..0xFFB9).
    if 0xFFB1 <= key_raw <= 0xFFB9:
        idx = int(key_raw - 0xFFB1)
        return idx if idx < source_count else None

    return None