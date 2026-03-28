"""CLI and lightweight runtime helpers for stereo app entrypoints."""

import argparse
import time
from dataclasses import dataclass
from typing import Optional

from stereo_vision.roi import ROI


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


def parse_args() -> argparse.Namespace:
    """Parse CLI options for camera, disparity, depth, and filtering."""
    parser = argparse.ArgumentParser(description="RK3588 Stereo Distance Measurement")

    parser.add_argument("--device", default="/dev/video20")
    parser.add_argument(
        "--devices",
        default="",
        help="Comma-separated stereo input devices, e.g. /dev/video20,/dev/video22,/dev/video24,/dev/video26",
    )
    parser.add_argument(
        "--active-input",
        type=int,
        default=1,
        help="1-based input index selected at startup when --devices is used",
    )
    parser.add_argument(
        "--switch-timeout-ms",
        type=float,
        default=500.0,
        help="Max wait for frame from selected input after switching",
    )
    parser.add_argument("--calib", default="stereo_calib_params.npz")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=1,
        help="Frames discarded after camera open; lower values reduce switch latency",
    )
    parser.add_argument(
        "--skip-prewarm-inputs",
        action="store_true",
        help="Skip startup prewarm for non-active inputs in single-active mode",
    )
    parser.add_argument(
        "--capture-mode",
        choices=["auto", "parallel", "single-active"],
        default="auto",
        help=(
            "Multi-input capture strategy: parallel opens all inputs concurrently; "
            "single-active keeps only selected input streaming; auto prefers parallel and "
            "falls back to single-active when parallel availability is poor."
        ),
    )
    parser.add_argument(
        "--bus-groups",
        default="",
        help=(
            "Comma-separated bus/group label for each input device, "
            "e.g. 0,0,2,2 for 4 inputs"
        ),
    )
    parser.add_argument(
        "--keep-one-live-per-group",
        action="store_true",
        help=(
            "In single-active mode, keep one source live in each bus group "
            "(requires --bus-groups)."
        ),
    )
    parser.add_argument("--gstreamer", action="store_true")
    parser.add_argument("--swap-lr", action="store_true", help="Swap left/right camera halves")
    parser.add_argument(
        "--use-precomputed-rect",
        action="store_true",
        help="Use R1/R2/P1/P2/Q from calibration if available",
    )

    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--preprocess-backend",
        choices=["auto", "cpu", "rga"],
        default="auto",
        help="Frame preprocess backend (RGA requires external module)",
    )
    parser.add_argument(
        "--rga-module",
        type=str,
        default="rga_helper",
        help="Python module name providing RGA preprocess_pair_bgr_to_gray",
    )
    parser.add_argument(
        "--allow-unsafe-rga-multicam",
        action="store_true",
        help=(
            "Allow RGA preprocess in multi-camera mode. "
            "Disabled by default because some RGA stacks can hang the system during source switching."
        ),
    )
    parser.add_argument("--roi", type=str, default="270,175,100,70", help="x,y,w,h")
    parser.add_argument("--roi-disparity-only", action="store_true")

    parser.add_argument("--num-disp", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--min-disp", type=int, default=0)
    parser.add_argument("--depth-min-disp", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument(
        "--baseline-unit",
        choices=["auto", "m", "mm"],
        default="auto",
        help="Unit of T baseline stored in calibration",
    )

    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--max-jump", type=float, default=1.0)
    parser.add_argument("--filter-window", type=int, default=5)
    parser.add_argument("--min-valid-pixels", type=int, default=10)

    return parser.parse_args()


def parse_roi(text: str) -> ROI:
    """Parse ROI text in x,y,w,h format into an ROI object."""
    vals = [int(v.strip()) for v in text.split(",")]
    if len(vals) != 4:
        raise ValueError("ROI must be x,y,w,h")
    return ROI(*vals)


def get_screen_size() -> tuple[int, int] | None:
    """Best-effort screen size query for window centering."""
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


def resolve_baseline_m(raw_baseline: float, baseline_unit: str) -> float:
    """Convert baseline value to meters using explicit or inferred unit."""
    if baseline_unit == "m":
        return raw_baseline
    if baseline_unit == "mm":
        return raw_baseline / 1000.0
    # Auto: most stereo rigs are in cm-scale baseline; values >2 are likely mm.
    if raw_baseline > 2.0:
        return raw_baseline / 1000.0
    return raw_baseline


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