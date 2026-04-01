#!/usr/bin/env python3
"""Entry point for real-time stereo distance measurement and visualization.

This script wires together capture, rectification, disparity estimation,
depth conversion, robust ROI distance extraction, temporal smoothing,
and live OpenCV visualization.
"""

import cv2

from stereo_vision.app_cli import parse_args
from stereo_vision.runtime_builder import build_runtime_context
from stereo_vision.runtime_loop import run_runtime_loop


def _quiet_opencv_logs() -> None:
    """Lower OpenCV logging verbosity in a version-compatible way."""
    try:
        if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
            level = getattr(cv2.utils.logging, "LOG_LEVEL_ERROR", None)
            if level is not None:
                cv2.utils.logging.setLogLevel(level)
                return
        if hasattr(cv2, "setLogLevel"):
            level = getattr(cv2, "LOG_LEVEL_ERROR", None)
            if level is None:
                level = 2
            cv2.setLogLevel(level)
    except Exception:
        # Logging control differs between OpenCV builds; ignore unsupported APIs.
        return


def main() -> None:
    """Run the full stereo pipeline loop and render diagnostic overlays.

    The loop sequence is:
    capture -> rectify -> resize -> disparity -> depth -> ROI measurement
    -> temporal filtering -> visualization.
    """
    args = parse_args()
    if bool(getattr(args, "quiet_opencv_log", False)):
        _quiet_opencv_logs()

    ctx = build_runtime_context(args)

    try:
        run_runtime_loop(args=args, cfg=ctx)

    finally:
        if ctx.cam is not None:
            ctx.cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
