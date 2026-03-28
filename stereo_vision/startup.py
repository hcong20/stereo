"""Camera startup orchestration for stereo app."""

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from stereo_vision.camera_multi import MultiStereoCamera
from stereo_vision.camera_single import CameraConfig, StereoCamera


@dataclass
class StartupResult:
    """Capture startup state passed into the runtime loop."""

    cam: MultiStereoCamera | StereoCamera
    multi_mode: bool
    switch_timeout_s: float
    active_idx: int
    left0: np.ndarray
    right0: np.ndarray


def initialize_capture(args: argparse.Namespace, device_list: list[str]) -> StartupResult:
    """Initialize capture based on CLI settings and return first valid stereo frame."""
    switch_timeout_s = max(0.05, float(args.switch_timeout_ms) / 1000.0)
    multi_mode = len(device_list) > 1
    cam: Optional[MultiStereoCamera | StereoCamera] = None
    active_idx = 0

    cam_cfgs = [
        CameraConfig(
            device=device,
            width=args.width,
            height=args.height,
            fps=args.fps,
            use_gstreamer=args.gstreamer,
            warmup_frames=max(0, int(args.warmup_frames)),
        )
        for device in device_list
    ]
    requested_idx = max(0, min(int(args.active_input) - 1, len(device_list) - 1))

    try:
        if multi_mode:
            capture_mode = str(args.capture_mode)
            single_active_mode = capture_mode == "single-active"
            cam = MultiStereoCamera(
                cam_cfgs,
                single_active_mode=single_active_mode,
                initial_active_index=requested_idx,
            )
            cam.start(stagger_s=0.15)
            if requested_idx != int(args.active_input) - 1:
                print(
                    "[WARN] --active-input is out of range; "
                    f"clamped to {requested_idx + 1}/{len(device_list)}"
                )

            def print_source_statuses(tag: str) -> None:
                statuses = cam.source_statuses()
                print(f"[INFO] {tag} source status:")
                for st in statuses:
                    age_text = "N/A" if st["frame_age_ms"] is None else f"{st['frame_age_ms']:.0f}ms"
                    err_text = st["last_error"] if st["last_error"] else "-"
                    print(
                        "[INFO] "
                        f"input={st['index'] + 1} dev={st['device']} has_frame={st['has_frame']} "
                        f"age={age_text} frame_id={st['frame_id']} err={err_text}"
                    )

            try:
                left0, right0, _, _, active_idx = cam.read(
                    timeout_s=max(8.0, switch_timeout_s * 4.0),
                    allow_fallback=False,
                )
            except RuntimeError as exc:
                print(
                    "[INFO] Requested startup input is not ready yet; "
                    f"requested={requested_idx + 1}, reason={exc}"
                )
                print_source_statuses("startup")

                if capture_mode == "auto":
                    statuses = cam.source_statuses()
                    ready_count = sum(1 for st in statuses if st["has_frame"])
                    if ready_count <= 1:
                        print(
                            "[WARN] Parallel capture appears constrained "
                            f"(ready_inputs={ready_count}/{len(statuses)}). "
                            "Auto-switching to single-active capture mode."
                        )
                        cam.release()
                        time.sleep(0.2)
                        cam = MultiStereoCamera(
                            cam_cfgs,
                            single_active_mode=True,
                            initial_active_index=requested_idx,
                        )
                        cam.start(stagger_s=0.0)

                # Retry requested input first; only allow fallback when strict
                # retry still fails.
                t_retry = time.perf_counter()
                cam.switch_to(requested_idx)
                try:
                    left0, right0, _, _, active_idx = cam.read(
                        timeout_s=max(10.0, switch_timeout_s * 4.0),
                        min_timestamp_s=t_retry,
                        allow_fallback=False,
                        max_fallback_age_s=0.8,
                    )
                    print(
                        "[INFO] Startup recovered on requested input: "
                        f"input={active_idx + 1}"
                    )
                except RuntimeError:
                    left0, right0, _, _, active_idx = cam.read(
                        timeout_s=max(10.0, switch_timeout_s * 4.0),
                        allow_fallback=True,
                    )
                    if active_idx != requested_idx:
                        print(
                            "[WARN] Startup fallback applied: "
                            f"requested={requested_idx + 1}, using={active_idx + 1}"
                        )
                    else:
                        print(
                            "[INFO] Requested startup input became ready after retry: "
                            f"input={active_idx + 1}"
                        )

            print(
                f"[INFO] Multi-input mode enabled: inputs={len(device_list)}, "
                f"active={active_idx + 1}, devices={device_list}"
            )
            print(f"[INFO] capture_mode={args.capture_mode}, single_active_mode={cam.single_active_mode}")

            if cam.single_active_mode and not bool(args.skip_prewarm_inputs):
                prewarm_timeout_s = max(3.0, switch_timeout_s)
                active_start_idx = active_idx
                print(
                    "[INFO] Prewarming non-active inputs for faster first switch "
                    f"(timeout={prewarm_timeout_s:.1f}s each)"
                )
                for idx in range(len(device_list)):
                    if idx == active_start_idx:
                        continue
                    t_sw = time.perf_counter()
                    cam.switch_to(idx)
                    try:
                        cam.read(
                            timeout_s=prewarm_timeout_s,
                            min_timestamp_s=t_sw,
                            allow_fallback=False,
                            max_fallback_age_s=0.8,
                        )
                        print(f"[INFO] prewarm input {idx + 1}: ready")
                    except Exception as exc:
                        print(f"[WARN] prewarm input {idx + 1} failed: {exc}")

                t_sw = time.perf_counter()
                cam.switch_to(active_start_idx)
                left0, right0, _, _, active_idx = cam.read(
                    timeout_s=max(3.0, switch_timeout_s),
                    min_timestamp_s=t_sw,
                    allow_fallback=False,
                    max_fallback_age_s=0.8,
                )
                print(f"[INFO] prewarm complete; restored active input {active_idx + 1}")
        else:
            cam = StereoCamera(cam_cfgs[0])
            cam.open()
            left0, right0, _ = cam.read()
            active_idx = 0
    except Exception:
        if cam is not None:
            cam.release()
        raise

    return StartupResult(
        cam=cam,
        multi_mode=multi_mode,
        switch_timeout_s=switch_timeout_s,
        active_idx=active_idx,
        left0=left0,
        right0=right0,
    )