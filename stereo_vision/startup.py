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


def _parse_bus_groups(text: str, source_count: int) -> Optional[list[str]]:
    """Parse bus/group labels provided as comma-separated text."""
    raw = str(text).strip()
    if raw == "":
        return None
    groups = [part.strip() for part in raw.split(",") if part.strip() != ""]
    if len(groups) != int(source_count):
        raise ValueError(
            "--bus-groups count must match number of inputs: "
            f"got {len(groups)} labels for {source_count} devices"
        )
    return groups


def initialize_capture(args: argparse.Namespace, device_list: list[str]) -> StartupResult:
    """Initialize capture based on CLI settings and return first valid stereo frame."""
    switch_timeout_s = max(0.05, float(args.switch_timeout_ms) / 1000.0)
    multi_mode = len(device_list) > 1
    cam: Optional[MultiStereoCamera | StereoCamera] = None
    active_idx = 0
    bus_groups = _parse_bus_groups(getattr(args, "bus_groups", ""), len(device_list))
    gst_pipeline_template = str(getattr(args, "gstreamer_pipeline", "") or "").strip()

    def resolve_gst_pipeline(device: str) -> Optional[str]:
        if gst_pipeline_template == "":
            return None
        try:
            return gst_pipeline_template.format(
                device=device,
                width=int(args.width),
                height=int(args.height),
                fps=int(args.fps),
            )
        except KeyError as exc:
            raise ValueError(
                "Invalid --gstreamer-pipeline template placeholder; "
                "allowed: {device}, {width}, {height}, {fps}"
            ) from exc

    cam_cfgs = [
        CameraConfig(
            device=device,
            width=args.width,
            height=args.height,
            fps=args.fps,
            use_gstreamer=args.gstreamer,
            gstreamer_pipeline=resolve_gst_pipeline(device),
            gstreamer_decode=str(getattr(args, "gst_decode", "auto")),
            gstreamer_output=str(getattr(args, "gst_output", "auto")),
            gstreamer_split_lr=bool(getattr(args, "gst_split_lr", False)),
            gstreamer_split_scale=float(getattr(args, "gst_split_scale", 0.5)),
            warmup_frames=max(0, int(args.warmup_frames)),
        )
        for device in device_list
    ]
    requested_idx = max(0, min(int(args.active_input) - 1, len(device_list) - 1))

    try:
        if multi_mode:
            if bus_groups is None:
                raise ValueError(
                    "Multi-input mode now requires --bus-groups. "
                    "Example: --bus-groups 0,0,2,2"
                )

            cam = MultiStereoCamera(
                cam_cfgs,
                single_active_mode=True,
                initial_active_index=requested_idx,
                group_ids=bus_groups,
                keep_one_live_per_group=True,
            )
            cam.start(stagger_s=0.0)
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
                        f"group={st.get('group', '-')} group_live={st.get('group_live', False)} "
                        f"age={age_text} frame_id={st['frame_id']} err={err_text}"
                    )

            try:
                t_req = time.perf_counter()
                cam.switch_to(requested_idx)
                left0, right0, _, _, active_idx = cam.read(
                    timeout_s=max(10.0, switch_timeout_s * 4.0),
                    min_timestamp_s=t_req,
                    allow_fallback=False,
                )
            except RuntimeError as exc:
                print(
                    "[WARN] Requested startup input is not ready; "
                    f"requested={requested_idx + 1}, reason={exc}"
                )
                print_source_statuses("startup")
                raise RuntimeError(
                    "Grouped linked single-active startup failed on requested input."
                ) from exc

            print(
                f"[INFO] Multi-input mode enabled: inputs={len(device_list)}, "
                f"active={active_idx + 1}, devices={device_list}"
            )
            print("[INFO] capture_mode=group-linked-single-active")
            print(f"[INFO] bus_groups={bus_groups}, group_live_mode={cam.group_live_mode}")
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