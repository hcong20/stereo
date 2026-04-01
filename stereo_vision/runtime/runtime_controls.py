"""Keyboard/runtime control handling for the stereo runtime loop."""

from __future__ import annotations

import time
from typing import Any, Callable

from stereo_vision.app_cli import decode_switch_index
from stereo_vision.runtime.runtime_switching import request_switch


def process_runtime_key_events(
    *,
    key_raw: int,
    swap_lr: bool,
    multi_mode: bool,
    device_list: list[str],
    active_idx: int,
    cam: Any,
    switch_state: Any,
    apply_roi_tune_preset: Callable[[str], None],
) -> tuple[bool, str | None, float, bool]:
    """Apply key-driven runtime actions and return updated UI/runtime flags."""
    key = key_raw & 0xFF
    runtime_note_text: str | None = None
    runtime_note_until = 0.0

    if key == ord("s"):
        swap_lr = not swap_lr

    preset_map = {
        ord("z"): "near",
        ord("x"): "mid",
        ord("c"): "far",
        ord("v"): "off",
    }
    if key in preset_map:
        preset = preset_map[key]
        apply_roi_tune_preset(preset)
        runtime_note_text = f"Preset switched: {preset}"
        runtime_note_until = time.perf_counter() + 2.0

    if multi_mode:
        req_idx = decode_switch_index(key_raw, len(device_list))
        if req_idx is not None and req_idx != active_idx:
            switch_state.pending = request_switch(
                cam=cam,
                from_idx=active_idx,
                to_idx=req_idx,
                key_label="number",
            )

    if multi_mode and key == ord("n"):
        next_idx = (active_idx + 1) % len(device_list)
        switch_state.pending = request_switch(
            cam=cam,
            from_idx=active_idx,
            to_idx=next_idx,
            key_label="next",
        )

    should_exit = key == 27 or key == ord("q")
    return swap_lr, runtime_note_text, runtime_note_until, should_exit
