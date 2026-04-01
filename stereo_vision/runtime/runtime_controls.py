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
    """Apply keyboard actions and return updated runtime/UI state.

    Args:
        key_raw: Raw key code from ``cv2.waitKeyEx``.
        swap_lr: Current left/right swap state.
        multi_mode: Whether multi-input switching mode is enabled.
        device_list: Ordered capture device list for source switching.
        active_idx: Current active device index (0-based).
        cam: Camera manager used to request source switches.
        switch_state: Mutable runtime switch state container.
        apply_roi_tune_preset: Callback applying ROI tuning preset by name.

    Returns:
        Tuple ``(swap_lr, runtime_note_text, runtime_note_until, should_exit)`` where:
        - ``swap_lr``: Updated left/right swap toggle.
        - ``runtime_note_text``: Optional transient UI message.
        - ``runtime_note_until``: Monotonic timestamp until note stays visible.
        - ``should_exit``: True when runtime loop should terminate.
    """
    key = key_raw & 0xFF
    runtime_note_text: str | None = None
    runtime_note_until = 0.0

    # Toggle left/right visualization and processing channels.
    if key == ord("s"):
        swap_lr = not swap_lr

    preset_map = {
        ord("z"): "near",
        ord("x"): "mid",
        ord("c"): "far",
        ord("v"): "off",
    }
    # ROI tuning preset hotkeys: z/x/c/v -> near/mid/far/off.
    if key in preset_map:
        preset = preset_map[key]
        apply_roi_tune_preset(preset)
        runtime_note_text = f"Preset switched: {preset}"
        runtime_note_until = time.perf_counter() + 2.0

    # Direct numeric source selection (1..9 and keypad variants in multi-mode).
    if multi_mode:
        req_idx = decode_switch_index(key_raw, len(device_list))
        if req_idx is not None and req_idx != active_idx:
            switch_state.pending = request_switch(
                cam=cam,
                from_idx=active_idx,
                to_idx=req_idx,
                key_label="number",
            )

    # Cycle to next source with 'n' in multi-mode.
    if multi_mode and key == ord("n"):
        next_idx = (active_idx + 1) % len(device_list)
        switch_state.pending = request_switch(
            cam=cam,
            from_idx=active_idx,
            to_idx=next_idx,
            key_label="next",
        )

    # Exit on ESC or 'q'.
    should_exit = key == 27 or key == ord("q")
    return swap_lr, runtime_note_text, runtime_note_until, should_exit
