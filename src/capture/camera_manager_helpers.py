"""Helper utilities for multi-source stereo capture orchestration."""

from typing import Dict, List, Optional, Tuple

import numpy as np

FramePacket = Tuple[np.ndarray, np.ndarray, float, int]


def clamp_index(index: int, source_count: int) -> int:
    """Clamp an index to valid source bounds."""
    if source_count <= 0:
        raise ValueError("source_count must be positive")
    return max(0, min(int(index), source_count - 1))


def normalize_group_ids(group_ids: Optional[List[str]], source_count: int) -> List[str]:
    """Validate and normalize group labels for all sources."""
    if group_ids is not None and len(group_ids) != int(source_count):
        raise ValueError("group_ids length must match number of sources")
    if group_ids is None:
        return [str(i) for i in range(source_count)]
    return [str(g) for g in group_ids]


def build_group_maps(group_ids: List[str]) -> tuple[Dict[str, List[int]], Dict[int, int]]:
    """Build group->indices and source->slot maps."""
    group_to_indices: Dict[str, List[int]] = {}
    index_slot: Dict[int, int] = {}
    for src_idx, group in enumerate(group_ids):
        group_to_indices.setdefault(group, []).append(src_idx)
    for _, indices in group_to_indices.items():
        for slot, src_idx in enumerate(indices):
            index_slot[src_idx] = slot
    return group_to_indices, index_slot


def aligned_live_indices(
    group_to_indices: Dict[str, List[int]],
    index_slot: Dict[int, int],
    reference_index: int,
    source_count: int,
) -> Dict[str, int]:
    """Pick one live source per group aligned by intra-group slot."""
    ref_idx = clamp_index(reference_index, source_count)
    ref_slot = index_slot.get(ref_idx, 0)
    out: Dict[str, int] = {}
    for group, indices in group_to_indices.items():
        slot = ref_slot if ref_slot < len(indices) else len(indices) - 1
        out[group] = indices[slot]
    return out


def build_pending_switch_payload(
    request_ts: float,
    from_idx: int,
    to_idx: int,
    stop_ms: float,
    start_call_ms: float,
    target_prev_frame_id: int,
    target_open_count_before: int,
) -> dict:
    """Build pending-switch payload stored until first target frame arrives."""
    return {
        "request_ts": float(request_ts),
        "from_idx": int(from_idx),
        "to_idx": int(to_idx),
        "stop_ms": float(stop_ms),
        "start_call_ms": float(start_call_ms),
        "target_prev_frame_id": int(target_prev_frame_id),
        "target_open_count_before": int(target_open_count_before),
    }


def finalize_switch_breakdown(
    pending: Optional[dict],
    source_idx: int,
    frame_ts: float,
    frame_id: int,
    open_ms: float,
    open_count: int,
    now_ts: float,
) -> tuple[Optional[dict], Optional[dict]]:
    """Finalize pending switch metrics when first valid frame is observed."""
    if pending is None:
        return None, None
    if int(pending["to_idx"]) != int(source_idx):
        return pending, None

    request_ts = float(pending["request_ts"])
    prev_id = int(pending["target_prev_frame_id"])
    if frame_ts < request_ts or int(frame_id) <= prev_id:
        return pending, None

    open_count_before = int(pending["target_open_count_before"])
    open_ms_effective = float(open_ms) if int(open_count) > open_count_before else 0.0
    total_ms = max(0.0, (float(now_ts) - request_ts) * 1000.0)
    stop_ms = float(pending["stop_ms"])
    first_frame_ms = max(0.0, total_ms - stop_ms - open_ms_effective)

    breakdown = {
        "from_idx": int(pending["from_idx"]),
        "to_idx": int(pending["to_idx"]),
        "total_ms": total_ms,
        "stop_ms": stop_ms,
        "open_ms": open_ms_effective,
        "first_frame_ms": first_frame_ms,
        "start_call_ms": float(pending["start_call_ms"]),
    }
    return None, breakdown


def pick_freshest_fallback(
    latest_frames_by_index: List[Optional[FramePacket]],
    now_ts: float,
    max_fallback_age_s: float,
) -> Optional[Tuple[int, FramePacket]]:
    """Pick freshest frame among sources that is not older than max age."""
    freshest: Optional[Tuple[int, FramePacket]] = None
    max_age = max(0.0, float(max_fallback_age_s))
    for fallback_idx, fallback in enumerate(latest_frames_by_index):
        if fallback is None:
            continue
        frame_age = float(now_ts) - float(fallback[2])
        if frame_age > max_age:
            continue
        if freshest is None or float(fallback[2]) > float(freshest[1][2]):
            freshest = (fallback_idx, fallback)
    return freshest