"""Temporal filtering helpers for stable distance readouts.

Combines a short-window median with EMA smoothing and jump rejection.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TemporalFilterConfig:
    """Configuration for median+EMA smoothing of distance estimates."""

    ema_alpha: float = 0.35
    max_jump_m: float = 1.5
    window: int = 5


class DistanceFilter:
    """Suppress spikes and smooth ROI distance over time.

    Median protects against single-frame outliers; EMA provides continuity.
    """

    def __init__(self, cfg: TemporalFilterConfig):
        """Initialize history buffer and EMA state."""
        self.cfg = cfg
        self.history = deque(maxlen=max(1, cfg.window))
        self.ema: Optional[float] = None

    def update(self, measurement_m: Optional[float]) -> Optional[float]:
        """Update filter with a measurement and return smoothed value.

        Args:
            measurement_m: New distance measurement in meters.

        Returns:
            Filtered distance in meters, or last stable value when rejected.
        """
        if measurement_m is None or not np.isfinite(measurement_m):
            return self.ema

        # Reject abrupt changes relative to current trend.
        if self.ema is not None and abs(measurement_m - self.ema) > self.cfg.max_jump_m:
            return self.ema

        # Median is computed over recent history to reduce transient spikes.
        self.history.append(float(measurement_m))
        median_value = float(np.median(np.array(self.history, dtype=np.float32)))

        if self.ema is None:
            self.ema = median_value
        else:
            a = self.cfg.ema_alpha
            self.ema = a * median_value + (1.0 - a) * self.ema

        return self.ema
