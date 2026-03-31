"""Per-stage runtime profiling helpers for the main loop."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StageProfiler:
    """Accumulate and periodically report per-stage latencies in milliseconds."""

    enabled: bool = False
    interval: int = 60
    _acc: dict[str, float] = field(
        default_factory=lambda: {
            "capture": 0.0,
            "rectify": 0.0,
            "preprocess": 0.0,
            "disparity": 0.0,
            "depth": 0.0,
            "viz": 0.0,
            "total": 0.0,
        }
    )
    _count: int = 0

    def record(
        self,
        t0: float,
        t_capture: float,
        t_rectify: float,
        t_preprocess: float,
        t_disparity: float,
        t_depth: float,
        t_viz: float,
    ) -> Optional[str]:
        """Record one frame and return a profile line when report interval is reached."""
        if not self.enabled:
            return None

        self._acc["capture"] += (t_capture - t0) * 1000.0
        self._acc["rectify"] += (t_rectify - t_capture) * 1000.0
        self._acc["preprocess"] += (t_preprocess - t_rectify) * 1000.0
        self._acc["disparity"] += (t_disparity - t_preprocess) * 1000.0
        self._acc["depth"] += (t_depth - t_disparity) * 1000.0
        self._acc["viz"] += (t_viz - t_depth) * 1000.0
        self._acc["total"] += (t_viz - t0) * 1000.0
        self._count += 1

        interval = max(1, int(self.interval))
        if self._count < interval:
            return None

        c = float(self._count)
        line = (
            "[PROFILE] avg_ms "
            f"capture={self._acc['capture'] / c:.2f} "
            f"rectify={self._acc['rectify'] / c:.2f} "
            f"preprocess={self._acc['preprocess'] / c:.2f} "
            f"disparity={self._acc['disparity'] / c:.2f} "
            f"depth={self._acc['depth'] / c:.2f} "
            f"viz={self._acc['viz'] / c:.2f} "
            f"total={self._acc['total'] / c:.2f}"
        )
        for key in self._acc:
            self._acc[key] = 0.0
        self._count = 0
        return line