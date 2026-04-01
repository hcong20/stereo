"""Runtime ROI tuning presets and filter lifecycle management."""

from __future__ import annotations

from argparse import Namespace

from stereo_vision.filters import DistanceFilter, TemporalFilterConfig


class RoiTuneController:
    """Keeps ROI tuning state and rebuilds temporal filter on preset changes."""

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.manual_roi_valid_ratio_min = float(args.roi_valid_ratio_min)
        self.manual_roi_p10_weight = float(args.roi_p10_weight)
        self.manual_roi_min_weight = float(args.roi_min_weight)
        self.manual_filter_window = int(args.filter_window)
        self.manual_ema_alpha = float(args.ema_alpha)
        self.manual_max_jump = float(args.max_jump)

        self.roi_tune_preset = str(getattr(args, "roi_tune_preset", "off"))
        self.distance_filter = self._build_filter()

    def _build_filter(self) -> DistanceFilter:
        return DistanceFilter(
            TemporalFilterConfig(
                ema_alpha=float(self.args.ema_alpha),
                max_jump_m=float(self.args.max_jump),
                window=int(self.args.filter_window),
            )
        )

    def apply_preset(self, preset: str) -> None:
        self.roi_tune_preset = preset

        if preset == "off":
            self.args.roi_valid_ratio_min = self.manual_roi_valid_ratio_min
            self.args.roi_p10_weight = self.manual_roi_p10_weight
            self.args.roi_min_weight = self.manual_roi_min_weight
            self.args.filter_window = self.manual_filter_window
            self.args.ema_alpha = self.manual_ema_alpha
            self.args.max_jump = self.manual_max_jump
        elif preset == "near":
            self.args.roi_valid_ratio_min = 0.25
            self.args.roi_p10_weight = 0.80
            self.args.roi_min_weight = 0.05
            self.args.filter_window = 7
            self.args.ema_alpha = 0.45
            self.args.max_jump = 0.60
        elif preset == "mid":
            self.args.roi_valid_ratio_min = 0.15
            self.args.roi_p10_weight = 0.70
            self.args.roi_min_weight = 0.10
            self.args.filter_window = 5
            self.args.ema_alpha = 0.35
            self.args.max_jump = 1.00
        elif preset == "far":
            self.args.roi_valid_ratio_min = 0.08
            self.args.roi_p10_weight = 0.55
            self.args.roi_min_weight = 0.12
            self.args.filter_window = 3
            self.args.ema_alpha = 0.25
            self.args.max_jump = 1.80
        else:
            raise ValueError(f"Unsupported ROI tune preset: {preset}")

        # Rebuild temporal filter so window/jump/EMA changes take effect immediately.
        self.distance_filter = self._build_filter()

        print(
            "[INFO] Applied ROI tuning preset: "
            f"{preset} -> valid_ratio_min={float(self.args.roi_valid_ratio_min):.2f}, "
            f"p10_weight={float(self.args.roi_p10_weight):.2f}, min_weight={float(self.args.roi_min_weight):.2f}, "
            f"window={int(self.args.filter_window)}, ema_alpha={float(self.args.ema_alpha):.2f}, "
            f"max_jump={float(self.args.max_jump):.2f}"
        )
