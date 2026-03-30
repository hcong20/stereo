"""Single stereo camera capture helpers.

The expected raw frame layout is [left | right] in a single image.
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class CameraConfig:
    """Runtime camera settings for OpenCV V4L2/GStreamer capture.

    Attributes:
        device: Video device node, e.g. /dev/video0.
        width: Combined frame width (left + right cameras).
        height: Frame height.
        fps: Target capture FPS.
        use_gstreamer: Use custom GStreamer pipeline when True.
        gstreamer_pipeline: Optional explicit pipeline string.
        gstreamer_decode: GStreamer decode mode (auto/hw/sw) when pipeline is not explicit.
        gstreamer_output: Preferred GStreamer output path (auto/nv12/bgr).
        warmup_frames: Frames to discard after open.
    """

    device: str = "/dev/video0"
    width: int = 1280
    height: int = 480
    fps: int = 30
    use_gstreamer: bool = False
    gstreamer_pipeline: Optional[str] = None
    gstreamer_decode: str = "auto"
    gstreamer_output: str = "auto"
    warmup_frames: int = 1
    fast_reopen: bool = True


def _build_usb_gstreamer_sw_pipeline(device: str, width: int, height: int, fps: int) -> str:
    """Build software MJPEG decode pipeline for broad compatibility."""
    return (
        f"v4l2src device={device} io-mode=2 ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        "jpegdec ! videoconvert n-threads=2 ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def _build_usb_gstreamer_hw_pipeline(device: str, width: int, height: int, fps: int) -> str:
    """Build RK3588 hardware MJPEG decode pipeline via Rockchip MPP (no jpegparse)."""
    return (
        f"v4l2src device={device} io-mode=2 ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        "mppjpegdec ! videoconvert n-threads=2 ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def _build_usb_gstreamer_hw_nv12_pipeline(device: str, width: int, height: int, fps: int) -> str:
    """Build RK3588 hardware MJPEG decode pipeline to NV12 output (no jpegparse)."""
    return (
        f"v4l2src device={device} io-mode=2 ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        "mppjpegdec ! videoconvert n-threads=2 ! video/x-raw,format=NV12 ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def _build_usb_gstreamer_sw_nv12_pipeline(device: str, width: int, height: int, fps: int) -> str:
    """Build software MJPEG decode pipeline to NV12 output."""
    return (
        f"v4l2src device={device} io-mode=2 ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        "jpegdec ! videoconvert n-threads=2 ! video/x-raw,format=NV12 ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def build_usb_gstreamer_pipeline_candidates(
    device: str,
    width: int,
    height: int,
    fps: int,
    decode_mode: str = "auto",
    output_mode: str = "auto",
) -> list[str]:
    """Build candidate pipelines for RK3588 USB camera capture.

    decode_mode:
        auto: prefer Rockchip hardware decode, fallback to software decode.
        hw:   force Rockchip hardware decode candidates only.
        sw:   force software decode path.
    output_mode:
        auto: prefer NV12 path first, fallback to BGR.
        nv12: force NV12 outputs only.
        bgr:  force BGR outputs only.
    """
    mode = str(decode_mode).strip().lower()
    out_mode = str(output_mode).strip().lower()

    sw_bgr = _build_usb_gstreamer_sw_pipeline(device, width, height, fps)
    hw_bgr_primary = _build_usb_gstreamer_hw_pipeline(device, width, height, fps)
    hw_nv12_primary = _build_usb_gstreamer_hw_nv12_pipeline(device, width, height, fps)
    sw_nv12 = _build_usb_gstreamer_sw_nv12_pipeline(device, width, height, fps)


    if mode == "auto":
        decode_nv12 = [hw_nv12_primary, sw_nv12]
        decode_bgr = [hw_bgr_primary, sw_bgr]
    elif mode == "hw":
        decode_nv12 = [hw_nv12_primary]
        decode_bgr = [hw_bgr_primary]
    elif mode == "sw":
        decode_nv12 = [sw_nv12]
        decode_bgr = [sw_bgr]
    else:
        raise ValueError("gstreamer_decode must be one of: auto, hw, sw")

    if out_mode == "auto":
        return decode_nv12 + decode_bgr
    if out_mode == "nv12":
        return decode_nv12
    if out_mode == "bgr":
        return decode_bgr
    raise ValueError("gstreamer_output must be one of: auto, nv12, bgr")


def build_usb_gstreamer_pipeline(device: str, width: int, height: int, fps: int) -> str:
    """Backward-compatible single pipeline builder.

    Returns the preferred auto-mode candidate.
    """
    return build_usb_gstreamer_pipeline_candidates(
        device,
        width,
        height,
        fps,
        decode_mode="auto",
        output_mode="auto",
    )[0]


class StereoCamera:
    """Capture combined stereo frame and split into left/right images.

    This wrapper hides backend differences (V4L2 vs GStreamer) and always
    returns synchronized left/right images from a single captured frame.
    """

    def __init__(self, cfg: CameraConfig):
        """Initialize camera wrapper with immutable capture configuration."""
        self.cfg = cfg
        self.cap: Optional[cv2.VideoCapture] = None
        self.last_ts: float = 0.0
        self._configured_once = False
        self._gst_selected_pipeline: Optional[str] = None

    @staticmethod
    def _short_pipeline_text(text: str, max_len: int = 180) -> str:
        """Return a compact one-line pipeline snippet for startup logs."""
        compact = " ".join(str(text).split())
        if len(compact) <= max_len:
            return compact
        return compact[: max_len - 3] + "..."

    def _convert_gstreamer_frame_if_needed(self, frame: np.ndarray) -> np.ndarray:
        """Convert NV12 appsink frames to BGR when OpenCV does not auto-convert."""
        if not self.cfg.use_gstreamer:
            return frame
        if not self._gst_selected_pipeline or "format=NV12" not in self._gst_selected_pipeline:
            return frame
        # Some OpenCV builds already return BGR even if appsink requested NV12.
        if frame.ndim == 3 and frame.shape[2] == 3:
            return frame

        in_frame = frame
        if in_frame.ndim == 3 and in_frame.shape[2] == 1:
            in_frame = in_frame[:, :, 0]

        if in_frame.ndim != 2:
            raise RuntimeError(
                "Unexpected NV12 frame layout from GStreamer appsink: "
                f"shape={frame.shape}. Try --gst-output bgr"
            )

        h2, _ = in_frame.shape[:2]
        if h2 % 3 != 0:
            raise RuntimeError(
                "Unexpected NV12 frame height from GStreamer appsink: "
                f"shape={frame.shape}. Try --gst-output bgr"
            )

        return cv2.cvtColor(in_frame, cv2.COLOR_YUV2BGR_NV12)

    def _split_nv12_stereo_to_gray(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split side-by-side NV12 frame into left/right grayscale images.

        For NV12, the first H rows are the Y plane, which is already grayscale.
        Using Y directly avoids per-frame full BGR conversion.
        """
        in_frame = frame
        if in_frame.ndim == 3 and in_frame.shape[2] == 1:
            in_frame = in_frame[:, :, 0]

        # Some OpenCV/GStreamer combinations may still return BGR.
        if in_frame.ndim == 3 and in_frame.shape[2] == 3:
            bgr_h, bgr_w = in_frame.shape[:2]
            if bgr_w % 2 != 0:
                raise ValueError(f"Expected combined frame width to be even, got {bgr_w}")
            half = bgr_w // 2
            left_gray = cv2.cvtColor(in_frame[:, :half], cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(in_frame[:, half:], cv2.COLOR_BGR2GRAY)
            return left_gray, right_gray

        if in_frame.ndim != 2:
            raise RuntimeError(
                "Unexpected NV12 frame layout from GStreamer appsink: "
                f"shape={frame.shape}. Try --gst-output bgr"
            )

        h3_2, w = in_frame.shape[:2]
        if h3_2 % 3 != 0:
            raise RuntimeError(
                "Unexpected NV12 frame height from GStreamer appsink: "
                f"shape={frame.shape}. Try --gst-output bgr"
            )
        if w % 2 != 0:
            raise ValueError(f"Expected combined frame width to be even, got {w}")

        h = (h3_2 * 2) // 3
        y_plane = in_frame[:h, :]
        half = w // 2
        left_gray = y_plane[:, :half]
        right_gray = y_plane[:, half:]
        return left_gray, right_gray

    def _open_gstreamer_single_pipeline(self) -> None:
        """Open one GStreamer pipeline (single appsink) with candidate fallback."""
        if self.cfg.gstreamer_pipeline:
            pipeline_candidates = [self.cfg.gstreamer_pipeline]
        else:
            pipeline_candidates = build_usb_gstreamer_pipeline_candidates(
                self.cfg.device,
                self.cfg.width,
                self.cfg.height,
                self.cfg.fps,
                decode_mode=self.cfg.gstreamer_decode,
                output_mode=self.cfg.gstreamer_output,
            )

        self.cap = None
        selected_idx = -1
        for idx, pipeline in enumerate(pipeline_candidates):
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap is not None and cap.isOpened():
                self.cap = cap
                selected_idx = idx
                self._gst_selected_pipeline = pipeline
                break
            if cap is not None:
                cap.release()

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(
                "Failed to open stereo camera via GStreamer. "
                f"device={self.cfg.device}, decode={self.cfg.gstreamer_decode}, "
                f"output={self.cfg.gstreamer_output}, "
                f"attempted_pipelines={pipeline_candidates}"
            )

        if (
            not self.cfg.gstreamer_pipeline
            and str(self.cfg.gstreamer_decode).strip().lower() == "auto"
            and selected_idx > 0
        ):
            print(
                "[WARN] Preferred RK3588 hardware decode pipeline was unavailable; "
                f"fallback candidate index={selected_idx + 1}/{len(pipeline_candidates)}"
            )
        selected = self._gst_selected_pipeline or ""
        selected_out = "nv12" if "format=NV12" in selected else "bgr"
        print(
            "[INFO] GStreamer pipeline selected: "
            f"decode={self.cfg.gstreamer_decode}, output={selected_out}, "
            f"candidate={selected_idx + 1}/{len(pipeline_candidates)}"
        )
        print(f"[INFO] gst_pipeline={self._short_pipeline_text(selected)}")

    def open(self) -> None:
        """Open camera stream and warm it up before first use.

        Raises:
            RuntimeError: If camera cannot be opened.
        """
        was_configured = self._configured_once
        if self.cfg.use_gstreamer:
            self._open_gstreamer_single_pipeline()
        else:
            self.cap = cv2.VideoCapture(self.cfg.device, cv2.CAP_V4L2)

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Failed to open stereo camera")

        if not self.cfg.use_gstreamer:
            # Always enforce combined stereo dimensions; otherwise some drivers
            # reopen at half width and produce visibly half-frame output.
            force_full_config = (not bool(self.cfg.fast_reopen)) or (not was_configured)
            if force_full_config:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            else:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            expected_min_w = max(2, int(self.cfg.width * 0.75))
            expected_min_h = max(2, int(self.cfg.height * 0.75))
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            if actual_w > 0 and actual_h > 0 and (actual_w < expected_min_w or actual_h < expected_min_h):
                # Retry one strict reconfiguration before failing the open.
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                time.sleep(0.01)
                actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            if actual_w > 0 and actual_h > 0 and (actual_w < expected_min_w or actual_h < expected_min_h):
                raise RuntimeError(
                    "Unexpected frame size after open "
                    f"({actual_w}x{actual_h}); expected around {self.cfg.width}x{self.cfg.height}"
                )

        self._configured_once = True

        # Drop initial frames so exposure/auto-controls stabilize.
        warmup = max(0, int(self.cfg.warmup_frames))
        if bool(self.cfg.fast_reopen) and was_configured:
            # On reopen, skipping warmup reduces switch delay significantly.
            warmup = 0
        for _ in range(warmup):
            self.cap.read()

    def read(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Read one combined frame, split into left/right images, and timestamp it.

        Returns:
            (left_bgr, right_bgr, capture_timestamp_seconds).
        """
        if self.cap is None:
            raise RuntimeError("Camera is not open")

        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read camera frame")

        if self.cfg.use_gstreamer and self._gst_selected_pipeline and "format=NV12" in self._gst_selected_pipeline:
            left, right = self._split_nv12_stereo_to_gray(frame)
            w = left.shape[1] + right.shape[1]
        else:
            frame = self._convert_gstreamer_frame_if_needed(frame)
            # Height is kept for readability and potential future validation hooks.
            h, w = frame.shape[:2]
            if w % 2 != 0:
                raise ValueError(f"Expected combined frame width to be even, got {w}")

            # Input frame is expected to be side-by-side: left half + right half.
            half = w // 2
            left = frame[:, :half]
            right = frame[:, half:]

        min_expected_w = max(2, int(self.cfg.width * 0.75))
        if w < min_expected_w:
            raise RuntimeError(
                "Captured frame width is smaller than expected stereo width: "
                f"got {w}, expected around {self.cfg.width}"
            )

        self.last_ts = time.perf_counter()
        return left, right, self.last_ts

    def release(self) -> None:
        """Release camera resources if a stream is currently open."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None