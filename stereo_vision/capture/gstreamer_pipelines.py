"""GStreamer pipeline builders for RK3588 stereo capture."""


def _build_usb_gstreamer_sw_pipeline(device: str, width: int, height: int, fps: int) -> str:
    """Build software MJPEG decode pipeline with direct BGR output."""
    return (
        f"v4l2src device={device} io-mode=2 ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        "jpegdec ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def _build_usb_gstreamer_hw_pipeline(device: str, width: int, height: int, fps: int) -> str:
    """Build RK3588 hardware MJPEG decode pipeline to BGR output.

    On tested RK3588 setups, videoconvert is required for stable BGR negotiation.
    """
    return (
        f"v4l2src device={device} io-mode=2 ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        "mppjpegdec ! videoconvert n-threads=2 ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def _build_usb_gstreamer_hw_nv12_pipeline(device: str, width: int, height: int, fps: int) -> str:
    """Build RK3588 hardware MJPEG decode pipeline to NV12 output."""
    return (
        f"v4l2src device={device} io-mode=2 ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        "mppjpegdec ! video/x-raw,format=NV12 ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def _build_usb_gstreamer_sw_nv12_pipeline(device: str, width: int, height: int, fps: int) -> str:
    """Build software MJPEG decode pipeline to NV12 output.

    On tested setups, software decode requires videoconvert for NV12 output.
    """
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
    hw_bgr = _build_usb_gstreamer_hw_pipeline(device, width, height, fps)
    hw_nv12 = _build_usb_gstreamer_hw_nv12_pipeline(device, width, height, fps)
    sw_nv12 = _build_usb_gstreamer_sw_nv12_pipeline(device, width, height, fps)

    if mode == "auto":
        decode_nv12 = [hw_nv12, sw_nv12]
        decode_bgr = [hw_bgr, sw_bgr]
    elif mode == "hw":
        decode_nv12 = [hw_nv12]
        decode_bgr = [hw_bgr]
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