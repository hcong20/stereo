"""Camera compatibility facade.

This module preserves existing imports while delegating implementation to
smaller focused modules.
"""

from stereo_vision.camera_buffered import BufferedStereoCamera
from stereo_vision.camera_multi import MultiStereoCamera
from stereo_vision.camera_single import CameraConfig, StereoCamera, build_usb_gstreamer_pipeline

__all__ = [
    "CameraConfig",
    "StereoCamera",
    "BufferedStereoCamera",
    "MultiStereoCamera",
    "build_usb_gstreamer_pipeline",
]
