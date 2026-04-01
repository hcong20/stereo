"""Camera compatibility facade.

This module preserves existing imports while delegating implementation to
smaller focused modules.
"""

from stereo_vision.camera_worker_buffered import BufferedCameraWorker
from stereo_vision.camera_manger import CameraManger
from stereo_vision.camera_worker import CameraConfig, CameraWorker
from stereo_vision.gstreamer_pipelines import build_usb_gstreamer_pipeline

__all__ = [
    "CameraConfig",
    "CameraWorker",
    "BufferedCameraWorker",
    "CameraManger",
    "build_usb_gstreamer_pipeline",
]
