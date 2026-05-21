"""Camera compatibility facade.

This module preserves existing imports while delegating implementation to
smaller focused modules.
"""

from src.capture.camera_worker_buffered import BufferedCameraWorker
from src.capture.camera_manger import CameraManger
from src.capture.camera_worker import CameraConfig, CameraWorker
from src.capture.gstreamer_pipelines import build_usb_gstreamer_pipeline

__all__ = [
    "CameraConfig",
    "CameraWorker",
    "BufferedCameraWorker",
    "CameraManger",
    "build_usb_gstreamer_pipeline",
]
