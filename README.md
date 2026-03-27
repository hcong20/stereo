# RK3588 Stereo Vision Distance System (Python, OpenCV)

Production-oriented stereo distance measurement pipeline for RK3588 (Ubuntu 22.04, aarch64) using CPU-only OpenCV.

## Features

- Dual-view capture from a single combined stereo USB stream (left + right)
- Stereo calibration loading from `stereo_calib_params.npz`
- Rectification map precomputation and fast remap
- StereoSGBM disparity computation with low-latency defaults
- Disparity to metric depth conversion
- ROI-based robust distance estimation
- Temporal stabilization (median + EMA + outlier rejection)
- Real-time visualization, latency/FPS monitor, click-to-measure
- Optional GStreamer capture path and ROI-only disparity mode

## Project Structure

- `main.py`: Integrated real-time pipeline
- `stereo_vision/camera.py`: V4L2/GStreamer capture module
- `stereo_vision/calibration.py`: Calibration loader
- `stereo_vision/rectification.py`: Rectification map builder and remap
- `stereo_vision/disparity.py`: StereoSGBM wrapper
- `stereo_vision/depth.py`: Depth conversion and validity handling
- `stereo_vision/roi.py`: ROI model and robust depth statistics
- `stereo_vision/filters.py`: Temporal smoothing filters
- `stereo_vision/visualization.py`: Display overlays and click callbacks
- `stereo_vision/optimization.py`: Runtime optimization helpers

## Dependencies (APT only)

Install OpenCV from Ubuntu repos:

```bash
sudo apt update
sudo apt install -y python3-opencv python3-numpy v4l-utils gstreamer1.0-tools \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

## Run

```bash
python3 main.py --device /dev/video0 --calib stereo_calib_params.npz --fps 30
```

## Build Executable (Ubuntu 22.04)

Use the included build helper:

```bash
./build.sh
```

This creates a standalone executable at:

```bash
dist/stereo_app
```

Run it with:

```bash
./dist/stereo_app --device /dev/video0 --calib ./stereo_calib_params.npz --fps 30
```

Optional: build a folder bundle with faster startup:

```bash
./build.sh onedir
```

Useful flags:

```bash
python3 main.py \
  --device /dev/video0 \
  --calib stereo_calib_params.npz \
  --fps 30 \
  --roi 220,140,200,140 \
  --num-disp 128 \
  --block-size 5 \
  --scale 1.0
```

For low-latency GStreamer capture:

```bash
python3 main.py --gstreamer --device /dev/video0 --calib stereo_calib_params.npz
```

RGA-ready preprocessing backend selection:

```bash
# Auto mode: try RGA module first, fallback to CPU
python3 main.py --preprocess-backend auto --rga-module rga_helper

# Force CPU preprocessing
python3 main.py --preprocess-backend cpu

# Force RGA (fails fast if module is not installed)
python3 main.py --preprocess-backend rga --rga-module rga_helper
```

RGA backend module contract (`rga_helper` by default):

- Must export `preprocess_pair_bgr_to_gray(left_bgr, right_bgr, scale)`
- Must return `(left_resized_bgr, right_resized_bgr, left_gray, right_gray)`
- Optional: export `is_available()` returning `True` only when hardware backend is usable

Included module:

- `rga_helper.py` is provided as an adapter skeleton.
- In `auto` mode, it reports unavailable unless a compatible RGA Python binding exists.
- To enable real hardware acceleration, implement your binding adapter in `rga_helper.py` function `_rga_preprocess_impl(...)`.

Troubleshooting (still using CPU):

- Startup log now prints `preprocess_detail=...` with fallback reason.
- If you see fallback to CPU, check that your selected module exports:
  - `is_available()` returning `True`
  - `preprocess_pair_bgr_to_gray(...)`
- On RK3588, having `/dev/rga` and `librga` alone is not enough for this Python path; you still need a Python binding or a custom C++/pybind adapter.

Build native `rockchip_rga` backend on this repo:

```bash
cd native_rga
./build.sh
cd ..
python3 main.py --preprocess-backend rga --rga-module rga_helper
```

If startup prints `preprocess_backend=rga`, hardware preprocess path is active.

## SGBM Tuning Guide (RK3588)

- Start with `--num-disp 96` or `--num-disp 128`.
- Use `--block-size 5` for balance; increase to 7 for noisy scenes.
- If CPU is high, try:
  - Smaller ROI (`--roi ...`) with `--roi-disparity-only`
  - Lower scale (`--scale 0.75` or `0.5`)
  - Lower camera FPS
- Keep camera exposure fixed to reduce disparity flicker.

## Performance Tips for < 50ms

- Keep capture queue short (`CAP_PROP_BUFFERSIZE=1` and appsink `drop=true`).
- Precompute rectification maps once at startup.
- Use grayscale for matcher input.
- Prefer ROI-only disparity when your measurement target is fixed.
- Avoid full-frame post-processing; only process ROI for statistics.
- If available, offload resize/colorspace to RGA in a C++ helper.

## RGA Integration Suggestion

Python OpenCV does not expose RGA directly. For best RK3588 performance:

- Build a small C++ helper with `librga` (`im2d`) to perform:
  - YUYV/NV12 -> GRAY/BGR conversion
  - Resize to matcher resolution
- Share frame memory using DMA-BUF where possible.
- Call helper from Python via a lightweight binding (`ctypes` or `pybind11`) only if needed.

## Multi-Camera Switching (< 500ms)

- Pre-open all candidate camera devices in background threads.
- Keep one active processing pipeline and hot-swap frame source pointer.
- Maintain independent frame queues per camera (`maxlen=1`).
- Warm up each camera stream ahead of switching.
- Trigger switch on next frame boundary to keep transition deterministic.

## Optional C++ Optimization Path

When Python CPU budget is tight:

- Port capture + rectification + SGBM to C++ OpenCV.
- Enable ARM optimizations in build:

```bash
cmake -D CMAKE_BUILD_TYPE=Release -D ENABLE_NEON=ON -D ENABLE_VFPV3=ON ..
```

- Keep Python for UI/control and run heavy compute as a shared library.
