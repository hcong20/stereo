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
- Multi-input pre-opened capture with single active stereo pipeline

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

4-input deployment mode (single active pipeline, runtime switching):

```bash
python3 main.py \
  --devices /dev/video20,/dev/video22,/dev/video24,/dev/video26 \
  --active-input 1 \
  --switch-timeout-ms 500 \
  --capture-mode auto \
  --calib stereo_calib_params.npz \
  --fps 30
```

If your hardware cannot stream all camera nodes in parallel (common on shared USB hubs),
use single-active capture mode:

```bash
python3 main.py \
  --devices /dev/video20,/dev/video22,/dev/video24,/dev/video26 \
  --active-input 2 \
  --switch-timeout-ms 500 \
  --capture-mode single-active \
  --warmup-frames 1 \
  --calib stereo_calib_params.npz \
  --fps 30
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

ROI distance tuning presets (fast field setup):

```bash
# near: stricter quality gate, stronger smoothing
python3 main.py --roi-tune-preset near

# mid: balanced defaults for general scenes
python3 main.py --roi-tune-preset mid

# far: looser gate, faster response for sparse/long-range ROI
python3 main.py --roi-tune-preset far
```

Manual override still works and takes effect after preset selection:

```bash
python3 main.py --roi-tune-preset mid --roi-valid-ratio-min 0.20 --ema-alpha 0.40
```

Default capture backend is OpenCV V4L2. To enable GStreamer backend:

```bash
python3 main.py --gstreamer --device /dev/video0 --calib stereo_calib_params.npz
```

By default, GStreamer capture now prefers RK3588 hardware MJPEG decode (`mppjpegdec`)
and automatically falls back to software decode (`jpegdec`) if hardware decode
is unavailable.

It also prefers NV12 output path first (lower CPU pipeline) and falls back to
BGR output when NV12 negotiation or frame layout is unsupported.

In NV12 mode, the runtime now consumes the Y plane directly for stereo matching
grayscale input to avoid full-frame per-frame BGR conversion in the capture path.

If you want color preview while keeping NV12 grayscale matching path, enable:

```bash
python3 main.py --gstreamer --gst-output nv12 --nv12-preview-bgr
```

Force decode mode explicitly:

```bash
# Force RK3588 hardware decode only
python3 main.py --gst-decode hw

# Force software decode only
python3 main.py --gst-decode sw
```

Force output path explicitly:

```bash
# Prefer NV12 zero-copy-friendly path only
python3 main.py --gst-output nv12

# Force BGR output only (compatibility mode)
python3 main.py --gst-output bgr
```

For custom GStreamer capture pipeline:

```bash
python3 main.py \
  --device /dev/video0 \
  --gstreamer-pipeline "v4l2src device={device} io-mode=2 ! image/jpeg,width={width},height={height},framerate={fps}/1 ! jpegdec ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false" \
  --calib stereo_calib_params.npz
```

To suppress non-fatal OpenCV/GStreamer startup warnings:

```bash
python3 main.py --quiet-opencv-log
```

Preprocessing path is CPU-only:

- Runtime resize + grayscale conversion are handled in Python/OpenCV.
- Startup logs print `preprocess_backend=cpu` and `preprocess_detail=cpu-only preprocess path`.

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

## Multi-Camera Switching (< 500ms)

- Start with `--devices` and pass 4 camera nodes in fixed order.
- All inputs are opened in background capture threads and continuously refreshed.
- Only one stereo processing pipeline runs per frame (active input only).
- Press `1`, `2`, `3`, `4` to switch directly to each input.
- Press `n` to switch to next input cyclically.
- Overlay shows current input and measured switch latency (`Switch latency: ... ms`).
- `--switch-timeout-ms` defaults to `500`, matching the system latency target.
- `--capture-mode auto` prefers parallel capture and auto-falls back to `single-active` when only one input is producing frames.
- `--warmup-frames` reduces camera-open settling time; use `1` for lower switch latency.

## Optional C++ Optimization Path

When Python CPU budget is tight:

- Port capture + rectification + SGBM to C++ OpenCV.
- Enable ARM optimizations in build:

```bash
cmake -D CMAKE_BUILD_TYPE=Release -D ENABLE_NEON=ON -D ENABLE_VFPV3=ON ..
```

- Keep Python for UI/control and run heavy compute as a shared library.
