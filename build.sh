#!/usr/bin/env bash
set -euo pipefail

# Build helper for Ubuntu 22.04+ (RK3588 friendly)
# Usage:
#   ./build.sh            # build onefile binary
#   ./build.sh onedir     # build onedir bundle (faster startup)
#   ./build.sh deb        # build a Debian .deb package

MODE="${1:-onefile}"
if [[ "$MODE" != "onefile" && "$MODE" != "onedir" && "$MODE" != "deb" ]]; then
  echo "Invalid mode: $MODE"
  echo "Usage: ./build.sh [onefile|onedir|deb]"
  exit 1
fi

APP_NAME="stereo_app"

if [[ "$MODE" == "deb" ]]; then
  if ! command -v dpkg-buildpackage >/dev/null 2>&1; then
    echo "[ERROR] dpkg-buildpackage is not installed."
    echo "[ERROR] Install dpkg-dev and debhelper first, then rerun ./build.sh deb"
    exit 1
  fi

  dpkg-buildpackage -us -uc -b

  echo "Debian package build complete."
  echo "- Package: ../stereo-app_*_all.deb"
  exit 0
fi

VENV_DIR=".venv"
USE_VENV=1

if [[ ! -d "$VENV_DIR" ]]; then
  if ! python3 -m venv "$VENV_DIR"; then
    echo "[WARN] Could not create virtualenv (python3-venv missing?)."
    echo "[WARN] Falling back to user-level Python environment."
    USE_VENV=0
  fi
fi

if [[ -f "$VENV_DIR/bin/activate" && $USE_VENV -eq 1 ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PYTHON_BIN="python"
  PIP_USER_FLAG=()
else
  if [[ -d "$VENV_DIR" && ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "[WARN] Ignoring incomplete virtualenv in $VENV_DIR"
  fi
  PYTHON_BIN="python3"
  PIP_USER_FLAG=(--user)
fi

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install "${PIP_USER_FLAG[@]}" pyinstaller

PYI_ARGS=(
  --noconfirm
  --clean
  --name "$APP_NAME"
  --collect-all cv2
  --add-data "stereo_calib_params.npz:."
)

if [[ "$MODE" == "onefile" ]]; then
  PYI_ARGS+=(--onefile)
else
  PYI_ARGS+=(--onedir)
fi

"$PYTHON_BIN" -m PyInstaller "${PYI_ARGS[@]}" main.py

echo "Build complete."
echo "- Binary: dist/$APP_NAME"
echo "- Run example: ./dist/$APP_NAME --device /dev/video0 --calib ./stereo_calib_params.npz --fps 30"
