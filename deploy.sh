#!/usr/bin/env bash
set -euo pipefail

# ===== Configuration =====

SERVICE_NAME="stereo"
USER_NAME="linaro"
WORK_DIR="/home/linaro"
LOCAL_BIN="/home/linaro/stereo_app"
PKG_BIN="/usr/bin/stereo-app"
# Headless service mode: keep processing/logging active without GTK window init.
DEFAULT_ARGS="--gstreamer --no-display --log-measurements --crop-height-ratio 1.0"
CAMERA_DEVICE="/dev/video20"
CAMERA_WAIT_RETRIES=10

SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
WRAPPER_SCRIPT="${WORK_DIR}/run_stereo.sh"
LOG_DIR="${WORK_DIR}/logs"

echo "==== Industrial Deployment Setup ===="

mkdir -p "${LOG_DIR}"

# ===== 1. Create wrapper script =====

cat > "${WRAPPER_SCRIPT}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="/home/linaro"
LOG_DIR="${WORK_DIR}/logs"
LOCAL_BIN="/home/linaro/stereo_app"
PKG_BIN="/usr/bin/stereo-app"
# Match default service runtime flags from deploy configuration above.
DEFAULT_ARGS="--gstreamer --no-display --log-measurements --crop-height-ratio 1.0"
CAMERA_DEVICE="/dev/video20"
CAMERA_WAIT_RETRIES=10

mkdir -p "${LOG_DIR}"

echo "[$(date)] Preparing stereo runtime..." | tee -a "${LOG_DIR}/runtime.log"

for i in $(seq 1 "${CAMERA_WAIT_RETRIES}"); do
    if [[ -e "${CAMERA_DEVICE}" ]]; then
        break
    fi
    echo "[$(date)] Waiting for ${CAMERA_DEVICE} (${i}/${CAMERA_WAIT_RETRIES})..." | tee -a "${LOG_DIR}/runtime.log"
    sleep 1
done

if [[ -x "${LOCAL_BIN}" ]]; then
    APP_BIN="${LOCAL_BIN}"
elif [[ -x "${PKG_BIN}" ]]; then
    APP_BIN="${PKG_BIN}"
else
    echo "[$(date)] ERROR: No executable found at ${LOCAL_BIN} or ${PKG_BIN}" | tee -a "${LOG_DIR}/runtime.log"
    exit 1
fi

echo "[$(date)] Starting ${APP_BIN} ${DEFAULT_ARGS}" | tee -a "${LOG_DIR}/runtime.log"
# Mirror app output to both runtime.log and the systemd journal.
"${APP_BIN}" ${DEFAULT_ARGS} 2>&1 | tee -a "${LOG_DIR}/runtime.log"
exit ${PIPESTATUS[0]}
EOF

chmod +x "${WRAPPER_SCRIPT}"

# ===== 2. Create systemd service =====

sudo bash -c "cat > ${SERVICE_FILE}" <<EOF
[Unit]
Description=Stereo Vision System (Industrial)
After=network.target systemd-udev-settle.service

[Service]
Type=simple
User=${USER_NAME}
Group=video
WorkingDirectory=${WORK_DIR}
ExecStartPre=/bin/sleep 2
ExecStart=${WRAPPER_SCRIPT}

# Match interactive shell runtime for codec and display-dependent libraries.
Environment=LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu
Environment=XDG_RUNTIME_DIR=/run/user/1000
Environment=DISPLAY=:0

# Allow direct access to video/render/input nodes used by capture/decode stack.
SupplementaryGroups=video render input
PrivateDevices=no

Restart=always
RestartSec=2
StartLimitInterval=0

MemoryMax=2G
CPUQuota=300%

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# ===== 3. Configure log rotation =====

sudo bash -c "cat > /etc/logrotate.d/${SERVICE_NAME}" <<EOF
${LOG_DIR}/*.log {
daily
rotate 7
compress
missingok
notifempty
copytruncate
}
EOF

# ===== 4. Reload and start service =====

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}.service"
sudo systemctl restart "${SERVICE_NAME}.service"

echo "==== Setup Complete ===="
echo "Check status: systemctl status ${SERVICE_NAME}"
echo "View logs: journalctl -u ${SERVICE_NAME} -f"
