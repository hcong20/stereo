import cv2
import numpy as np
from collections import deque

# =========================
# Const Parameters
# =========================
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480

SINGLE_WIDTH = FRAME_WIDTH // 2

ROI_RATIO = 0.2

ALPHA = 0.8  # EMA smoothing

# Focus lock parameters
HISTORY_SIZE = 20
STABLE_THRESHOLD = 0.02     # stability (2%)
PEAK_THRESHOLD = 0.95       # 95% of max
LR_DIFF_THRESHOLD = 0.15    # left-right consistency

# =========================
# Focus Function
# =========================
def compute_focus(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var() / (gray.mean() + 1e-6)


def get_center_roi(img):
    h, w = img.shape[:2]

    roi_h = int(h * ROI_RATIO)
    roi_w = int(w * ROI_RATIO)

    y1 = h // 2 - roi_h // 2
    y2 = h // 2 + roi_h // 2

    x1 = w // 2 - roi_w // 2
    x2 = w // 2 + roi_w // 2

    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


# =========================
# Main
# =========================
cap = cv2.VideoCapture(26, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

prev_left = 0
prev_right = 0

focus_history = deque(maxlen=HISTORY_SIZE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    left = frame[:, :SINGLE_WIDTH]
    right = frame[:, SINGLE_WIDTH:]

    roi_left, (x1, y1, x2, y2) = get_center_roi(left)
    roi_right, _ = get_center_roi(right)

    # Focus
    left_focus = compute_focus(roi_left)
    right_focus = compute_focus(roi_right)

    # EMA smoothing
    left_focus = ALPHA * prev_left + (1 - ALPHA) * left_focus
    right_focus = ALPHA * prev_right + (1 - ALPHA) * right_focus

    prev_left = left_focus
    prev_right = right_focus

    # Combined focus (average)
    focus_value = (left_focus + right_focus) / 2.0

    # Save history
    focus_history.append(focus_value)

    # =========================
    # Focus Lock Detection
    # =========================
    status = "ADJUSTING"

    if len(focus_history) == HISTORY_SIZE:
        max_focus = max(focus_history)
        min_focus = min(focus_history)

        # Stability: variation is small
        stable = (max_focus - min_focus) / (max_focus + 1e-6) < STABLE_THRESHOLD

        # Near peak
        near_peak = focus_value > PEAK_THRESHOLD * max_focus

        # Left-right consistency
        lr_diff = abs(left_focus - right_focus) / (max(left_focus, right_focus) + 1e-6)
        consistent = lr_diff < LR_DIFF_THRESHOLD

        if stable and near_peak and consistent:
            status = "FOCUS LOCKED"

    # =========================
    # Draw ROI
    # =========================
    cv2.rectangle(left, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.rectangle(right, (x1, y1), (x2, y2), (0,255,0), 2)

    # Display values
    cv2.putText(left, f"Focus: {left_focus:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.putText(right, f"Focus: {right_focus:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Status display
    color = (0,255,0) if status == "FOCUS LOCKED" else (0,255,255)

    cv2.putText(frame, status, (450, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Combine view
    combined = np.hstack((left, right))
    cv2.imshow("Auto Focus Assist", combined)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()