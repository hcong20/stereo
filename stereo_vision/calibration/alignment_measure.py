import cv2
import numpy as np

# =========================
# Const Parameters
# =========================
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480

SINGLE_WIDTH = FRAME_WIDTH // 2

# Thresholds
VERTICAL_ERROR_GOOD = 1.0
VERTICAL_ERROR_OK = 2.5
MIN_MATCH_COUNT = 30

# Horizontal line spacing
LINE_STEP = 50

# =========================
# Feature Detector
# =========================
orb = cv2.ORB_create(1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# =========================
# Draw horizontal lines
# =========================
def draw_horizontal_lines(img, step=50):
    h, w = img.shape[:2]
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (0, 255, 0), 1)
    return img

# =========================
# Main
# =========================
cap = cv2.VideoCapture(20, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    left = frame[:, :SINGLE_WIDTH].copy()
    right = frame[:, SINGLE_WIDTH:].copy()

    gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # Detect features
    kp1, des1 = orb.detectAndCompute(gray_l, None)
    kp2, des2 = orb.detectAndCompute(gray_r, None)

    status = "NO DATA"
    color = (0, 0, 255)
    avg_error = -1
    vertical_errors = []

    if des1 is not None and des2 is not None:
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:100]

        for m in good_matches:
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt

            dy = abs(pt1[1] - pt2[1])
            vertical_errors.append(dy)

        if len(vertical_errors) >= MIN_MATCH_COUNT:
            avg_error = np.mean(vertical_errors)

            if avg_error < VERTICAL_ERROR_GOOD:
                status = "GOOD"
                color = (0, 255, 0)
            elif avg_error < VERTICAL_ERROR_OK:
                status = "OK"
                color = (0, 255, 255)
            else:
                status = "BAD"
                color = (0, 0, 255)
        else:
            status = "NOT ENOUGH FEATURES"

    # =========================
    # Draw horizontal guide lines
    # =========================
    left = draw_horizontal_lines(left, LINE_STEP)
    right = draw_horizontal_lines(right, LINE_STEP)

    # =========================
    # Draw match lines (important!)
    # =========================
    for i, m in enumerate(matches[:20] if des1 is not None and des2 is not None else []):
        pt1 = tuple(map(int, kp1[m.queryIdx].pt))
        pt2 = tuple(map(int, kp2[m.trainIdx].pt))

        # Shift right image x for combined display
        pt2_shifted = (int(pt2[0] + SINGLE_WIDTH), int(pt2[1]))

        cv2.line(frame, pt1, pt2_shifted, (255, 0, 0), 1)

    # =========================
    # Combine images
    # =========================
    combined = np.hstack((left, right))

    # =========================
    # Display info
    # =========================
    text = f"Vertical Error: {avg_error:.2f}px | Matches: {len(vertical_errors)} | {status}"
    cv2.putText(combined, text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Stereo Alignment Check + Lines", combined)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()