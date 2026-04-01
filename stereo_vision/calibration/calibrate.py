import cv2
import numpy as np
import os
import sys
from typing import List, Tuple, Dict

# -------------------------------
# Parameters
# -------------------------------
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480

checkerboard_size = (10, 6)      # Internal corners (width, height)
square_size = 17.8               # mm, size of one square
calib_file = 'stereo_calib_params.npz'
save_dir = "calib_images"
os.makedirs(save_dir, exist_ok=True)

# Depth engineering parameters (record only, not used directly here)
min_range = 200   # mm
max_range = 4000  # mm
baseline = 60     # mm (converted from meters)

# Object points template (Z=0 plane, units: mm)
objp = np.zeros((checkerboard_size[1]*checkerboard_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size


# -------------------------------
# Helper functions
# -------------------------------
def draw_capture_guides(
    image: np.ndarray,
    checkerboard_found: bool,
    saved_count: int,
    target_index: int,
    show_grid: bool = True
) -> np.ndarray:
    """Draw visual guides to help checkerboard placement during capture."""
    vis = image.copy()
    h, w = vis.shape[:2]

    # 25 target positions (5x5) from top-left to bottom-right.
    cols, rows = 5, 5
    tx_idx = int(np.clip(target_index, 0, cols * rows - 1))
    tx_col = tx_idx % cols
    tx_row = tx_idx // cols
    x_positions = np.linspace(0.18, 0.82, cols)
    y_positions = np.linspace(0.18, 0.82, rows)
    target_x = int(w * x_positions[tx_col])
    target_y = int(h * y_positions[tx_row])

    # Target rectangle centered at current target position.
    rect_w = int(w * 0.32)
    rect_h = int(h * 0.34)
    x1 = int(np.clip(target_x - rect_w // 2, 0, max(0, w - rect_w - 1)))
    y1 = int(np.clip(target_y - rect_h // 2, 0, max(0, h - rect_h - 1)))
    x2 = min(w - 1, x1 + rect_w)
    y2 = min(h - 1, y1 + rect_h)

    guide_color = (0, 220, 0) if checkerboard_found else (0, 180, 255)
    cv2.rectangle(vis, (x1, y1), (x2, y2), guide_color, 2)
    cv2.circle(vis, (target_x, target_y), 6, guide_color, -1)

    # Rule-of-thirds style grid helps move the board to diverse positions.
    if show_grid:
        grid_color = (90, 90, 90)
        for i in range(1, 6):
            gx = (w * i) // 6
            gy = (h * i) // 6
            cv2.line(vis, (gx, 0), (gx, h), grid_color, 1)
            cv2.line(vis, (0, gy), (w, gy), grid_color, 1)

    # Center crosshair remains as global reference.
    cx, cy = w // 2, h // 2
    cv2.line(vis, (cx - 16, cy), (cx + 16, cy), (170, 170, 170), 1)
    cv2.line(vis, (cx, cy - 16), (cx, cy + 16), (170, 170, 170), 1)

    status = "Board: DETECTED" if checkerboard_found else "Board: NOT DETECTED"
    cv2.putText(vis, status, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, guide_color, 2, cv2.LINE_AA)
    cv2.putText(vis, f"Saved: {saved_count}/25", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Target: {tx_idx + 1}/25", (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(vis, "Place board inside current box and press 's'", (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
    return vis


def compute_reprojection_errors(
    objpoints: List[np.ndarray], 
    imgpoints: List[np.ndarray], 
    rvecs: List[np.ndarray], 
    tvecs: List[np.ndarray], 
    K: np.ndarray, 
    dist: np.ndarray
) -> Tuple[List[float], float]:
    """Compute per-frame reprojection errors and overall mean (pixels)."""
    per_view_errors = []
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
        per_view_errors.append(float(err))
    mean_error = float(np.mean(per_view_errors)) if per_view_errors else 0.0
    return per_view_errors, mean_error


def compute_rectified_row_error(
    imgpoints_l: List[np.ndarray], 
    imgpoints_r: List[np.ndarray], 
    K_l: np.ndarray, dist_l: np.ndarray, 
    K_r: np.ndarray, dist_r: np.ndarray, 
    R1: np.ndarray, P1: np.ndarray, 
    R2: np.ndarray, P2: np.ndarray
) -> Tuple[float, float]:
    """Compute vertical alignment error after rectification |y_left - y_right| (pixels)."""
    row_errors = []
    for pts_l, pts_r in zip(imgpoints_l, imgpoints_r):
        rect_l = cv2.undistortPoints(pts_l, K_l, dist_l, R=R1, P=P1)
        rect_r = cv2.undistortPoints(pts_r, K_r, dist_r, R=R2, P=P2)
        dy = np.abs(rect_l[:, 0, 1] - rect_r[:, 0, 1])
        row_errors.extend(dy.tolist())
    if not row_errors:
        return 0.0, 0.0
    row_errors = np.asarray(row_errors, dtype=np.float32)
    return float(np.mean(row_errors)), float(np.max(row_errors))


def print_calibration_report(
    ret_l: float, ret_r: float, ret_stereo: float,
    objpoints: List[np.ndarray], imgpoints_l: List[np.ndarray], imgpoints_r: List[np.ndarray],
    rvecs_l: List[np.ndarray], tvecs_l: List[np.ndarray],
    rvecs_r: List[np.ndarray], tvecs_r: List[np.ndarray],
    K_l: np.ndarray, dist_l: np.ndarray,
    K_r: np.ndarray, dist_r: np.ndarray,
    T: np.ndarray,
    R1: np.ndarray, P1: np.ndarray, R2: np.ndarray, P2: np.ndarray,
    verbose: bool = True
):
    """Print calibration quality report in English."""
    per_l, mean_l = compute_reprojection_errors(objpoints, imgpoints_l, rvecs_l, tvecs_l, K_l, dist_l)
    per_r, mean_r = compute_reprojection_errors(objpoints, imgpoints_r, rvecs_r, tvecs_r, K_r, dist_r)
    row_mean, row_max = compute_rectified_row_error(imgpoints_l, imgpoints_r, K_l, dist_l, K_r, dist_r, R1, P1, R2, P2)

    fx_l, fy_l, cx_l, cy_l = K_l[0,0], K_l[1,1], K_l[0,2], K_l[1,2]
    fx_r, fy_r, cx_r, cy_r = K_r[0,0], K_r[1,1], K_r[0,2], K_r[1,2]
    baseline_mm = float(np.linalg.norm(T))  # mm

    dist_l_flat = np.asarray(dist_l).reshape(-1)
    dist_r_flat = np.asarray(dist_r).reshape(-1)
    dist_l_head = ", ".join(f"{v:.5f}" for v in dist_l_flat[:5])
    dist_r_head = ", ".join(f"{v:.5f}" for v in dist_r_flat[:5])

    if verbose:
        print("\n========== Stereo Calibration Report ==========")
        print("[Main Parameters]")
        print(f"Left Camera Intrinsics: fx={fx_l:.2f}, fy={fy_l:.2f}, cx={cx_l:.2f}, cy={cy_l:.2f}")
        print(f"Right Camera Intrinsics: fx={fx_r:.2f}, fy={fy_r:.2f}, cx={cx_r:.2f}, cy={cy_r:.2f}")
        print(f"Left Camera Distortion (first 5): [{dist_l_head}]")
        print(f"Right Camera Distortion (first 5): [{dist_r_head}]")
        print(f"Stereo Baseline: {baseline_mm:.2f} mm")
        print("\n[Quality Metrics]")
        print(f"RMS Left: {ret_l:.4f} px")
        print(f"RMS Right: {ret_r:.4f} px")
        print(f"RMS Stereo: {ret_stereo:.4f} px")
        print(f"Mean Reprojection Left: {mean_l:.4f} px")
        print(f"Mean Reprojection Right: {mean_r:.4f} px")
        print(f"Rectified Row Error Mean: {row_mean:.4f} px, Max: {row_max:.4f} px")

        if ret_stereo < 0.5:
            print("Conclusion: Excellent stereo calibration (RMS < 0.5 px).")
        elif ret_stereo < 1.0:
            print("Conclusion: Acceptable stereo calibration (0.5 <= RMS < 1.0 px).")
        elif ret_stereo < 1.5:
            print("Conclusion: Moderate calibration (1.0 <= RMS < 1.5 px), check bad frames.")
        else:
            print("Conclusion: High RMS (>1.5 px), consider recalibration.")

        if row_mean < 1.0:
            print("Conclusion: Rectified images well aligned (mean row error < 1 px).")
        else:
            print("Conclusion: Rectified images alignment moderate, check corner quality.")
        print("Per-frame reprojection errors (px):")
        for i, (e_l, e_r) in enumerate(zip(per_l, per_r)):
            print(f"  Frame {i:02d}: left={e_l:.4f}, right={e_r:.4f}")
        print("===============================================\n")


# -------------------------------
# Online calibration
# -------------------------------
def calibrate_stereo(cap: cv2.VideoCapture, verbose: bool = True) -> Dict:
    """Perform online stereo calibration from camera capture."""
    print("Starting stereo calibration. Move the checkerboard to cover the view.")
    print("Press 's' to save the current frame, ESC to finish.")
    print("Guides: moving target box enabled (25 positions).")

    objpoints, imgpoints_l, imgpoints_r = [], [], []
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
    saved_count = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Dynamically split left/right halves
        h, w = frame.shape[:2]
        half_w = w // 2
        left_img = frame[:, :half_w]
        right_img = frame[:, half_w:]
        gray_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        small_l = cv2.resize(gray_l, (0,0), fx=0.5, fy=0.5)
        ret_l, corners_l = cv2.findChessboardCorners(small_l, checkerboard_size)
        if ret_l:
            corners_l = corners_l * 2  # scale back to original size
        
        small_r = cv2.resize(gray_r, (0,0), fx=0.5, fy=0.5)
        ret_r, corners_r = cv2.findChessboardCorners(small_r, checkerboard_size)
        if ret_r:
            corners_r = corners_r * 2  # scale back to original size

        if verbose:
            target_index = min(saved_count, 24)
            vis_l = draw_capture_guides(left_img, bool(ret_l), saved_count, target_index, show_grid=True)
            vis_r = draw_capture_guides(right_img, bool(ret_r), saved_count, target_index, show_grid=True)
            if ret_l: cv2.drawChessboardCorners(vis_l, checkerboard_size, corners_l, ret_l)
            if ret_r: cv2.drawChessboardCorners(vis_r, checkerboard_size, corners_r, ret_r)
            cv2.imshow("Left Camera - Calibration", vis_l)
            cv2.imshow("Right Camera - Calibration", vis_r)
            print(f"Frame {frame_count}: ret_l={ret_l}, ret_r={ret_r}, saved_count={saved_count}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if ret_l and ret_r:
                corners_l2 = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), criteria)
                corners_r2 = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), criteria)
                objpoints.append(objp)
                imgpoints_l.append(corners_l2)
                imgpoints_r.append(corners_r2)

                img_name = os.path.join(save_dir, f"calib_{saved_count:02d}.png")
                cv2.imwrite(img_name, frame)
                if verbose:
                    print(f"Saved calibration image {img_name}")
                saved_count += 1
            else:
                if verbose:
                    print("Chessboard not detected in both images. Frame not saved.")

        if key == 27 or saved_count >= 25:
            break

    cv2.destroyAllWindows()
    if saved_count == 0:
        raise RuntimeError("No valid calibration frames detected. Adjust checkerboard or camera.")

    # Single camera calibration
    ret_l, K_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, gray_l.shape[::-1], None, None)
    ret_r, K_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, gray_r.shape[::-1], None, None)

    # Stereo calibration with fixed intrinsics
    flags = cv2.CALIB_FIX_INTRINSIC
    ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        K_l, dist_l, K_r, dist_r,
        gray_l.shape[::-1], criteria=criteria, flags=flags
    )

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_l, dist_l, K_r, dist_r, gray_l.shape[::-1], R, T, alpha=0
    )

    print_calibration_report(
        ret_l, ret_r, ret_stereo,
        objpoints, imgpoints_l, imgpoints_r,
        rvecs_l, tvecs_l, rvecs_r, tvecs_r,
        K_l, dist_l, K_r, dist_r, T,
        R1, P1, R2, P2,
        verbose
    )

    params = {'K_l':K_l, 'dist_l':dist_l, 'K_r':K_r, 'dist_r':dist_r,
              'R':R, 'T':T, 'R1':R1, 'R2':R2, 'P1':P1, 'P2':P2, 'Q':Q}
    np.savez(calib_file, **params)
    if verbose:
        print("Calibration complete and saved.")
    return params


# -------------------------------
# Main entry
# -------------------------------
def main():
    verbose = True  # toggle this to False for silent mode
    if "--recalibrate-images" in sys.argv:
        calibrate_stereo_from_saved_images(save_dir)
        return

    cap = cv2.VideoCapture(20, cv2.CAP_V4L2)  # Adjust camera index as needed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    try:
        calibrate_stereo(cap, verbose=verbose)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()