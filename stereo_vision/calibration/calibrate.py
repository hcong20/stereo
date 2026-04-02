import cv2
import numpy as np
import os
import sys
from typing import List, Tuple, Dict, Any

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
def build_capture_plan() -> List[Dict[str, Any]]:
    """Create a 25-step capture plan.

    Args:
        None.

    Returns:
        A list of 25 capture targets covering near, mid, and far distances.
    """
    plan: List[Dict[str, Any]] = []

    # These five locations define the frontal coverage: four corners plus center.
    # Corner/center points are reserved for fronto-parallel captures.
    fronto_points = [
        ("top-left", 0.20, 0.20),
        ("top-right", 0.80, 0.20),
        ("bottom-left", 0.20, 0.80),
        ("bottom-right", 0.80, 0.80),
        ("center", 0.50, 0.50),
    ]

    # Tilt captures use the cardinal directions so the on-screen shape can match the pose.
    tilt_points = [
        ("top", 0.50, 0.20, "down"),
        ("bottom", 0.50, 0.80, "up"),
        ("right", 0.80, 0.50, "left"),
        ("left", 0.20, 0.50, "right"),
        ("center", 0.50, 0.50, "roll"),
    ]

    def add_group(distance: str, include_tilt: bool) -> None:
        size = {"near": "large", "mid": "medium", "far": "small"}[distance]
        for target_name, x_ratio, y_ratio in fronto_points:
            plan.append(
                {
                    "distance": distance,
                    "size": size,
                    "orientation": "fronto-parallel",
                    "target": target_name,
                    "x_ratio": x_ratio,
                    "y_ratio": y_ratio,
                    "tilt_direction": None,
                }
            )

        if include_tilt:
            for target_name, x_ratio, y_ratio, tilt_direction in tilt_points:
                plan.append(
                    {
                        "distance": distance,
                        "size": size,
                        "orientation": "tilt",
                        "target": target_name,
                        "x_ratio": x_ratio,
                        "y_ratio": y_ratio,
                        "tilt_direction": tilt_direction,
                    }
                )

    add_group("near", include_tilt=True)
    add_group("mid", include_tilt=True)
    add_group("far", include_tilt=False)
    return plan


def checklist_progress(saved_count: int, capture_plan: List[Dict[str, Any]]) -> Dict[str, bool]:
    """Summarize which calibration checklist categories are covered.

    Args:
        saved_count: Number of already saved calibration frames.
        capture_plan: Full ordered capture plan.

    Returns:
        Dictionary of checklist flags describing coverage progress.
    """
    done_items = capture_plan[:saved_count]

    # Track coverage by distance, position, and pose so the overlay can show progress.
    sizes = {it["size"] for it in done_items}
    orientations = {it["orientation"] for it in done_items}
    target_counts: Dict[str, int] = {
        "top-left": 0,
        "top-right": 0,
        "bottom-left": 0,
        "bottom-right": 0,
        "center": 0,
        "top": 0,
        "bottom": 0,
        "left": 0,
        "right": 0,
    }
    for item in done_items:
        target_counts[item["target"]] = target_counts.get(item["target"], 0) + 1

    near_count = sum(1 for it in done_items if it["distance"] == "near")
    mid_count = sum(1 for it in done_items if it["distance"] == "mid")
    far_count = sum(1 for it in done_items if it["distance"] == "far")
    near_fronto = sum(1 for it in done_items if it["distance"] == "near" and it["orientation"] == "fronto-parallel")
    mid_fronto = sum(1 for it in done_items if it["distance"] == "mid" and it["orientation"] == "fronto-parallel")
    far_fronto = sum(1 for it in done_items if it["distance"] == "far" and it["orientation"] == "fronto-parallel")
    near_tilt = sum(1 for it in done_items if it["distance"] == "near" and it["orientation"] == "tilt")
    mid_tilt = sum(1 for it in done_items if it["distance"] == "mid" and it["orientation"] == "tilt")

    return {
        "distance_ratio": near_count >= 10 and mid_count >= 10 and far_count >= 5,
        "fov_coverage": target_counts["top-left"] >= 1 and target_counts["top-right"] >= 1 and target_counts["bottom-left"] >= 1 and target_counts["bottom-right"] >= 1 and target_counts["center"] >= 3 and target_counts["top"] >= 2 and target_counts["bottom"] >= 2 and target_counts["left"] >= 2 and target_counts["right"] >= 2,
        "orientation_coverage": {"fronto-parallel", "tilt"}.issubset(orientations),
        "scale_coverage": {"large", "small"}.issubset(sizes),
        "near_mid_far_plan": near_fronto >= 5 and near_tilt >= 5 and mid_fronto >= 5 and mid_tilt >= 5 and far_fronto >= 5,
    }


def draw_capture_guides(
    image: np.ndarray,
    checkerboard_found: bool,
    saved_count: int,
    target_index: int,
    target_plan: Dict[str, Any],
    progress: Dict[str, bool],
    sync_ok: bool,
    show_grid: bool = True
) -> np.ndarray:
    """Draw visual guides to help checkerboard placement during capture.

    Args:
        image: Source image to annotate.
        checkerboard_found: Whether the checkerboard is currently detected.
        saved_count: Number of saved calibration frames.
        target_index: Index of the current capture target.
        target_plan: Metadata for the current capture target.
        progress: Checklist progress flags.
        sync_ok: Whether both stereo views currently detect the board.
        show_grid: Whether to draw the rule-of-thirds helper grid.

    Returns:
        Annotated visualization image.
    """
    vis = image.copy()
    h, w = vis.shape[:2]

    tx_idx = int(np.clip(target_index, 0, 24))
    target_x = int(w * float(target_plan["x_ratio"]))
    target_y = int(h * float(target_plan["y_ratio"]))

    # Box size follows guidance text (near/large -> bigger, far/small -> smaller).
    size_scale = {"large": 1.40, "medium": 1.00, "small": 0.60}
    scale = size_scale.get(target_plan["size"], 1.0)
    box_ratio = float(checkerboard_size[0]) / float(checkerboard_size[1])
    rect_h = int(h * 0.3 * scale)
    rect_w = int(round(rect_h * box_ratio))
    rect_w = max(1, min(rect_w, w))
    rect_h = max(1, min(rect_h, h))
    x1 = int(np.clip(target_x - rect_w // 2, 0, max(0, w - rect_w - 1)))
    y1 = int(np.clip(target_y - rect_h // 2, 0, max(0, h - rect_h - 1)))
    cx = x1 + rect_w // 2
    cy = y1 + rect_h // 2

    guide_color = (0, 220, 0) if checkerboard_found else (0, 180, 255)

    orientation = target_plan["orientation"]
    if orientation == "fronto-parallel":
        # Frontal targets stay as rectangles.
        box = np.array(
            [
                [x1, y1],
                [x1 + rect_w, y1],
                [x1 + rect_w, y1 + rect_h],
                [x1, y1 + rect_h],
            ],
            dtype=np.int32,
        )
    else:
            # Directional tilt shapes: each one encodes the requested slant in the outline.
        shrink_x = int(rect_w * 0.26)
        shrink_y = int(rect_h * 0.26)
        tilt_direction = target_plan.get("tilt_direction")
        if tilt_direction == "down":
            # top wide, bottom narrow
            box = np.array(
                [
                    [x1, y1],
                    [x1 + rect_w, y1],
                    [x1 + rect_w - shrink_x, y1 + rect_h],
                    [x1 + shrink_x, y1 + rect_h],
                ],
                dtype=np.int32,
            )
        elif tilt_direction == "up":
            # bottom wide, top narrow
            box = np.array(
                [
                    [x1 + shrink_x, y1],
                    [x1 + rect_w - shrink_x, y1],
                    [x1 + rect_w, y1 + rect_h],
                    [x1, y1 + rect_h],
                ],
                dtype=np.int32,
            )
        elif tilt_direction == "left":
            # right wide, left narrow, with the wide side reaching the right edge.
            box = np.array(
                [
                    [x1 + shrink_x, y1 + shrink_y],
                    [x1 + rect_w, y1],
                    [x1 + rect_w, y1 + rect_h],
                    [x1 + shrink_x, y1 + rect_h - shrink_y],
                ],
                dtype=np.int32,
            )
        elif tilt_direction == "right":
            # left wide, right narrow, with the wide side reaching the left edge.
            box = np.array(
                [
                    [x1, y1],
                    [x1 + rect_w - shrink_x, y1 + shrink_y],
                    [x1 + rect_w - shrink_x, y1 + rect_h - shrink_y],
                    [x1, y1 + rect_h],
                ],
                dtype=np.int32,
            )
        else:  # roll
            # Roll guidance uses a parallelogram (not trapezoid).
            slant = int(rect_w * 0.18)
            box = np.array(
                [
                    [x1 + slant, y1],
                    [x1 + rect_w, y1],
                    [x1 + rect_w - slant, y1 + rect_h],
                    [x1, y1 + rect_h],
                ],
                dtype=np.int32,
            )

    cv2.polylines(vis, [box], isClosed=True, color=guide_color, thickness=2)
    cv2.circle(vis, (cx, cy), 6, guide_color, -1)

    tilt_direction = target_plan.get("tilt_direction")
    if tilt_direction == "down":
        # Arrow shows the board should lean downward.
        cv2.arrowedLine(vis, (cx, cy - rect_h // 4), (cx, cy + rect_h // 4), guide_color, 2, tipLength=0.2)
    elif tilt_direction == "up":
        # Arrow shows the board should lean upward.
        cv2.arrowedLine(vis, (cx, cy + rect_h // 4), (cx, cy - rect_h // 4), guide_color, 2, tipLength=0.2)
    elif tilt_direction == "left":
        # Arrow shows the board should lean left.
        cv2.arrowedLine(vis, (cx + rect_w // 4, cy), (cx - rect_w // 4, cy), guide_color, 2, tipLength=0.2)
    elif tilt_direction == "right":
        # Arrow shows the board should lean right.
        cv2.arrowedLine(vis, (cx - rect_w // 4, cy), (cx + rect_w // 4, cy), guide_color, 2, tipLength=0.2)
    elif tilt_direction == "roll":
        # Rotation icon inside the box.
        icon_r = max(8, min(rect_w, rect_h) // 7)
        cv2.ellipse(vis, (cx, cy), (icon_r, icon_r), 0, 40, 330, guide_color, 2)
        end_ang = np.deg2rad(330.0)
        ex = cx + int(icon_r * np.cos(end_ang))
        ey = cy + int(icon_r * np.sin(end_ang))
        cv2.arrowedLine(vis, (ex - 8, ey + 3), (ex, ey), guide_color, 2, tipLength=0.55)

    # Rule-of-thirds style grid helps move the board to diverse positions.
    if show_grid:
        grid_color = (90, 90, 90)
        for i in range(1, 6):
            gx = (w * i) // 6
            gy = (h * i) // 6
            cv2.line(vis, (gx, 0), (gx, h), grid_color, 1)
            cv2.line(vis, (0, gy), (w, gy), grid_color, 1)

    # Center crosshair remains as global reference.
    center_x, center_y = w // 2, h // 2
    cv2.line(vis, (center_x - 16, center_y), (center_x + 16, center_y), (170, 170, 170), 1)
    cv2.line(vis, (center_x, center_y - 16), (center_x, center_y + 16), (170, 170, 170), 1)

    status = "Board: DETECTED" if checkerboard_found else "Board: NOT DETECTED"
    distance_range = {
        "near": "0.2-0.4m",
        "mid": "0.5-1.5m",
        "far": "2.0-4.0m",
    }.get(target_plan["distance"], "-")
    cv2.putText(vis, status, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, guide_color, 2, cv2.LINE_AA)
    cv2.putText(vis, f"Saved: {saved_count}/25", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Target: {tx_idx + 1}/25", (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(
        vis,
        f"Now: {target_plan['distance']}({distance_range}), {target_plan['size']}, {target_plan['orientation']}"
        + (f"/{tilt_direction}" if tilt_direction else ""),
        (12, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    sync_text = "Sync: OK" if sync_ok else "Sync: NG"
    sync_color = (0, 220, 0) if sync_ok else (0, 0, 255)
    cv2.putText(vis, sync_text, (w - 120, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sync_color, 2, cv2.LINE_AA)

    p1 = "Y" if progress["distance_ratio"] else "N"
    p2 = "Y" if progress["fov_coverage"] else "N"
    p3 = "Y" if progress["orientation_coverage"] else "N"
    p4 = "Y" if progress["scale_coverage"] else "N"
    cv2.putText(vis, f"Checklist Depth:{p1} FOV:{p2}", (12, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Pose:{p3} Scale:{p4} Clear+Sync: manual/{'Y' if sync_ok else 'N'}", (12, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(vis, "Place board inside current box and press 's'", (12, h - 60),
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
    """Compute per-frame reprojection errors and overall mean in pixels.

    Args:
        objpoints: Calibrated 3D object points for each frame.
        imgpoints: Detected 2D image points for each frame.
        rvecs: Per-frame rotation vectors from calibration.
        tvecs: Per-frame translation vectors from calibration.
        K: Camera intrinsic matrix.
        dist: Lens distortion coefficients.

    Returns:
        Tuple of per-frame errors and their mean error.
    """
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
    """Compute vertical alignment error after rectification.

    Args:
        imgpoints_l: Left image points per frame.
        imgpoints_r: Right image points per frame.
        K_l: Left camera intrinsic matrix.
        dist_l: Left camera distortion coefficients.
        K_r: Right camera intrinsic matrix.
        dist_r: Right camera distortion coefficients.
        R1: Left rectification transform.
        P1: Left projection matrix after rectification.
        R2: Right rectification transform.
        P2: Right projection matrix after rectification.

    Returns:
        Mean and maximum absolute vertical row error in pixels.
    """
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
    """Print a calibration quality report.

    Args:
        ret_l: Left camera RMS reprojection error from calibration.
        ret_r: Right camera RMS reprojection error from calibration.
        ret_stereo: Stereo calibration RMS error.
        objpoints: 3D object points used for calibration.
        imgpoints_l: Left detected image points.
        imgpoints_r: Right detected image points.
        rvecs_l: Left camera rotation vectors.
        tvecs_l: Left camera translation vectors.
        rvecs_r: Right camera rotation vectors.
        tvecs_r: Right camera translation vectors.
        K_l: Left camera intrinsic matrix.
        dist_l: Left camera distortion coefficients.
        K_r: Right camera intrinsic matrix.
        dist_r: Right camera distortion coefficients.
        T: Stereo baseline translation vector.
        R1: Left rectification rotation.
        P1: Left rectification projection matrix.
        R2: Right rectification rotation.
        P2: Right rectification projection matrix.
        verbose: Whether to print the detailed report.

    Returns:
        None.
    """
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
    """Perform online stereo calibration from camera capture.

    Args:
        cap: OpenCV video capture device.
        verbose: Whether to print progress and diagnostics.

    Returns:
        Dictionary of stereo calibration parameters.
    """
    print("Starting stereo calibration. Move the checkerboard to cover the view.")
    print("Press 's' to save the current frame, ESC to finish.")
    print("Guides: 25-step target plan enabled (depth/FOV/orientation/scale).")
    print("Depth targets: near=0.2-0.4m (10), mid=0.5-1.5m (10), far=2.0-4.0m (5).")
    print("FOV targets: top-left/top-right/bottom-left/bottom-right/center.")
    print("Orientation targets: fronto-parallel, yaw, pitch, roll.")

    objpoints, imgpoints_l, imgpoints_r = [], [], []
    capture_plan = build_capture_plan()
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
            target_plan = capture_plan[target_index]
            progress = checklist_progress(saved_count, capture_plan)
            sync_ok = bool(ret_l and ret_r)
            vis_l = draw_capture_guides(left_img, bool(ret_l), saved_count, target_index, target_plan, progress, sync_ok, show_grid=True)
            vis_r = draw_capture_guides(right_img, bool(ret_r), saved_count, target_index, target_plan, progress, sync_ok, show_grid=True)
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
                    lap_l = cv2.Laplacian(gray_l, cv2.CV_64F).var()
                    lap_r = cv2.Laplacian(gray_r, cv2.CV_64F).var()
                    clarity = "OK" if (lap_l > 80 and lap_r > 80) else "SOFT"
                    step = capture_plan[min(saved_count, 24)]
                    print(f"Saved calibration image {img_name}")
                    print(
                        f"Checklist step {saved_count + 1}/25 -> "
                        f"distance={step['distance']}, size={step['size']}, orientation={step['orientation']}, "
                        f"clarity={clarity} (L={lap_l:.1f}, R={lap_r:.1f}), sync=OK"
                    )
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


def calibrate_stereo_from_saved_images(image_dir: str, verbose: bool = True) -> Dict:
    """Recalibrate stereo cameras from previously saved side-by-side images.

    Args:
        image_dir: Directory containing saved calibration images.
        verbose: Whether to print progress and diagnostics.

    Returns:
        Dictionary of stereo calibration parameters.
    """
    print(f"Starting stereo recalibration from saved images in: {image_dir}")

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Calibration image directory not found: {image_dir}")

    image_files = sorted(
        [
            os.path.join(image_dir, name)
            for name in os.listdir(image_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
    )
    if not image_files:
        raise RuntimeError(f"No calibration images found in: {image_dir}")

    objpoints, imgpoints_l, imgpoints_r = [], [], []
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
    target_size = None
    used_files: List[str] = []

    for image_path in image_files:
        frame = cv2.imread(image_path)
        if frame is None:
            if verbose:
                print(f"Skipping unreadable image: {image_path}")
            continue

        h, w = frame.shape[:2]
        if w < 2:
            if verbose:
                print(f"Skipping too-small image: {image_path}")
            continue

        half_w = w // 2
        left_img = frame[:, :half_w]
        right_img = frame[:, half_w:]
        gray_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        small_l = cv2.resize(gray_l, (0, 0), fx=0.5, fy=0.5)
        ret_l, corners_l = cv2.findChessboardCorners(small_l, checkerboard_size)
        if ret_l:
            corners_l = corners_l * 2

        small_r = cv2.resize(gray_r, (0, 0), fx=0.5, fy=0.5)
        ret_r, corners_r = cv2.findChessboardCorners(small_r, checkerboard_size)
        if ret_r:
            corners_r = corners_r * 2

        if verbose:
            print(f"{os.path.basename(image_path)}: ret_l={ret_l}, ret_r={ret_r}")

        if ret_l and ret_r:
            corners_l2 = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            corners_r2 = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints_l.append(corners_l2)
            imgpoints_r.append(corners_r2)
            target_size = gray_l.shape[::-1]
            used_files.append(image_path)

    if not used_files:
        raise RuntimeError(
            f"No valid stereo calibration pairs found in {image_dir}. "
            "Make sure the saved images contain a detectable checkerboard in both halves."
        )

    if len(used_files) < 5 and verbose:
        print(f"Warning: only {len(used_files)} valid calibration pairs were found.")

    if target_size is None:
        raise RuntimeError("Could not determine calibration image size from saved images.")

    # Single camera calibration from saved observations.
    ret_l, K_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        objpoints, imgpoints_l, target_size, None, None
    )
    ret_r, K_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        objpoints, imgpoints_r, target_size, None, None
    )

    # Stereo calibration with fixed intrinsics.
    flags = cv2.CALIB_FIX_INTRINSIC
    ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_l,
        imgpoints_r,
        K_l,
        dist_l,
        K_r,
        dist_r,
        target_size,
        criteria=criteria,
        flags=flags,
    )

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_l, dist_l, K_r, dist_r, target_size, R, T, alpha=0
    )

    print_calibration_report(
        ret_l,
        ret_r,
        ret_stereo,
        objpoints,
        imgpoints_l,
        imgpoints_r,
        rvecs_l,
        tvecs_l,
        rvecs_r,
        tvecs_r,
        K_l,
        dist_l,
        K_r,
        dist_r,
        T,
        R1,
        P1,
        R2,
        P2,
        verbose,
    )

    params = {
        "K_l": K_l,
        "dist_l": dist_l,
        "K_r": K_r,
        "dist_r": dist_r,
        "R": R,
        "T": T,
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
    }
    np.savez(calib_file, **params)
    if verbose:
        print(f"Recalibration complete and saved to {calib_file}. Used {len(used_files)} image pairs.")
    return params


# -------------------------------
# Main entry
# -------------------------------
def main():
    """Run the calibration entry point.

    Args:
        None.

    Returns:
        None.
    """
    verbose = True  # toggle this to False for silent mode
    if "--images" in sys.argv:
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