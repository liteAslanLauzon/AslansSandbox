import cv2
import numpy as np
import math


def compute_right_edge_angle(
    img_path,
    roi=None,
    threshold_val=50,
    open_close_kernel=(5, 5),
    hough_params=None,
    display=False,
):
    """
    Load an image (grayscale or color), isolate the right edge within an ROI,
    and compute its angle relative to the Y-axis in degrees.

    Args:
        img_path (str): Path to the image file.
        roi (tuple, optional): (y1, y2, x1, x2) crop coordinates. If None, uses full image.
        threshold_val (int, optional): Threshold value for binary inversion.
        open_close_kernel (tuple, optional): Kernel size for morphological ops.
        hough_params (dict, optional): Parameters for HoughLinesP:
            {'rho':1, 'theta':np.pi/180, 'threshold':80,
             'minLineLength':50, 'maxLineGap':10}
        display (bool, optional): If True, show intermediate images.

    Returns:
        float: Angle in degrees relative to the Y-axis.
    """
    # 1. Load image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image at '{img_path}'")

    # 2. Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 3. Determine crop region
    h0, w0 = gray.shape
    if roi:
        y1, y2, x1, x2 = roi
        # clamp to image bounds
        y1c = max(0, min(y1, h0))
        y2c = max(0, min(y2, h0))
        x1c = max(0, min(x1, w0))
        x2c = max(0, min(x2, w0))
        if y1c >= y2c or x1c >= x2c:
            raise ValueError(
                f"Empty crop: image is {w0}×{h0}, "
                f"requested ROI rows {y1}:{y2}, cols {x1}:{x2}"
            )
        crop = gray[y1c:y2c, x1c:x2c]
    else:
        crop = gray

    h, w = crop.shape

    # 4. Threshold
    _, thresh = cv2.threshold(crop, threshold_val, 255, cv2.THRESH_BINARY_INV)
    if thresh.size == 0:
        raise RuntimeError("Thresholding resulted in empty mask")

    # 5. Morphological open then close
    kernel = np.ones(open_close_kernel, np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    # 6. Edge detection + Hough
    edges = cv2.Canny(clean, 50, 150, apertureSize=3)
    params = hough_params or {}
    lines = cv2.HoughLinesP(
        edges,
        rho=params.get("rho", 1),
        theta=params.get("theta", np.pi / 180),
        threshold=params.get("threshold", 80),
        minLineLength=params.get("minLineLength", 50),
        maxLineGap=params.get("maxLineGap", 10),
    )

    # 7. Try Hough-based right-edge
    segment = None
    if lines is not None:
        for margin in (50, 100, 200):
            x_thresh = w - margin
            candidates = [
                tuple(l[0]) for l in lines if l[0][0] > x_thresh or l[0][2] > x_thresh
            ]
            if candidates:
                segment = max(
                    candidates, key=lambda s: math.hypot(s[2] - s[0], s[3] - s[1])
                )
                break

    # 8. Fallback: fitLine on all bright points near right border
    if segment is None:
        x_border = w - 50
        pts = np.column_stack(np.where(clean > 0))  # [[y,x],...]
        pts = pts[pts[:, 1] > x_border]
        if len(pts) < 2:
            raise RuntimeError("Still no line data on the right edge.")
        xy = np.fliplr(pts)
        vx, vy, x0, y0 = cv2.fitLine(xy, cv2.DIST_L2, 0, 0.01, 0.01)
        t_top = (0 - y0) / vy
        x_top = x0 + vx * t_top
        t_bot = (h - y0) / vy
        x_bot = x0 + vx * t_bot
        segment = (int(x_top), 0, int(x_bot), h)

    x1_s, y1_s, x2_s, y2_s = segment

    # 9. Angle relative to Y-axis
    dx, dy = x2_s - x1_s, y2_s - y1_s
    angle_rad = math.atan2(dx, dy)
    angle_deg = abs(math.degrees(angle_rad))

    # 10. Optional display
    if display:
        disp_w = 800
        disp_h = int(disp_w * h / w)
        clean_disp = cv2.resize(clean, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        sx, sy = disp_w / w, disp_h / h
        p1 = (int(x1_s * sx), int(y1_s * sy))
        p2 = (int(x2_s * sx), int(y2_s * sy))
        vis = cv2.cvtColor(clean_disp, cv2.COLOR_GRAY2BGR)
        cv2.line(vis, p1, p2, (0, 255, 0), 2)
        cv2.imshow("Right-Edge Line", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return angle_deg


if __name__ == "__main__":

    compute_right_edge_angle("Projects\\lcosEdgeFinder\\photo\\cropped.jpeg", roi=None, threshold_val=50, open_close_kernel=(5, 5), hough_params=None, display=True)
