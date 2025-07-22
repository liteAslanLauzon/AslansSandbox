import cv2
import numpy as np
from typing import Optional, Tuple, List


def load_and_crop(path: str, roi: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        print(f"Error: could not load image at '{path}'")
        return None
    x_min, x_max, y_min, y_max = roi
    return img[y_min:y_max, x_min:x_max]


def get_diff_mask(
    before: np.ndarray, after: np.ndarray, thresh: int = 25, speck_area: int = 50
) -> np.ndarray:
    diff = cv2.absdiff(before, after)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    clean = np.zeros_like(mask)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= speck_area:
            clean[labels == lbl] = 255
    return clean


def detect_circles_hough(
    mask: np.ndarray,
    dp: float = 1.2,
    minDist: float = 200,
    param1: float = 50,
    param2: float = 30,
    minRadius: int = 100,
    maxRadius: int = 1000,
) -> List[Tuple[int, int, int]]:
    edges = cv2.Canny(mask, 50, 150)
    blurred = cv2.GaussianBlur(edges, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius,
    )
    if circles is not None:
        return [tuple(c.astype(int)) for c in circles[0]]
    return []


def extrapolate_missing_arcs(
    mask: np.ndarray, circles: List[Tuple[int, int, int]], angle_step: int = 1
) -> List[Tuple[int, int, int, List[Tuple[float, float]]]]:
    h, w = mask.shape
    results = []
    for x, y, r in circles:
        theta = np.deg2rad(np.arange(0, 360, angle_step))
        xs = (x + r * np.cos(theta)).astype(int)
        ys = (y + r * np.sin(theta)).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        present = np.zeros_like(theta, dtype=bool)
        present[valid] = mask[ys[valid], xs[valid]] > 0

        missing = ~present
        pad = np.concatenate(([0], missing.view(np.uint8), [0]))
        diff = np.diff(pad)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        segments: List[Tuple[float, float]] = []
        for s, e in zip(starts, ends):
            start_ang = np.rad2deg(theta[s])
            end_ang = np.rad2deg(theta[e - 1])
            segments.append((start_ang, end_ang))
        results.append((x, y, r, segments))
    return results


if __name__ == "__main__":
    # ---- Parameters ----
    path_before = "Projects/SacDropV3/Images/before.bmp"
    path_after = "Projects/SacDropV3/Images/after.bmp"
    roi = (1930, 4730, 1486, 2184)
    diff_thresh = 25
    speck_area = 50

    dp, minDist, param1, param2 = 1.2, 200, 50, 30
    minR, maxR = 100, 1000

    before = load_and_crop(path_before, roi)
    after = load_and_crop(path_after, roi)
    if before is None or after is None or before.shape != after.shape:
        print("Error loading images or mismatched sizes.")
        exit(1)

    mask = get_diff_mask(before, after, thresh=diff_thresh, speck_area=speck_area)
    circles = detect_circles_hough(mask, dp, minDist, param1, param2, minR, maxR)
    arcs = extrapolate_missing_arcs(mask, circles)

    # ---- Prepare output ----
    output = after.copy()
    overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    output = cv2.addWeighted(output, 0.8, overlay, 0.2, 0)

    for x, y, r in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (0, 255, 0), -1)

    for x, y, r, segments in arcs:
        for start_ang, end_ang in segments:
            cv2.ellipse(
                output,
                (x, y),
                (r, r),
                angle=0,
                startAngle=start_ang,
                endAngle=end_ang,
                color=(255, 0, 0),
                thickness=2,
            )

    # compute desired window sizes
    mask_h, mask_w = mask.shape
    out_h, out_w = output.shape[:2]
    mask_win_h, circ_win_h = 600, 900
    mask_win_w = int(mask_w * (mask_win_h / mask_h))
    circ_win_w = int(out_w * (circ_win_h / out_h))

    # 1) Create windows in NORMAL mode
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Circles + Extrapolated Arcs", cv2.WINDOW_NORMAL)

    # 2) Clear any fullscreen flags (must do *before* imshow)
    cv2.setWindowProperty("Mask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "Circles + Extrapolated Arcs", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
    )

    # 3) Pre‐resize to your target dimensions
    cv2.resizeWindow("Mask", mask_win_w, mask_win_h)
    cv2.resizeWindow("Circles + Extrapolated Arcs", circ_win_w, circ_win_h)

    # 4) Now show your images
    cv2.imshow("Mask", mask)
    cv2.imshow("Circles + Extrapolated Arcs", output)

    # 5) Wait for a key and clean up
    cv2.waitKey(0)
    cv2.destroyAllWindows()
