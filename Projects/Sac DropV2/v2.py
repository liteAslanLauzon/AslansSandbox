import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from time import time
from typing import Dict, Tuple, Any


def load_and_crop(path: str, roi: Tuple[int, int, int, int] = None) -> np.ndarray:
    path_norm = os.path.normpath(path)
    img = cv2.imread(path_norm, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Unable to load image: {path_norm}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if roi:
        x, y, w, h = roi
        img = img[y : y + h, x : x + w]
    return img


def compute_diff_mask(
    before: np.ndarray,
    after: np.ndarray,
    blur_ksize: Tuple[int, int] = (17, 17),
    sigma: float = 3.0,
    thresh: int = 20,
    close_ksize: Tuple[int, int] = (25, 25),
) -> np.ndarray:
    """
    Compute absolute difference, Gaussian blur, binary threshold, and close holes to fill circle interiors.
    """
    # Grayscale diff
    gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_after, gray_before)
    # Blur to reduce noise
    blur = cv2.GaussianBlur(diff, blur_ksize, sigma)
    # Threshold to binary mask
    _, mask = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)
    # Open to remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    # Close to fill circle interiors
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_ksize)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    return mask


def cluster_binary_findcontours(binary_image: np.ndarray) -> Dict[int, np.ndarray]:
    cnts, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clusters: Dict[int, np.ndarray] = {}
    for i, cnt in enumerate(cnts, start=1):
        mask = np.zeros_like(binary_image)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        coords = np.column_stack(np.where(mask > 0))
        clusters[i] = coords
    return clusters


def cluster_binary_dbscan(
    binary_image: np.ndarray,
    eps: float = 5.0,
    min_samples: int = 5,
    downsample_factor: float = 0.5,
) -> Dict[int, np.ndarray]:
    small = cv2.resize(
        binary_image,
        (0, 0),
        fx=downsample_factor,
        fy=downsample_factor,
        interpolation=cv2.INTER_NEAREST,
    )
    coords = np.column_stack(np.where(small > 0))
    if coords.size == 0:
        return {}
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
    valid = labels != -1
    coords = coords[valid] * (1.0 / downsample_factor)
    labels = labels[valid]
    clusters: Dict[int, np.ndarray] = {}
    for lbl in np.unique(labels):
        clusters[int(lbl)] = coords[labels == lbl].astype(int)
    return clusters


def fit_min_enclosing_circles(
    clusters: Dict[int, np.ndarray],
) -> Dict[int, Tuple[int, int, int]]:
    results: Dict[int, Tuple[int, int, int]] = {}
    for cid, pts in clusters.items():
        if pts.size == 0:
            continue
        pts_xy = np.fliplr(pts).astype(np.int32)
        (x, y), r = cv2.minEnclosingCircle(pts_xy)
        results[cid] = (int(x), int(y), int(r))
    return results


def filter_aligned_circles(
    circles: Dict[int, Tuple[int, int, int]],
    expected_num: int,
    roi_height: int,
    tol_factor: float = 0.05,
) -> Dict[int, Tuple[int, int, int]]:
    if not circles:
        return {}
    ys = np.array([y for (_, y, _) in circles.values()])
    median_y = np.median(ys)
    y_tol = tol_factor * roi_height
    filtered = {
        cid: cyr for cid, cyr in circles.items() if abs(cyr[1] - median_y) <= y_tol
    }
    if len(filtered) > expected_num:
        sel = sorted(filtered.items(), key=lambda item: abs(item[1][1] - median_y))[
            :expected_num
        ]
        filtered = dict(sel)
    return filtered


def annotate_and_save(
    image: np.ndarray, circles: Dict[int, Tuple[int, int, int]], fname: str
) -> None:
    out = image.copy()
    for cid, (x, y, r) in circles.items():
        cv2.circle(out, (x, y), r, (0, 255, 0), 2)
        cv2.putText(out, str(cid), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(fname, out)
    print(f"Saved {fname}, {len(circles)} circles drawn")


def run_shotgun_clustering(
    before_path: str,
    after_path: str,
    expected_num: int,
    roi: Tuple[int, int, int, int],
    eps: float = 5.0,
    min_samples: int = 5,
) -> Dict[str, Any]:
    before = load_and_crop(before_path, roi)
    after = load_and_crop(after_path, roi)
    h, _ = before.shape[:2]

    mask = compute_diff_mask(before, after)
    bin_mask = (mask > 0).astype(np.uint8)

    # Contour clustering
    ct_clusters = cluster_binary_findcontours(bin_mask)
    ct_circles = fit_min_enclosing_circles(ct_clusters)
    ct_aligned = filter_aligned_circles(ct_circles, expected_num, roi[3])
    annotate_and_save(after, ct_aligned, "contour_circles.png")
    contour_results = [
        (cid, x, y, 2 * r)
        for cid, (x, y, r) in sorted(ct_aligned.items())[:expected_num]
    ]

    dbscan_results = []
    if len(ct_aligned) < expected_num:
        db_clusters = cluster_binary_dbscan(bin_mask, eps=eps, min_samples=min_samples)
        db_circles = fit_min_enclosing_circles(db_clusters)
        db_aligned = filter_aligned_circles(db_circles, expected_num, roi[3])
        annotate_and_save(after, db_aligned, "dbscan_circles.png")
        dbscan_results = [
            (cid, x, y, 2 * r)
            for cid, (x, y, r) in sorted(db_aligned.items())[:expected_num]
        ]
    else:
        print("Skipping DBSCAN; contour aligned detected enough circles.")

    return {"contour": contour_results, "dbscan": dbscan_results}


if __name__ == "__main__":
    before = r"Projects\\Sac DropV2\\Images\\overview_photo_20250711_103353.bmp"
    after = r"Projects\\Sac DropV2\\Images\\overview_photo_20250711_103547.bmp"
    expected_num = 4
    roi = (2960, 1300, 4300 - 2960, 1600 - 1300)
    results = run_shotgun_clustering(before, after, expected_num, roi)
    print(results)
