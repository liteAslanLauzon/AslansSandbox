import cv2
import numpy as np


def estimate_template_radius(tpl_bw):
    """
    Given a binary (0/255) circle template image, find its radius via
    the minimum enclosing circle of its largest contour.
    """
    cnts, _ = cv2.findContours(tpl_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("No contours found in template!")
    # assume the circle template is the largest contour
    c = max(cnts, key=cv2.contourArea)
    (_, _), radius = cv2.minEnclosingCircle(c)
    return int(round(radius))


def make_radii_list(base_r, scales=(0.5, 1.5), num_steps=20):
    """
    Returns a list of integer radii from scales[0]*base_r up to scales[1]*base_r,
    evenly sampled in num_steps steps.
    """
    min_r = max(1, int(round(base_r * scales[0])))
    max_r = int(round(base_r * scales[1]))
    # avoid zero‐step
    if max_r <= min_r:
        return [min_r]
    return list(np.linspace(min_r, max_r, num_steps, dtype=int))


def circle_template_match_bw(img, radii):
    """
    Match a perfect circle (any radius in `radii`) on a BW image via
    distance‐transform + sliding‐mask.
    `img` must be single‐channel (0–255).
    """
    # 1. Threshold & invert for DT
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    dist = cv2.distanceTransform(255 - bw, cv2.DIST_L2, 5)

    best_score = float("inf")
    best_r = None
    best_loc = None

    for r in radii:
        k = 2 * r + 1
        mask = np.zeros((k, k), dtype=np.uint8)
        cv2.circle(mask, (r, r), r, color=1, thickness=1)
        conv = cv2.filter2D(dist, -1, mask.astype(np.float32))
        min_val, _, min_loc, _ = cv2.minMaxLoc(conv)
        if min_val < best_score:
            best_score, best_r = min_val, r
            best_loc = (min_loc[0] + r, min_loc[1] + r)

    return best_r, best_loc, best_score


if __name__ == "__main__":
    # 1) load your TEMPLATE and IMAGE as grayscale
    tpl = cv2.imread("Projects\\Sac DropV2\\Images\\template.bmp", cv2.IMREAD_GRAYSCALE)
    img = cv2.imread("Projects\\Sac DropV2\\Images\\cropped.bmp", cv2.IMREAD_GRAYSCALE)
    if tpl is None or img is None:
        raise FileNotFoundError("Check your template or image path!")

    # 2) compute base template radius
    base_r = estimate_template_radius(tpl)
    print(f"Template radius = {base_r}px")

    # 3) build radii from 0.5× to 1.5× template size
    radii = make_radii_list(base_r, scales=(0.5, 1.5), num_steps=30)
    print(f"Searching radii: {radii[0]}…{radii[-1]} (total {len(radii)})")

    # 4) run matching
    best_r, (cx, cy), score = circle_template_match_bw(img, radii)
    print(f"Found circle: r={best_r}, center=({cx},{cy}), error={score:.1f}")

    # 5) visualize
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.circle(out, (cx, cy), best_r, (0, 255, 0), 2)
    cv2.circle(out, (cx, cy), 2, (0, 0, 255), 3)
    cv2.imshow("Result", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
