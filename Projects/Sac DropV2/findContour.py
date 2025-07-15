import cv2
import numpy as np


def contour_circle_match(img, min_radius=10, max_radius=100, min_circularity=0.7):
    """
    Finds the most-circle-like contour in a binary image.
    Args:
      img              : single-channel binary (0/255) or grayscale image
      min_radius       : ignore circles smaller than this
      max_radius       : ignore circles larger than this
      min_circularity  : drop contours with circularity below this
    Returns:
      (best_r, (cx,cy), best_circ) or (None, None, None) if no good circle
    """

    # 1. Threshold to clean up any gray (if already 0/255 you can skip this)
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 2. Find contours
    cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = (None, None, 0.0)  # (radius, center, circularity)

    for c in cnts:
        area = cv2.contourArea(c)
        if area <= 0:
            continue
        peri = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (peri * peri)  # circularity in [0,1]

        # reject low-circularity shapes
        if circ < min_circularity:
            continue

        # get enclosing circle
        (x, y), r = cv2.minEnclosingCircle(c)
        if not (min_radius <= r <= max_radius):
            continue

        # pick the one with circularity closest to 1
        if circ > best[2]:
            best = (int(round(r)), (int(round(x)), int(round(y))), circ)

    return best  # e.g. (r, (cx,cy), circularity)


if __name__ == "__main__":
    # Load your BW image
    img = cv2.imread(
        r"Projects\Sac DropV2\\Images\\croppedRotated.png", cv2.IMREAD_GRAYSCALE
    )
    if img is None:
        raise FileNotFoundError("Failed to load image—check path!")

    # Run contour-based detection
    result = contour_circle_match(
        img, min_radius=10, max_radius=200, min_circularity=0.3
    )

    if result[0] is None:
        print("No sufficiently circular contour found.")
    else:
        best_r, (cx, cy), best_circ = result
        print(f"Detected circle: center=({cx},{cy}), r={best_r}, circ={best_circ:.3f}")

        # visualize
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.circle(out, (cx, cy), best_r, (0, 255, 0), 2)
        cv2.circle(out, (cx, cy), 2, (0, 0, 255), 3)
        cv2.imshow("Contour-based Circle", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
