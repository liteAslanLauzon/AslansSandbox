import cv2
import numpy as np
import math


def main():
    # 1. Load the image in grayscale
    img_path = "Projects/lcosEdgeFinder/photo/ORIGINAL.bmp"
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not load image at '{img_path}'")

    # 2. Crop to your ROI
    #    (y1:y2, x1:x2) = (2530:3600, 2240:3130)
    crop = gray[2530:3600, 2240:3130]
    h, w = crop.shape

    # 3. Threshold to isolate dark parts (black → white)
    _, thresh = cv2.threshold(crop, 50, 255, cv2.THRESH_BINARY_INV)

    # 4. Morphological clean (open then close)
    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.morphologyEx(
        cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel
    )

    # 5. Edge detect + Probabilistic Hough
    edges = cv2.Canny(clean, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10
    )

    # 6. Try filtering for right-edge segments (OR filter, multiple margins)
    segment = None
    if lines is not None:
        for margin in (50, 100, 200):
            x_thresh = w - margin
            candidates = [
                tuple(l[0])  # (x1,y1,x2,y2)
                for l in lines
                if l[0][0] > x_thresh or l[0][2] > x_thresh
            ]
            if candidates:
                # pick the longest one
                segment = max(
                    candidates, key=lambda s: math.hypot(s[2] - s[0], s[3] - s[1])
                )
                break

    # 7. Fallback: fitLine on all edge‐points near the right border
    if segment is None:
        x_border = w - 50
        pts = np.column_stack(np.where(clean > 0))  # pts = [[y,x],...]
        pts = pts[pts[:, 1] > x_border]  # keep only x > border
        if len(pts) < 2:
            raise RuntimeError("Still no line data on the right edge.")
        # Prepare for fitLine: convert to [[x,y],...]
        xy = np.fliplr(pts)
        vx, vy, x0, y0 = cv2.fitLine(xy, cv2.DIST_L2, 0, 0.01, 0.01)

        # Compute the intersections with the top (y=0) and bottom (y=h) of the crop
        t_top = (0 - y0) / vy
        x_top = x0 + vx * t_top
        t_bot = (h - y0) / vy
        x_bot = x0 + vx * t_bot

        segment = (int(x_top), 0, int(x_bot), h)

    # Unpack the final segment endpoints
    x1, y1, x2, y2 = segment

    # 8. Compute angle relative to Y‑axis
    dx, dy = x2 - x1, y2 - y1
    angle_rad = math.atan2(dx, dy)
    angle_deg = abs(math.degrees(angle_rad))
    print(f"Angle relative to Y‑axis: {angle_deg:.2f}°")

    # 9. Visualize on a resized clean mask
    disp_w = 800
    disp_h = int(disp_w * h / w)
    clean_disp = cv2.resize(clean, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    sx, sy = disp_w / w, disp_h / h

    p1 = (int(x1 * sx), int(y1 * sy))
    p2 = (int(x2 * sx), int(y2 * sy))
    vis = cv2.cvtColor(clean_disp, cv2.COLOR_GRAY2BGR)
    cv2.line(vis, p1, p2, (0, 255, 0), 2)

    cv2.imshow("Clean Mask", clean_disp)
    cv2.imshow("Right‑Edge Line", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
