import cv2
import numpy as np


def pipeline(image: np.ndarray, min_area: float = 10, max_error: float = 50.0):
    """
    Crop, blur, threshold, filter contours by area & circularity,
    then interpolate both gaps between the two best arcs.
    - min_area: minimum contour area to keep
    - max_error: max std dev of radial deviation to keep
    """
    # 1) Crop to centered 700×700 square
    h, w = image.shape[:2]
    size = 700
    cx, cy = w // 2, h // 2
    x1 = max(cx - size // 2, 0)
    y1 = max(cy - size // 2, 0)
    cropped = image[y1 : y1 + size, x1 : x1 + size]

    # 2) Blur + grayscale
    blur = cv2.bilateralFilter(cropped, 9, 75, 75)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY) if blur.ndim == 3 else blur

    # 3) Threshold
    _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

    # 4) Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5) Filter by area and circularity error
    valid = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        pts = cnt.reshape(-1, 2).astype(np.float32)
        (x0, y0), _ = cv2.minEnclosingCircle(pts)
        d = np.linalg.norm(pts - [x0, y0], axis=1)
        if np.std(d - np.mean(d)) < max_error:  # use deviation from mean radius
            valid.append(cnt)

    # 6) Draw filtered contours
    output = cropped.copy()
    cv2.drawContours(output, valid, -1, (0, 255, 0), 2)

    # 7) If ≥2 valid arcs, interpolate both gaps
    if len(valid) >= 2:
        cnt1, cnt2 = sorted(valid, key=cv2.contourArea, reverse=True)[:2]
        pts_all = np.vstack((cnt1.reshape(-1, 2), cnt2.reshape(-1, 2))).astype(
            np.float32
        )
        # recalc center via circle fit
        (x0, y0), _ = cv2.minEnclosingCircle(pts_all)
        # compute radius as mean distance to center
        dists = np.linalg.norm(pts_all - [x0, y0], axis=1)
        r = np.mean(dists)

        def sorted_angles(cnt):
            pts = cnt.reshape(-1, 2)
            rel = pts - np.array([x0, y0])
            ang = np.arctan2(rel[:, 1], rel[:, 0])
            return np.sort(ang)

        a1, a2 = sorted_angles(cnt1), sorted_angles(cnt2)
        min1, max1 = a1[0], a1[-1]
        min2, max2 = a2[0], a2[-1]

        # gap intervals
        s1, e1 = max1, min2
        if e1 <= s1:
            e1 += 2 * np.pi
        s2, e2 = max2, min1
        if e2 <= s2:
            e2 += 2 * np.pi

        # sample and draw both arcs including endpoints
        N = 100
        for start, end in [(s1, e1), (s2, e2)]:
            angles = np.linspace(start, end, N)
            pts_arc = np.column_stack(
                (x0 + r * np.cos(angles), y0 + r * np.sin(angles))
            ).astype(int)
            cv2.polylines(output, [pts_arc], False, (0, 0, 255), 2)

    # Show result
    cv2.imshow("Filtered Circle Fit", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return output


if __name__ == "__main__":
    img = cv2.imread("Projects/SacDropV3/Images/ccgCoverEx.bmp")
    if img is None:
        print("Failed to load image")
    else:
        pipeline(img)
