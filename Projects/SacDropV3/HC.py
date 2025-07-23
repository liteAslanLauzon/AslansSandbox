import cv2
import numpy as np
import os
import glob
import argparse


def pipeline(
    image: np.ndarray,
    min_area: float = 10,
    max_error: float = 50.0,
    debug: bool = False,
):
    """
    Detect a full circle (via Hough + arc extraction or LSQ on two arcs), mask it,
    and interpolate across the two outermost gaps. Returns:
      output: image with green arcs + red interpolation
      mask: binary mask of the circle
      region: original pixels inside the circle
      area_pixels: # pixels in the circle
    """
    # 1) Crop to centered 800×800
    h, w = image.shape[:2]
    size = 800
    cx, cy = w // 2, h // 2
    x1 = max(cx - size // 2, 0)
    y1 = max(cy - size // 2, 0)
    cropped = image[y1 : y1 + size, x1 : x1 + size]

    # 2) Blur + grayscale
    blur = cv2.bilateralFilter(cropped, 9, 75, 75)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # prep outputs
    output = cropped.copy()
    h_crop, w_crop = gray.shape[:2]
    center_img = np.array([w_crop / 2.0, h_crop / 2.0])
    mask = np.zeros_like(gray)
    region = np.zeros_like(cropped)
    area_pixels = 0

    # 3) Try HoughCircles first
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=int(size * 0.3),
        maxRadius=int(size * 0.45),
    )

    cnt1 = cnt2 = None
    if circles is not None:
        # pick the circle nearest center
        circles = np.round(circles[0, :]).astype(int)
        x0, y0, r = min(
            circles, key=lambda c: np.linalg.norm(c[:2].astype(float) - center_img)
        )

        # 4) extract arcs by intersecting circle outline with Canny edges
        edges = cv2.Canny(gray, 50, 150)
        circle_outline = np.zeros_like(edges)
        cv2.circle(circle_outline, (x0, y0), r, 255, 2)
        arcs_mask = cv2.bitwise_and(edges, circle_outline)
        contours, _ = cv2.findContours(
            arcs_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # pick the two longest arc‐contours
        if len(contours) >= 2:
            contours = sorted(contours, key=cv2.arcLength, reverse=True)
            cnt1, cnt2 = contours[:2]
        # else: fall back to LSQ below

    if cnt1 is None or cnt2 is None:
        # 5) Hough failed or not enough arcs → fallback to your original contour+LSQ method
        _, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # filter by area, circularity & margin
        valid = []
        edge_margin = 20
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            if (
                x < edge_margin
                or y < edge_margin
                or x + w_box > w_crop - edge_margin
                or y + h_box > h_crop - edge_margin
            ):
                continue
            pts = cnt.reshape(-1, 2).astype(np.float32)
            (_, _), _r = cv2.minEnclosingCircle(pts)
            d = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
            if np.std(d - d.mean()) < max_error:
                valid.append(cnt)

        # need at least two
        if len(valid) < 2:
            valid = sorted(valid, key=cv2.contourArea, reverse=True)[:2]

        # pick the two outermost by mean distance from center
        def mean_dist(cnt):
            pts = cnt.reshape(-1, 2).astype(np.float64)
            return np.mean(np.linalg.norm(pts - center_img, axis=1))

        cnt1, cnt2 = sorted(valid, key=mean_dist, reverse=True)[:2]

        # LSQ‐fit full circle from those two arcs
        pts_all = np.vstack((cnt1.reshape(-1, 2), cnt2.reshape(-1, 2))).astype(
            np.float64
        )
        X, Y = pts_all[:, 0], pts_all[:, 1]
        M = np.column_stack([X, Y, np.ones_like(X)])
        b = -(X**2 + Y**2)
        A, B, C = np.linalg.lstsq(M, b, rcond=None)[0]
        x0, y0 = -A / 2, -B / 2
        r = np.sqrt(x0**2 + y0**2 - C)
        x0, y0, r = map(int, map(round, (x0, y0, r)))

    # 6) Draw mask & extract region
    cv2.circle(mask, (x0, y0), r, 255, -1)
    region = cv2.bitwise_and(cropped, cropped, mask=mask)
    area_pixels = int(cv2.countNonZero(mask))

    # 7) Draw the two arc‐contours in green
    cv2.drawContours(output, [cnt1, cnt2], -1, (0, 255, 0), 2)

    # 8) Interpolate across their two gaps in red
    def sorted_angles(cnt):
        pts = cnt.reshape(-1, 2)
        rel = pts - np.array([x0, y0])
        return np.sort(np.arctan2(rel[:, 1], rel[:, 0]))

    a1, a2 = sorted_angles(cnt1), sorted_angles(cnt2)
    gaps = []
    for start, end in [(a1[-1], a2[0]), (a2[-1], a1[0])]:
        if end <= start:
            end += 2 * np.pi
        gaps.append((start, end))

    for start, end in gaps:
        angles = np.linspace(start, end, 100)
        pts_arc = np.column_stack(
            [x0 + r * np.cos(angles), y0 + r * np.sin(angles)]
        ).astype(int)
        cv2.polylines(output, [pts_arc], False, (0, 0, 255), 2)

    # 9) Debug display
    if debug:
        for name, img in [
            ("Final Output", output),
            ("Mask", mask),
            ("Region", region),
        ]:
            cv2.imshow(f"Debug - {name}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output, mask, region, area_pixels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process circle images")
    parser.add_argument(
        "--dir",
        type=str,
        default="Projects\\SacDropV3\\ccgOverviewNewer",
        help="Image directory",
    )
    parser.add_argument("--debug", action="store_true", help="Show debug windows")
    args = parser.parse_args()

    image_dir = args.dir
    exts = ("*.bmp", "*.png", "*.jpg", "*.jpeg")
    output_dir = os.path.join(image_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "area_summary.csv")
    with open(summary_path, "w") as f_summary:
        f_summary.write("filename,area_pixels\n")
        for ext in exts:
            for image_path in glob.glob(os.path.join(image_dir, ext)):
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Failed to load {image_path}")
                    continue
                base = os.path.splitext(os.path.basename(image_path))[0]
                print(f"Processing {base}...")
                out_img, mask, region, area = pipeline(img, debug=args.debug)
                cv2.imwrite(os.path.join(output_dir, f"{base}_processed.png"), out_img)
                cv2.imwrite(os.path.join(output_dir, f"{base}_mask.png"), mask)
                cv2.imwrite(os.path.join(output_dir, f"{base}_circle.png"), region)
                f_summary.write(f"{base},{area}\n")

    print(f"Batch complete. Masks, regions saved. Area summary at {summary_path}")
