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
    Detect circle arcs, fit full circle, extract the circular region, compute its area,
    and optionally display.
    - min_area: minimum contour area to keep
    - max_error: max std dev of radial deviation to keep
    - debug: if True, show intermediate and final images
    Returns:
      output: image with contours and interpolation drawn
      mask: single-channel mask of the filled circle region
      region: original image cropped to the circular region
      area: number of pixels in the circular region
    """
    # 1) Crop to centered 700×700 square
    h, w = image.shape[:2]
    size = 800
    cx, cy = w // 2, h // 2
    x1 = max(cx - size // 2, 0)
    y1 = max(cy - size // 2, 0)
    cropped = image[y1 : y1 + size, x1 : x1 + size]

    # 2) Blur + grayscale
    blur = cv2.bilateralFilter(cropped, 9, 75, 75)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY) if blur.ndim == 3 else blur
    if debug:
        cv2.imshow("Debug - Gray", gray)

    # 3) Threshold
    _, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    if debug:
        cv2.imshow("Debug - Threshold", thresh)

    # 4) Find contours **and** hierarchy, then keep only outer (parent==-1)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    hier = hierarchy[0]
    outer_contours = [cnt for cnt, h in zip(contours, hier) if h[3] == -1]

    # 5) Filter by area and circularity error
    h_crop, w_crop = gray.shape[:2]
    valid = []
    edge_margin = 20
    for cnt in outer_contours:
        area_cnt = cv2.contourArea(cnt)
        if area_cnt < min_area:
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
        (cx0, cy0), _ = cv2.minEnclosingCircle(pts)
        d = np.linalg.norm(pts - [cx0, cy0], axis=1)
        if np.std(d - np.mean(d)) < max_error:
            valid.append(cnt)
    if debug:
        dbg = cropped.copy()
        cv2.drawContours(dbg, valid, -1, (0, 255, 0), 2)
        cv2.imshow("Debug - Filtered Contours", dbg)

    # Prepare outputs
    output = cropped.copy()
    cv2.drawContours(output, valid, -1, (0, 255, 0), 2)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    region = np.zeros_like(cropped)
    area_pixels = 0

    # 6) If ≥2 valid arcs, fit circle and create mask/region
    if len(valid) >= 2:
        # a) compute crop‐center
        h_crop, w_crop = gray.shape[:2]
        center_img = np.array([w_crop / 2, h_crop / 2])

        # b) collect (cnt, area, mean_dist)
        stats = []
        for cnt in valid:
            area_cnt = cv2.contourArea(cnt)
            pts = cnt.reshape(-1, 2).astype(np.float64)
            mean_dist = np.mean(np.linalg.norm(pts - center_img, axis=1))
            stats.append((cnt, area_cnt, mean_dist))

        # c) filter by area threshold
        max_area = max(area for (_, area, _) in stats)
        area_thresh = 0.85 * max_area
        big = [(cnt, area, dist) for (cnt, area, dist) in stats if area >= area_thresh]

        if len(big) >= 2:
            # use the two biggest-ones that are farthest out
            big_sorted = sorted(big, key=lambda x: x[2], reverse=True)
            cnt1, cnt2 = big_sorted[0][0], big_sorted[1][0]
        else:
            # fallback → just take the two largest by area
            stats_sorted = sorted(stats, key=lambda x: x[1], reverse=True)
            cnt1, cnt2 = stats_sorted[0][0], stats_sorted[1][0]

        pts_all = np.vstack((cnt1.reshape(-1, 2), cnt2.reshape(-1, 2))).astype(
            np.float64
        )

        # least-squares circle fit
        X = pts_all[:, 0]
        Y = pts_all[:, 1]
        M = np.column_stack([X, Y, np.ones_like(X)])
        b = -(X**2 + Y**2)
        A, B, C = np.linalg.lstsq(M, b, rcond=None)[0]
        x0 = -A / 2
        y0 = -B / 2
        r = np.sqrt(x0**2 + y0**2 - C)

        # draw mask of circle region
        center_int = (int(round(x0)), int(round(y0)))
        radius_int = int(round(r))
        cv2.circle(mask, center_int, radius_int, 255, -1)
        # compute area
        area_pixels = int(cv2.countNonZero(mask))
        # extract region
        region = cv2.bitwise_and(cropped, cropped, mask=mask)
        if debug:
            cv2.imshow("Debug - Mask", mask)
            cv2.imshow("Debug - Extracted Region", region)

        # compute gaps and draw interpolations
        def sorted_angles(cnt):
            pts = cnt.reshape(-1, 2)
            rel = pts - np.array([x0, y0])
            return np.sort(np.arctan2(rel[:, 1], rel[:, 0]))

        a1 = sorted_angles(cnt1)
        a2 = sorted_angles(cnt2)
        gaps = []
        for start, end in [(a1[-1], a2[0]), (a2[-1], a1[0])]:
            if end <= start:
                end += 2 * np.pi
            gaps.append((start, end))

        N = 100
        for start, end in gaps:
            angles = np.linspace(start, end, N)
            pts_arc = np.column_stack(
                [x0 + r * np.cos(angles), y0 + r * np.sin(angles)]
            ).astype(int)
            cv2.polylines(output, [pts_arc], False, (0, 0, 255), 2)

    if debug:
        cv2.imshow("Debug - Final Output", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output, mask, region, area_pixels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process circle images")
    parser.add_argument(
        "--dir",
        type=str,
        default="Projects\\SacDropV3\\photosv3",
        help="Image directory",
    )
    parser.add_argument("--debug", action="store_true", help="Show debug windows")
    args = parser.parse_args()

    image_dir = args.dir
    exts = ("*.bmp", "*.png", "*.jpg", "*.jpeg")
    output_dir = os.path.join(image_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # optional summary file
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
                # save outputs
                cv2.imwrite(os.path.join(output_dir, f"{base}_processed.png"), out_img)
                cv2.imwrite(os.path.join(output_dir, f"{base}_mask.png"), mask)
                cv2.imwrite(os.path.join(output_dir, f"{base}_circle.png"), region)
                # log area
                f_summary.write(f"{base},{area}\n")
    print(f"Batch complete. Masks, regions saved. Area summary at {summary_path}")
