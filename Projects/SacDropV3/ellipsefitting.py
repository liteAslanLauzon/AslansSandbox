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
    Detect circle arcs, fit full ellipse, extract elliptical region, compute area,
    and optionally display—including highlighting the ROI on the original image.
    """
    # --- 1) Crop to centered 800×800 square ---
    h, w = image.shape[:2]
    size = 800
    cx, cy = w // 2, h // 2
    x1 = max(cx - size // 2, 0)
    y1 = max(cy - size // 2, 0)
    cropped = image[y1 : y1 + size, x1 : x1 + size]

    # --- 2) Blur + grayscale ---
    blur = cv2.bilateralFilter(cropped, 9, 75, 75)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY) if blur.ndim == 3 else blur
    if debug:
        cv2.imshow("Debug - Gray", gray)

    # --- 3) Threshold ---
    _, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    if debug:
        cv2.imshow("Debug - Threshold", thresh)

    # --- 4) Find outer contours ---
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    hier = hierarchy[0]
    outer = [cnt for cnt, h in zip(contours, hier) if h[3] == -1]

    # --- 5) Filter by area, margin, circularity error ---
    h_crop, w_crop = gray.shape[:2]
    valid = []
    edge_margin = 20
    for cnt in outer:
        area_cnt = cv2.contourArea(cnt)
        if area_cnt < min_area:
            continue

        x, y, wb, hb = cv2.boundingRect(cnt)
        if (
            x < edge_margin
            or y < edge_margin
            or x + wb > w_crop - edge_margin
            or y + hb > h_crop - edge_margin
        ):
            continue

        pts = cnt.reshape(-1, 2).astype(np.float32)
        (cx0, cy0), _ = cv2.minEnclosingCircle(pts)
        d = np.linalg.norm(pts - [cx0, cy0], axis=1)
        if np.std(d - np.mean(d)) < max_error:
            valid.append(cnt)

    # compute bounding-box of all valid contours (in cropped coords)
    if valid:
        all_pts = np.vstack([cnt.reshape(-1, 2) for cnt in valid]).astype(np.int32)
        vx, vy, vw, vh = cv2.boundingRect(all_pts)
    else:
        vx = vy = vw = vh = None

    if debug:
        dbg = cropped.copy()
        cv2.drawContours(dbg, valid, -1, (0, 255, 0), 2)
        cv2.imshow("Debug - Filtered Contours (Cropped)", dbg)

    # --- prepare outputs ---
    output = cropped.copy()
    cv2.drawContours(output, valid, -1, (0, 255, 0), 2)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    region = np.zeros_like(cropped)
    area_pixels = 0

    # --- 6) If ≥2 valid arcs, fit ellipse and draw only the missing gaps ---
    if len(valid) >= 2:
        stats = [(cnt, cv2.contourArea(cnt)) for cnt in valid]
        stats.sort(key=lambda x: x[1], reverse=True)
        cnt1, cnt2 = stats[0][0], stats[1][0]

        pts_all = np.vstack((cnt1.reshape(-1, 2), cnt2.reshape(-1, 2))).astype(
            np.float32
        )
        ellipse = cv2.fitEllipse(pts_all)
        (xc, yc), (MA, ma), rot = ellipse
        center_int = (int(round(xc)), int(round(yc)))
        axes_int = (int(round(MA / 2)), int(round(ma / 2)))

        # filled ellipse mask & region
        cv2.ellipse(mask, center_int, axes_int, rot, 0, 360, 255, -1)
        area_pixels = int(cv2.countNonZero(mask))
        region = cv2.bitwise_and(cropped, cropped, mask=mask)

        if debug:
            cv2.imshow("Debug - Mask", mask)
            cv2.imshow("Debug - Extracted Region", region)

        # helper to find each contour’s true angular span
        def contour_span(cnt):
            pts = cnt.reshape(-1, 2)
            rel = pts - np.array([xc, yc])
            ang = np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2 * np.pi)
            ang_sorted = np.sort(ang)
            diffs = np.diff(np.concatenate([ang_sorted, [ang_sorted[0] + 2 * np.pi]]))
            idx = np.argmax(diffs)
            start = ang_sorted[(idx + 1) % len(ang_sorted)]
            end = ang_sorted[idx]
            if end < start:
                end += 2 * np.pi
            return start, end

        span1 = contour_span(cnt1)
        span2 = contour_span(cnt2)

        # compute true gaps
        intervals = sorted([span1, span2], key=lambda x: x[0])
        gaps = [
            (intervals[0][1], intervals[1][0]),
            (intervals[1][1], intervals[0][0] + 2 * np.pi),
        ]

        # draw missing gaps
        for start, end in gaps:
            if end <= start:
                continue
            start_deg = (np.degrees(start)) % 360
            end_deg = (np.degrees(end)) % 360
            cv2.ellipse(
                output, center_int, axes_int, rot, start_deg, end_deg, (0, 0, 255), 2
            )

    # --- final debug: show output & then highlight ROI on original ---
    if debug:
        cv2.imshow("Debug - Final Output (Cropped)", output)

        print(
            "Here is the area that pipeline found—in the ORIGINAL image highlighted below:"
        )
        debug_orig = image.copy()
        if vx is not None and len(valid) >= 2:
            # Draw the fitted ellipse as a filled region on the original image
            ellipse_center = (int(round(x1 + center_int[0])), int(round(y1 + center_int[1])))
            ellipse_axes = axes_int
            ellipse_angle = rot
            overlay = debug_orig.copy()
            cv2.ellipse(
            overlay,
            ellipse_center,
            ellipse_axes,
            ellipse_angle,
            0,
            360,
            (0, 255, 255),  # yellow fill
            -1,
            )
            alpha = 0.1
            cv2.addWeighted(overlay, alpha, debug_orig, 1 - alpha, 0, debug_orig)
        else:
            print("  No valid contours to highlight.")
        # Resize debug_orig to 800 pixels in height, preserving aspect ratio
        h_orig, w_orig = debug_orig.shape[:2]
        new_h = 800
        new_w = int(w_orig * (new_h / h_orig))
        debug_resized = cv2.resize(debug_orig, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imshow("Debug - Detected Region on Original", debug_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output, mask, region, area_pixels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process with ellipse-gap interpolation and ROI debug"
    )
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
                out_img, mask_img, region_img, area = pipeline(img, debug=args.debug)
                cv2.imwrite(os.path.join(output_dir, f"{base}_processed.png"), out_img)
                cv2.imwrite(os.path.join(output_dir, f"{base}_mask.png"), mask_img)
                cv2.imwrite(os.path.join(output_dir, f"{base}_ellipse.png"), region_img)
                f_summary.write(f"{base},{area}\n")
    print(f"Batch complete. Masks, regions saved. Area summary at {summary_path}")
