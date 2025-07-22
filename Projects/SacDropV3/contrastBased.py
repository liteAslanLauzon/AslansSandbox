import cv2
import numpy as np
from typing import Tuple, Optional, List

# Toggle debug image display
DEBUG = True


def crop_roi(img: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop the image to the rectangle defined by (x_min, x_max, y_min, y_max).
    """
    x_min, x_max, y_min, y_max = roi
    return img[y_min:y_max, x_min:x_max]


def find_contrast_regions(
    img: np.ndarray,
    canny_thresh1: int = 50,
    canny_thresh2: int = 150,
    morph_kernel_size: int = 5,
    num_regions: int = 4,
) -> List[Tuple[int, int, int, int]]:
    """
    Use edge detection (Canny) to find high-contrast regions, clean up noise,
    then pick the top regions by area and return their bounding boxes.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Canny edge detection to find contrast edges
    edges = cv2.Canny(gray, canny_thresh1, canny_thresh2)
    if DEBUG:
        cv2.imshow("Edges", edges)
        cv2.waitKey(0)


    # Morphological closing to fill gaps
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
    )
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    if DEBUG:
        cv2.imshow("Edges Closed", closed)
        cv2.waitKey(0)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Sort contours by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Take top N contours and compute bounding boxes
    boxes = []
    for cnt in contours[:num_regions]:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))
    return boxes


if __name__ == "__main__":
    IMAGE_PATH = "Projects/SacDropV3/Images/contrast option.bmp"

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Error: could not load '{IMAGE_PATH}'")
        exit(1)

    # compute a square ROI at the image center (size configurable)
    h, w = img.shape[:2]
    half_size = 500
    cx, cy = w // 2, h // 2
    x_min = max(cx - half_size, 0)
    x_max = min(cx + half_size, w)
    y_min = max(cy - half_size, 0)
    y_max = min(cy + half_size, h)
    ROI = (x_min, x_max, y_min, y_max)

    roi_img = crop_roi(img, ROI)

    # find the four largest high-contrast regions
    boxes = find_contrast_regions(roi_img)

    # Draw bounding boxes on the ROI
    disp = roi_img.copy()
    for i, (x, y, w_box, h_box) in enumerate(boxes):
        color = (0, 255, 0)  # green boxes
        cv2.rectangle(disp, (x, y), (x + w_box, y + h_box), color, 2)
        if DEBUG:
            cv2.putText(
                disp, f"#{i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
    if DEBUG:
        cv2.imshow("Detected Regions", disp)
        cv2.waitKey(0)
    else:
        output_path = "detected_regions.png"
        cv2.imwrite(output_path, disp)
        print(f"Output saved to {output_path}")

    cv2.destroyAllWindows()
