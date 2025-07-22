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


def find_dark_ring(
        
    img: np.ndarray, thresh: int = 162, morph_kernel_size: int = 3
) -> Optional[Tuple[int, int, int]]:
    """
    Threshold to find dark regions, clean up noise, then pick the largest contour
    and return its center and radius (from its bounding box).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Invert threshold to isolate dark areas
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    if DEBUG:
        cv2.imshow("Dark mask (raw)", mask)
        cv2.waitKey(0)

        # Apply Gaussian blur to reduce noise before morphology
        blurred = cv2.bilateralFilter(mask, 9, 25, 100)


        # Morphological closing to fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size,) * 2)
        clean = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel, iterations=2)
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)
        if DEBUG:
            cv2.imshow("Dark mask (cleaned)", clean)
            cv2.waitKey(0)

    # Perform another threshold to further isolate dark regions
    _, clean = cv2.threshold(clean, thresh // 2, 255, cv2.THRESH_BINARY)
    if DEBUG:
        cv2.imshow("Dark mask (double threshold)", clean)
        cv2.waitKey(0)
        # Find all contours with hierarchy
    contours, hierarchy = cv2.findContours(
    clean,
    cv2.RETR_TREE,                # full nesting
    cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    # (Optional) show all raw contours for debugging
    if DEBUG:
        disp = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(disp, contours, -1, (0, 0, 255), 2)
        cv2.imshow("All contours", disp)
        cv2.waitKey(0)

    # Flatten hierarchy for easy access: each entry is [next, prev, first_child, parent]
    hier = hierarchy[0]

    # 2. Find all top-level (external) contours: parent == -1
    external_idxs = [i for i in range(len(contours)) if hier[i][3] == -1]
    if not external_idxs:
        return None

    # 3. Pick the largest external by area
    largest_ext_idx = max(
        external_idxs,
        key=lambda i: cv2.contourArea(contours[i])
    )

    # 4. Recursively find the deepest descendant of that contour
    def get_deepest(idx):
        # gather direct children
        child = hier[idx][2]
        children = []
        while child != -1:
            children.append(child)
            child = hier[child][0]  # next sibling

        # if no children, this is a leaf
        if not children:
            return idx, 0

        # otherwise, dive into each child and pick the branch with greatest depth
        max_depth = -1
        deepest_idx = idx
        for c in children:
            d_idx, depth = get_deepest(c)
            if depth > max_depth:
                max_depth = depth
                deepest_idx = d_idx

        return deepest_idx, max_depth + 1

    deepest_idx, _ = get_deepest(largest_ext_idx)
    target = contours[deepest_idx]

    # 5. Compute bounding box / center / “radius”
    x, y, w, h = cv2.boundingRect(target)
    cx, cy = x + w // 2, y + h // 2
    r = max(w, h) // 2

    if DEBUG:
        disp = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(disp, [target], -1, (0, 255, 0), 2)  # Green for target
        cv2.imshow("Target contour", disp)
        cv2.waitKey(0)

    if DEBUG:
        disp = img.copy()
        cv2.circle(disp, (cx, cy), r, (0, 255, 0), 2)
        cv2.imshow("Detected dark circle", disp)
        cv2.waitKey(0)

    return cx, cy, r


if __name__ == "__main__":
    IMAGE_PATH = "Projects\\SacDropV3\\Images\\camBg.bmp"

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Error: could not load '{IMAGE_PATH}'")
        exit(1)

    # compute a 400×400 box at the exact image center
    h, w = img.shape[:2]
    half_size = 500  # half of 400

    cx, cy = w // 2, h // 2
    x_min = max(cx - half_size, 0)
    x_max = min(cx + half_size, w)
    y_min = max(cy - half_size, 0)
    y_max = min(cy + half_size, h)

    # Display the ROI on the original image for verification
    roi_disp = img.copy()
    cv2.rectangle(roi_disp, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    # Resize the ROI display to width 800, keeping aspect ratio
    disp_h, disp_w = roi_disp.shape[:2]
    new_w = 800
    new_h = int(disp_h * (new_w / disp_w))
    cv2.waitKey(0)

    ROI = (x_min, x_max, y_min, y_max)

    roi_img = crop_roi(img, ROI)

    result = find_dark_ring(roi_img)
    
    
    cv2.destroyAllWindows()
