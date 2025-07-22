import cv2
import numpy as np


def method1_contours(gray, output):
    # 1. Simple binary threshold
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # 2. Find contours (external + hierarchy for holes)
    cnts, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # 3. Filter for ring-like shapes
    for idx, cnt in enumerate(cnts):
        # Only outer contours (hierarchy[0][idx][3] == -1)
        if hierarchy[0][idx][3] != -1:
            continue
        area = cv2.contourArea(cnt)
        if area < 100:  # skip small noise
            continue
        perim = cv2.arcLength(cnt, True)
        circ = 4 * np.pi * area / (perim * perim + 1e-6)
        # look for fairly circular shapes
        if circ < 0.6:
            continue

        # check if it has a child (hole)
        child_idx = hierarchy[0][idx][2]
        if child_idx != -1:
            # draw outer contour in green, inner in red
            cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
            cv2.drawContours(output, [cnts[child_idx]], -1, (0, 0, 255), 2)

    cv2.imshow("Method 1 – Contours", output)


def method2_hough(gray, color):
    # 1. Edge detect
    edges = cv2.Canny(gray, 50, 150)
    # 2. Hough Circle Transform
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=200,
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for x, y, r in circles:
            cv2.circle(color, (x, y), r, (255, 0, 255), 2)
    cv2.imshow("Method 2 – HoughCircles", color)


def method3_connected_components(gray, color):
    # 1. Threshold for blob detection
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # 2. Connected components
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw)
    for i in range(1, numLabels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 100 or area > 5000:
            continue
        # mask of this component
        mask = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = cnts[0]
        perim = cv2.arcLength(cnt, True)
        circ = 4 * np.pi * area / (perim * perim + 1e-6)
        if circ < 0.7:
            continue
        # draw min enclosing circle
        (x, y), r = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(r)
        cv2.circle(color, center, radius, (0, 255, 255), 2)

    cv2.imshow("Method 3 – Connected Components", color)


def main():
    # Load and hardcode image path
    img = cv2.imread("Projects\\Sac DropV2\\Images\\croppedv2.bmp")
    if img is None:
        print("Error: could not load croppedv2.bmp.")
        return

    # Resize image to width=800, maintain aspect ratio
    h, w = img.shape[:2]
    new_w = 800
    new_h = int(h * new_w / w)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out1 = img.copy()
    out2 = img.copy()
    out3 = img.copy()

    method1_contours(gray, out1)
    method2_hough(gray, out2)
    method3_connected_components(gray, out3)

    print("Press any key in a window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
