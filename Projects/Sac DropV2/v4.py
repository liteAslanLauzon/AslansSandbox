import cv2
import numpy as np

# 1) Load and preprocess
img = cv2.imread(r"Projects\Sac DropV2\Images\croppedRotated.png")
if img is None:
    raise FileNotFoundError("Image not found at specified path")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)

# 2) Edge detection
edges = cv2.Canny(gray, 50, 150)

# 3) Find all external contours in the edge map
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
if len(contours) < 4:
    raise RuntimeError(f"Only found {len(contours)} contours; need at least 4")

# 4) Sort contours by perimeter (arc length), descending, take top 4
sorted_contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)
top4 = sorted_contours[:4]

# 5) Prepare drawing canvas
out = img.copy()

# 6) For each of the four, fit circle & compute centroid, then draw
for idx, cnt in enumerate(top4, start=1):
    # 6a) fit minimum enclosing circle
    (xc, yc), r = cv2.minEnclosingCircle(cnt)
    center = (int(xc), int(yc))
    radius = int(r)

    # 6b) compute contour centroid via moments
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = center
    centroid = (cx, cy)


    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    col = colors[(idx - 1) % len(colors)]
    cv2.circle(out, center, radius, col, 2)
    cv2.circle(out, centroid, 4, col, -1)

    # optional: label them
    cv2.putText(out, f"{idx}", (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

# 7) Display all four at once
cv2.imshow("Top 4 Fitted Circles + Centroids", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
