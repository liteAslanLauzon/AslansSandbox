# Snippet: Ring Detection via Image Subtraction, Priority & Extrapolation
import cv2
import numpy as np

# Paths for before/after images
path_before = "Projects\\Sac DropV2\\Images\\before_circles.bmp"
path_after = "Projects\\Sac DropV2\\Images\\after_circles.bmp"

# Load images
print(f"Loading before-image '{path_before}'...")
before = cv2.imread(path_before)
if before is None:
    print(f"Error: could not load '{path_before}'")
    exit(1)
print(f"Loading after-image '{path_after}'...")
after = cv2.imread(path_after)
if after is None:
    print(f"Error: could not load '{path_after}'")
    exit(1)

# Resize both to width=800px (maintain aspect ratio)
h, w = after.shape[:2]
new_w = 800
new_h = int(h * new_w / w)
before = cv2.resize(before, (new_w, new_h), interpolation=cv2.INTER_AREA)
after = cv2.resize(after, (new_w, new_h), interpolation=cv2.INTER_AREA)
print(f"Resized images to: {new_w}x{new_h}")

# Convert to grayscale
gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# 1. Image subtraction (detect new circles)
diff = cv2.absdiff(gray_after, gray_before)
print("Computed absolute difference between after and before images.")
cv2.imshow("Difference", diff)

# 2. Threshold the difference
_, bw = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
print("Applied threshold to isolate circle regions.")
cv2.imshow("Thresholded Difference", bw)

# 3. Find contours on thresholded diff
cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Total contours found in diff: {len(cnts)}")

# Prioritize by proximity to center
dx, dy = new_w / 2, new_h / 2
contours = []
for cnt in cnts:
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    dist = np.hypot(cx - dx, cy - dy)
    contours.append((dist, cnt))
contours.sort(key=lambda x: x[0])

# 4. Draw initial contours
initial = after.copy()
print("Drawing prioritized contours on 'after' image...")
for _, cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 50:
        continue
    cv2.drawContours(initial, [cnt], -1, (0, 255, 0), 2)
cv2.imshow("Detected Contours", initial)

# 5. Extrapolate full circles
extrap = after.copy()
print("Drawing extrapolated circles via minEnclosingCircle...")
for _, cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 50:
        continue
    (x, y), r = cv2.minEnclosingCircle(cnt)
    cv2.circle(extrap, (int(x), int(y)), int(r), (0, 0, 255), 2)
cv2.imshow("Extrapolated Circles", extrap)

print("Press any key in any window to exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Done.")
