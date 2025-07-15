import cv2
import numpy as np

# --- CONFIG ---
IMAGE_PATH = "Projects\\Sac DropV2\\Images\\croppedRotated.png"  # path to your image
THRESH_DIFF = 30  # how different in gray-level from background
MIN_AREA = 1000  # ignore tiny noise

# --- LOAD & PREPARE ---
# 1. Read and convert to grayscale
img = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Estimate background gray level as the most common pixel value
hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
bg_gray = np.argmax(hist)

# 3. Compute absolute difference from background
diff = cv2.absdiff(gray, np.full_like(gray, bg_gray))

# 4. Threshold that difference to isolate shapes
_, mask = cv2.threshold(diff, THRESH_DIFF, 255, cv2.THRESH_BINARY)

# 5. (Optional) Clean up noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# --- FIND & DRAW CONTOURS ---
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out very small areas
shapes = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]

# Draw contours and label them
for i, cnt in enumerate(shapes, start=1):
    # draw outline in green
    cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
    # compute centroid for placing label
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(
            img, f"{i}", (cx - 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

# Report
print(f"Detected {len(shapes)} shapes different from background (expected 4).")

# --- SHOW & SAVE ---
cv2.imshow("Shapes Detected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Optionally save result:
cv2.imwrite("shapes_detected.png", img)
