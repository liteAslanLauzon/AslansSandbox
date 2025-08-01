import cv2
import numpy as np

# 1. Load and preprocess
img = cv2.imread(r"Projects\lcosEdgeFinder\photo\cropped.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 2. Global threshold (as before)
_, thresh = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY)

# 3. Top-Hat to highlight bright regions (rectangles + grid lines)
#    kernel_bg should be JUST larger than your brick motif
kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_bg)

# 4. Binarize the top-hat result
_, th_tophat = cv2.threshold(tophat, 50, 255, cv2.THRESH_BINARY)

# 5. Remove the cross-hatch: first detect & subtract horizontal lines...
horiz_size = 40  # length of your grid lines in pixels; tune this
kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_size, 1))
horiz_lines = cv2.morphologyEx(th_tophat, cv2.MORPH_OPEN, kernel_h)
no_horiz = cv2.subtract(th_tophat, horiz_lines)

# 6. …then vertical lines
vert_size = 40  # height of your grid lines in pixels; tune this
kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_size))
vert_lines = cv2.morphologyEx(no_horiz, cv2.MORPH_OPEN, kernel_v)
clean_lines = cv2.subtract(no_horiz, vert_lines)

# 7. Clean up residual noise
kernel_cleanup = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
clean = cv2.morphologyEx(clean_lines, cv2.MORPH_CLOSE, kernel_cleanup)

# 8. Find & filter contours
contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rectangles = []
for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    area = cv2.contourArea(approx)
    if len(approx) == 4 and area > 1000:
        rectangles.append(approx)

# 9. Draw the results
output = img.copy()
cv2.drawContours(output, rectangles, -1, (0, 255, 0), 2)

# 10. Show everything
cv2.imshow("Original Threshold", thresh)
cv2.imshow("Top-Hat (pre-line removal)", th_tophat)
cv2.imshow("After Removing Horiz Lines", no_horiz)
cv2.imshow("After Removing Vert Lines", clean_lines)
cv2.imshow("Final Clean Mask", clean)
cv2.imshow("Detected Rectangles", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
