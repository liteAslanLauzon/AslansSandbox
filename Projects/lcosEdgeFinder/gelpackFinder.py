import cv2
import numpy as np

# 1. Load & preprocess
img = cv2.imread("Projects\\lcosEdgeFinder\\photo\\cropped.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)

# 2. Morphological extraction of lines
#    Adjust kernel sizes to match your line thickness
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))

vert_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
horiz_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel)

grid_mask = cv2.bitwise_or(vert_lines, horiz_lines)

# 3. Find the largest contour (the grid)
contours, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(contours, key=cv2.contourArea)

# Get bounding rect (axis-aligned)
x, y, w, h = cv2.boundingRect(largest)

# If you want a rotated rectangle:
# rect = cv2.minAreaRect(largest)
# box = cv2.boxPoints(rect)
# box = np.int0(box)

# 4. Visualize / crop
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
# roi = img[y:y+h, x:x+w]

cv2.imshow("Detected Grid", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
