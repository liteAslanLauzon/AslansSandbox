import cv2
import numpy as np

# load as before
img = cv2.imread(r"Projects\Sac DropV2\Images\croppedRotated.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)

# debug: show what we're working with
cv2.imshow("Gray blurred", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# optional: work on edges
edges = cv2.Canny(gray, 50, 150)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# tune param2 to find at least one circle
for p2 in [15, 20, 25, 30]:
    circ = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=100,
        param2=p2,
        minRadius=5,
        maxRadius=200,
    )
    print(f"param2={p2}: {'Found' if circ is not None else 'None'}")
    if circ is not None:
        circles = np.uint16(np.around(circ[0]))
        # Sort circles by accumulator value (if available), else by radius (descending)
        if circ.shape[2] > 3:
            # If accumulator is present, sort by it
            sorted_idx = np.argsort(-circ[0, :, 3])
        else:
            # Otherwise, sort by radius
            sorted_idx = np.argsort(-circ[0, :, 2])
        best_circles = circles[sorted_idx[:4]]
        for x, y, r in best_circles:
            cv2.circle(img, (x, y), r, (0, 255, 255), 2)
            cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
        cv2.imshow(f"Detected @ p2={p2}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
