import cv2
import numpy as np


def contour_circle_match(
    img, min_radius=10, max_radius=100, min_circularity=0.0, top_n=4
):
    """
    Finds the top-N most-circle-like contours in a binary image.
    Args:
      img              : single-channel binary (0/255) or grayscale image
      min_radius       : ignore circles smaller than this
      max_radius       : ignore circles larger than this
      min_circularity  : drop contours with circularity below this
      top_n            : number of best circles to return
    Returns:
      List of (r, (cx,cy), circularity), sorted by circularity descending
    """

    # binarize
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # find all contours
    cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    circles = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area <= 0:
            continue
        peri = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (peri * peri)
        if circ < min_circularity:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if not (min_radius <= r <= max_radius):
            continue
        circles.append((int(round(r)), (int(round(x)), int(round(y))), circ))

    # Sort by circularity descending, take top_n
    return sorted(circles, key=lambda x: x[2], reverse=True)[:top_n]


if __name__ == "__main__":
    img = cv2.imread(
        r"Projects\\Sac DropV2\\Images\\croppedV2.bmp",  # make sure full filename
        cv2.IMREAD_GRAYSCALE,
    )
    if img is None:
        raise FileNotFoundError("Failed to load image—check path!")

    results = contour_circle_match(
        img, min_radius=10, max_radius=200, min_circularity=0.0, top_n=4
    )

    if not results:
        print("No sufficiently circular contours found.")
    else:
        # convert to BGR so we can draw colored outlines/text
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # define a distinct color for each of the top 4
        colors = [
            (0, 0, 255),  # red for #1
            (0, 255, 0),  # green for #2
            (255, 0, 0),  # blue for #3
            (0, 255, 255),  # yellow for #4
        ]

        for idx, (r, (cx, cy), circ) in enumerate(results):
            color = colors[idx % len(colors)]
            thickness = 2 + (idx == 0)  # slightly thicker for the top one

            # draw the enclosing circle
            cv2.circle(out, (cx, cy), r, color, thickness)

            # mark the center
            cv2.circle(out, (cx, cy), 3, color, -1)

            # put the rank number just outside the circle
            label = f"#{idx+1}"
            text_pos = (cx + r + 5, cy)
            cv2.putText(
                out,
                label,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

            print(f"Circle {idx+1}: center=({cx},{cy}), r={r}, circ={circ:.3f}")

        cv2.imshow("Top 4 Circular Contours", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
