import cv2 as cv
import numpy as np
from time import sleep


def apply_gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    # Build a lookup table mapping pixel values [0, 255] to their adjusted gamma values
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype(
        "uint8"
    )
    return cv.LUT(image, table)


def main():
    # Load image
    image = cv.imread(
        "Projects/SacDropV3/Images/after.bmp"
    ) 
    if image is None:
        print("Failed to load image.")
        return

    # gamma_corrected = apply_gamma_correction(image, gamma=4.0)

    filtered = cv.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    gray = cv.cvtColor(filtered, cv.COLOR_BGR2GRAY)

    _, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

    display_size = (400, 400)
    image_resized = cv.resize(image, display_size)
    #gamma_corrected_resized = cv.resize(gamma_corrected, display_size)
    filtered_resized = cv.resize(filtered, display_size)
    thresh_resized = cv.resize(thresh, display_size)

    cv.imshow("Original", image_resized)
    cv.imshow("Bilateral Filtered", filtered_resized)
    cv.imshow("Thresholded", thresh_resized)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
