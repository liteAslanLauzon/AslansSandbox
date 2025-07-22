import cv2
import numpy as np
from typing import Optional
from scipy.stats import entropy


path_before = "Projects\\SacDropV3\\Images\\before.bmp"
path_after = "Projects\\SacDropV3\\Images\\after.bmp"


def load_and_crop(path: str, roi: tuple) -> Optional[np.ndarray]:

    try:
        before = cv2.imread(path_before)
        if before is None:
            raise FileNotFoundError(f"Could not load image: {path_before}")
        after = cv2.imread(path_after)
        if after is None:
            raise FileNotFoundError(f"Could not load image: {path_after}")
    except FileNotFoundError as e:
        print(e)
        return None

    x_min, x_max, y_min, y_max = roi    
    before = before[y_min:y_max, x_min:x_max]
    after = after[y_min:y_max, x_min:x_max]

    return before, after


if __name__ == "__main__":

    roi = (1930, 4730, 1486, 2184)  
    result = load_and_crop(path_before, roi)
    
    if result is not None:
        before, after = result
        print("Images loaded and cropped successfully.")
    else:
        print("Failed to load or crop images.")


    if after.shape != before.shape:
        raise ValueError("Images must have the same dimensions")

    diff = cv2.absdiff(before, after)

    cv2.imshow("Difference", diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
