import cv2
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
import matplotlib.pyplot as plt

import numpy as np
import cv2
import matplotlib.pyplot as plt


def local_entropy_integral(gray, radius, n_bins=16):
    # 1) quantize to [0…n_bins-1]
    q = gray.astype(int) * n_bins // 256
    q[q == n_bins] = n_bins - 1
    H, W = gray.shape
    w = 2 * radius + 1

    # 2) build integral histogram: I[y,x,b] = # of pixels ≤(y,x) in bin b
    #    shape (H+1, W+1, n_bins)
    I = np.zeros((H + 1, W + 1, n_bins), dtype=np.int32)
    for b in range(n_bins):
        mask = (q == b).astype(np.int32)
        I[..., b] = cv2.integral(mask)[1:, 1:]  # cv2.integral → shape (H+1, W+1)

    # 3) for each bin, histogram in window = Br + Tl – Bl – Tr
    #    where Br = I[w:, w:], Tl = I[:-w, :-w], etc.
    Br = I[w:, w:]  # bottom-right corner sums
    Tl = I[:-w, :-w]
    Bl = I[w:, :-w]
    Tr = I[:-w, w:]
    counts = Br + Tl - Bl - Tr  # shape (H+1-w, W+1-w, n_bins)

    # 4) compute entropy only on the “interior” (drop a radius border)
    area = w * w
    p = counts / area
    ent = -np.sum(p * np.log2(p + 1e-12), axis=2)

    # 5) pad back to full size
    E = np.zeros_like(gray, dtype=float)
    E[radius:-radius, radius:-radius] = ent
    return E


# usage
gray = cv2.imread("Projects\\SacDropV3\\Images\\after.bmp", cv2.IMREAD_GRAYSCALE)
E = local_entropy_integral(gray, radius=5, n_bins=16)

# normalize & show
E_norm = ((E - E.min()) / E.ptp() * 255).astype(np.uint8)
plt.imshow(E_norm, cmap="inferno")
plt.axis("off")
plt.show()
