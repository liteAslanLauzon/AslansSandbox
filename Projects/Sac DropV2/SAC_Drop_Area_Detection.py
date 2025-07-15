import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from typing import Dict, Tuple, Any, Union


from imageprocessing import (
    apply_gaussian_blur,
    compute_absolute_difference,
    apply_bounding_box_filter,
    apply_binary_threshold
)

from visiongui import show_resized_image

def cluster_binary_dbscan(
    binary_image: np.ndarray,
    eps: float = 5.0,
    min_samples: int = 5,
    downsample_cluster_factor: float = 0.5
) -> Dict[int, np.ndarray]:
    """
    Clusters non-zero pixels in a binary image using DBSCAN on a downsampled version,
    then assigns each original foreground pixel to its nearest scaled-up cluster center.

    Args:
        binary_image (np.ndarray): 2D binary image (H, W) of type uint8 with values 0 or 1.
        eps (float): DBSCAN neighborhood radius for clustering downsampled points.
        min_samples (int): Minimum samples for a cluster in DBSCAN.
        downsample_cluster_factor (float): Resize factor (0 < factor <= 1) for clustering.

    Returns:
        Dict[int, np.ndarray]: A dictionary mapping cluster_id -> array of (row, col) coordinates in the original image.

    Raises:
        TypeError: If input is not a 2D numpy array of dtype uint8.
        ValueError: If downsample_cluster_factor is not in the range (0, 1].
    """
    if not isinstance(binary_image, np.ndarray):
        raise TypeError("binary_image must be a numpy array")
    if binary_image.ndim != 2 or binary_image.dtype != np.uint8:
        raise TypeError("binary_image must be a 2D array of dtype uint8")
    if not (0 < downsample_cluster_factor <= 1.0):
        raise ValueError("downsample_cluster_factor must be in the range (0, 1]")

    # Step 1: Downsample the binary image
    downsampled_binary_image = cv2.resize(
        binary_image,
        (0, 0),
        fx=downsample_cluster_factor,
        fy=downsample_cluster_factor,
        interpolation=cv2.INTER_NEAREST
    )

    downsampled_foreground_coordinates_pixels = np.column_stack(
        np.where(downsampled_binary_image > 0)
    )

    if downsampled_foreground_coordinates_pixels.size == 0:
        return {}

    # Step 2: DBSCAN clustering on downsampled coordinates
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples, algorithm="ball_tree")
    dbscan_labels = dbscan_model.fit_predict(downsampled_foreground_coordinates_pixels)

    valid_cluster_mask = dbscan_labels != -1
    downsampled_foreground_coordinates_pixels = downsampled_foreground_coordinates_pixels[valid_cluster_mask]
    dbscan_labels = dbscan_labels[valid_cluster_mask]

    if downsampled_foreground_coordinates_pixels.size == 0:
        return {}

    # Step 3: Compute upsampled cluster centers
    upsample_scale_factor = 1.0 / downsample_cluster_factor
    cluster_ids = np.unique(dbscan_labels)

    cluster_centers_coordinates_pixels = np.array([
        np.mean(downsampled_foreground_coordinates_pixels[dbscan_labels == cluster_id], axis=0) * upsample_scale_factor
        for cluster_id in cluster_ids
    ])  # shape (k, 2)

    # Step 4: Assign each original pixel to nearest cluster center
    original_foreground_coordinates_pixels = np.column_stack(np.where(binary_image > 0))  # shape (n, 2)

    coordinate_differences = (
        original_foreground_coordinates_pixels[:, None, :] -
        cluster_centers_coordinates_pixels[None, :, :]
    )  # shape (n, k, 2)

    squared_distances = np.sum(coordinate_differences ** 2, axis=2)  # shape (n, k)
    nearest_center_indices = np.argmin(squared_distances, axis=1)  # shape (n,)

    # Step 5: Group original coordinates by cluster ID
    sorted_point_indices = np.argsort(nearest_center_indices)
    sorted_coordinates = original_foreground_coordinates_pixels[sorted_point_indices]
    sorted_cluster_labels = nearest_center_indices[sorted_point_indices]

    change_points = np.where(np.diff(sorted_cluster_labels) != 0)[0] + 1
    split_coordinates = np.split(sorted_coordinates, change_points)
    unique_cluster_ids = np.unique(sorted_cluster_labels)

    clustered_points_by_id = {
        cluster_id: coords for cluster_id, coords in zip(unique_cluster_ids, split_coordinates)
    }

    return clustered_points_by_id


def cluster_binary_findcontours(binary_image: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Clusters non-zero pixels in a binary image using OpenCV's findContours,
    then assigns each foreground pixel to its nearest contour centroid.

    Args:
        binary_image (np.ndarray): 2D binary image (H, W) of type uint8 with values 0 or 1.

    Returns:
        Dict[int, np.ndarray]: A dictionary mapping cluster_id -> array of (row, col) pixel coordinates.

    Raises:
        TypeError: If input is not a 2D numpy array of dtype uint8.
    """
    if not isinstance(binary_image, np.ndarray):
        raise TypeError("binary_image must be a numpy array")
    if binary_image.ndim != 2 or binary_image.dtype != np.uint8:
        raise TypeError("binary_image must be a 2D array of dtype uint8")

    # Step 1: Find contours
    contours_list, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )


    if not contours_list:
        return {}

    # Step 2: Compute contour centroids
    contour_centroids_pixels = []

    for contour in contours_list:
        spatial_moments = cv2.moments(contour)

        moment_area = spatial_moments["m00"]
        if moment_area == 0:
            continue  # skip degenerate contour

        centroid_x_pixels = spatial_moments["m10"] / moment_area
        centroid_y_pixels = spatial_moments["m01"] / moment_area

        contour_centroids_pixels.append((centroid_y_pixels, centroid_x_pixels))

    if not contour_centroids_pixels:
        return {}

    contour_centroids_pixels = np.array(contour_centroids_pixels)  # shape (k, 2)

    # Step 3: Assign foreground pixels to nearest centroid
    foreground_pixel_coordinates_pixels = np.column_stack(np.where(binary_image > 0))  # shape (n, 2)

    coordinate_differences = (
        foreground_pixel_coordinates_pixels[:, None, :] -
        contour_centroids_pixels[None, :, :]
    )  # shape (n, k, 2)

    squared_distances = np.sum(coordinate_differences ** 2, axis=2)  # shape (n, k)
    nearest_centroid_indices = np.argmin(squared_distances, axis=1)  # shape (n,)

    # Step 4: Group pixels by cluster ID
    sorted_indices = np.argsort(nearest_centroid_indices)
    sorted_coordinates = foreground_pixel_coordinates_pixels[sorted_indices]
    sorted_cluster_ids = nearest_centroid_indices[sorted_indices]

    change_points = np.where(np.diff(sorted_cluster_ids) != 0)[0] + 1
    split_coordinates = np.split(sorted_coordinates, change_points)
    unique_cluster_ids = np.unique(sorted_cluster_ids)

    clustered_pixels_by_id = {
        cluster_id: coords for cluster_id, coords in zip(unique_cluster_ids, split_coordinates)
    }

    return clustered_pixels_by_id

def sort_clusters_by_weighted_position(
    clusters: Dict[int, np.ndarray],
    horizontal_weight: float = 1.0,
    vertical_weight: float = 1.0
) -> Dict[int, np.ndarray]:
    """
    Sorts clusters based on a weighted combination of their average (x, y) pixel position.

    Args:
        clusters (Dict[int, np.ndarray]): Mapping of cluster_id -> (N, 2) array of (row, col) pixel coordinates.
        horizontal_weight (float): Weight for x-coordinate (left-to-right = positive).
        vertical_weight (float): Weight for y-coordinate (top-to-bottom = positive).

    Returns:
        Dict[int, np.ndarray]: A new dictionary with clusters sorted by weighted position score.
                               Keys are reassigned integers starting from 1 in sorted order.
    """
    cluster_priority_scores: list[Tuple[int, float]] = []

    for cluster_id, pixel_coordinates in clusters.items():
        average_y_position_pixels = float(np.mean(pixel_coordinates[:, 0]))  # row index
        average_x_position_pixels = float(np.mean(pixel_coordinates[:, 1]))  # col index

        sort_priority_score = (
            horizontal_weight * average_x_position_pixels +
            vertical_weight * average_y_position_pixels
        )

        cluster_priority_scores.append((cluster_id, sort_priority_score))

    cluster_priority_scores.sort(key=lambda item: item[1])

    sorted_cluster_dict = {
        new_cluster_id: clusters[original_cluster_id]
        for new_cluster_id, (original_cluster_id, _) in enumerate(cluster_priority_scores, start=1)
    }

    return sorted_cluster_dict


def overlay_clusters_on_image(
    image: np.ndarray,
    clusters: Dict[int, np.ndarray],
    alpha: float = 0.4
) -> np.ndarray:
    """
    Overlays clusters on an image using random dark colors and adds cluster IDs as red labels.

    Args:
        image (np.ndarray): Grayscale (H, W) or BGR (H, W, 3) image, dtype uint8.
        clusters (Dict[int, np.ndarray]): Mapping of cluster_id -> (N, 2) array of (row, col) pixel coordinates.
        alpha (float): Transparency factor for overlay (0 = fully transparent, 1 = fully opaque).

    Returns:
        np.ndarray: BGR image with colored overlay and cluster labels.

    Raises:
        TypeError: If the input image is not a numpy array or not uint8.
        ValueError: If alpha is not in the range [0, 1].
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if image.dtype != np.uint8:
        raise TypeError("image must have dtype uint8")
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be in the range [0, 1]")

    is_grayscale_image = image.ndim == 2
    is_single_channel_image = image.ndim == 3 and image.shape[2] == 1

    if is_grayscale_image or is_single_channel_image:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    overlay_image = np.zeros_like(image, dtype=np.uint8)
    label_centroid_coordinates_pixels = {}
    rng = np.random.default_rng(seed=42)

    for cluster_id, pixel_coordinates in clusters.items():
        if pixel_coordinates.size == 0:
            continue

        row_indices = pixel_coordinates[:, 0]
        column_indices = pixel_coordinates[:, 1]

        cluster_color_bgr = rng.integers(30, 100, size=3).tolist()
        overlay_image[row_indices, column_indices] = cluster_color_bgr

        centroid_row_col_pixels = np.mean(pixel_coordinates, axis=0).astype(int)
        label_centroid_coordinates_pixels[cluster_id] = (
            centroid_row_col_pixels[1],  # x (col)
            centroid_row_col_pixels[0]   # y (row)
        )

    blended_image = cv2.addWeighted(image, 1.0, overlay_image, alpha, 0)

    for cluster_id, (x_pixel, y_pixel) in label_centroid_coordinates_pixels.items():
        cv2.putText(
            blended_image,
            str(cluster_id),
            (x_pixel, y_pixel),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=4.6,
            color=(0, 0, 255),
            thickness=15,
            lineType=cv2.LINE_AA
        )

    return blended_image


def run_sac_drop_pipeline(
    test_image_bgr: np.ndarray,
    golden_image_bgr: np.ndarray,
    x_min_pixels: int,
    x_max_pixels: int,
    y_min_pixels: int,
    y_max_pixels: int,
    blur_kernel_size: tuple[int, int] = (17, 17),
    blur_sigma: float = 3.0,
    threshold_value: int = 20,
    alpha: float = 1.0,
    show_plot: bool = False,
    return_clusters: bool = False
) -> Union[np.ndarray, Dict[int, np.ndarray]]:
    """
    Runs the SAC drop area detection pipeline between a test and golden image.

    Args:
        test_image_bgr (np.ndarray): BGR test image as a NumPy array.
        golden_image_bgr (np.ndarray): BGR golden/reference image as a NumPy array.
        x_min_pixels (int): Minimum x-coordinate (column) in pixels of the ROI.
        x_max_pixels (int): Maximum x-coordinate (column) in pixels of the ROI.
        y_min_pixels (int): Minimum y-coordinate (row) in pixels of the ROI.
        y_max_pixels (int): Maximum y-coordinate (row) in pixels of the ROI.
        blur_kernel_size (tuple[int, int]): Kernel size in pixels for Gaussian blur.
        blur_sigma (float): Sigma value for Gaussian blur.
        threshold_value (int): Pixel intensity threshold for binarization.
        alpha (float): Overlay transparency for cluster visualization.
        show_plot (bool): If True, display the resulting image using show_resized_image.

    Returns:
        np.ndarray: Final BGR image with clusters overlaid.
    """

    if test_image_bgr is None or golden_image_bgr is None:
        raise ValueError("Input images cannot be None.")

    if test_image_bgr.ndim == 3 and test_image_bgr.shape[2] == 3:
        test_image_gray = cv2.cvtColor(test_image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        test_image_gray = test_image_bgr

    if golden_image_bgr.ndim == 3 and golden_image_bgr.shape[2] == 3:
        golden_image_gray = cv2.cvtColor(golden_image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        golden_image_gray = golden_image_bgr

    blurred_test_image = apply_gaussian_blur(
        test_image_gray, kernel_size=blur_kernel_size, sigma=blur_sigma
    )
    blurred_golden_image = apply_gaussian_blur(
        golden_image_gray, kernel_size=blur_kernel_size, sigma=blur_sigma
    )

    absolute_difference_image = compute_absolute_difference(
        blurred_test_image, blurred_golden_image
    )

    cropped_difference_image = apply_bounding_box_filter(
        absolute_difference_image,
        x_min=x_min_pixels,
        x_max=x_max_pixels,
        y_min=y_min_pixels,
        y_max=y_max_pixels
    )

    binary_mask_image = apply_binary_threshold(
        cropped_difference_image, thresh=threshold_value
    )

    contour_clusters = cluster_binary_findcontours(binary_mask_image)

    sorted_contour_clusters = sort_clusters_by_weighted_position(
        contour_clusters,
        horizontal_weight=1.0,
        vertical_weight=0.0
    )

    overlay_image = overlay_clusters_on_image(
        test_image_bgr,
        sorted_contour_clusters,
        alpha=alpha
    )

    if show_plot:
        show_resized_image(overlay_image, target_width_pixels=800)

    if return_clusters:
        return sorted_contour_clusters
    return overlay_image

def compute_cluster_properties(
    clusters: Dict[int, np.ndarray]
) -> Dict[int, Dict[str, Any]]:
    """
    Compute the centroid location and area (pixel count) for each cluster.

    Args:
        clusters (Dict[int, np.ndarray]):
            Mapping from cluster ID to an (N, 2) array of pixel coordinates (y, x).

    Returns:
        Dict[int, Dict[str, Any]]:  
            For each cluster ID, a dict with:
            - "centroid": Tuple[float, float] = (mean_y, mean_x)
            - "area": int = number of pixels in the cluster
    """
    properties: Dict[int, Dict[str, Any]] = {}

    for cluster_id, coords in clusters.items():
        # coords is an array of shape (N, 2) where each row is (y, x)
        area = coords.shape[0]
        if area > 0:
            mean_y = float(np.mean(coords[:, 0]))
            mean_x = float(np.mean(coords[:, 1]))
        else:
            # No pixels → undefined centroid
            mean_y = float('nan')
            mean_x = float('nan')

        properties[cluster_id] = {
            "centroid": (mean_y, mean_x),
            "area": area
        }

    return properties

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SAC drop area detection pipeline.")

    parser.add_argument("image_path", type=str, help="Path to the input test image.")
    parser.add_argument("golden_path", type=str, help="Path to the golden/reference image.")

    parser.add_argument("--x_min", type=int, default=1961, help="Minimum x-coordinate of ROI bounding box.")
    parser.add_argument("--x_max", type=int, default=3339, help="Maximum x-coordinate of ROI bounding box.")
    parser.add_argument("--y_min", type=int, default=1488, help="Minimum y-coordinate of ROI bounding box.")
    parser.add_argument("--y_max", type=int, default=1900, help="Maximum y-coordinate of ROI bounding box.")

    parser.add_argument(
        "--blur_kernel", type=int, nargs=2, default=[17, 17],
        help="Gaussian blur kernel size as two odd integers (e.g., 17 17)."
    )
    parser.add_argument("--blur_sigma", type=float, default=3.0, help="Standard deviation (sigma) for Gaussian blur.")
    parser.add_argument("--threshold", type=int, default=20, help="Intensity threshold value.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Transparency level for overlay (0.0–1.0).")
    parser.add_argument(
        "--show_plot",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether to show the resulting image (True/False)."
    )

    args = parser.parse_args()

    test_image = cv2.imread(args.image_path)
    golden_image = cv2.imread(args.golden_path)

    if test_image is None or golden_image is None:
        raise FileNotFoundError(f"Could not read images: {args.image_path}, {args.golden_path}")

    result = run_sac_drop_pipeline(
        test_image_bgr=test_image,
        golden_image_bgr=golden_image,
        x_min_pixels=args.x_min,
        x_max_pixels=args.x_max,
        y_min_pixels =args.y_min,
        y_max_pixels =args.y_max,
        blur_kernel_size=tuple(args.blur_kernel),
        blur_sigma=args.blur_sigma,
        threshold_value=args.threshold,
        alpha=args.alpha,
        show_plot=args.show_plot
    )

