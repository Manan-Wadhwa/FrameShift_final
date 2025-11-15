"""
Routing features computation for semantic vs anomaly classification
"""
import cv2
import numpy as np
from scipy.stats import entropy


def compute_texture_complexity(region):
    """
    Compute Laplacian variance as texture complexity measure
    """
    if len(region.shape) == 3:
        region_gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    else:
        region_gray = region
    
    laplacian = cv2.Laplacian(region_gray, cv2.CV_64F)
    texture_var = laplacian.var()
    
    return texture_var


def compute_edge_density(region, mask_area):
    """
    Compute edge density using Canny edge detection
    """
    if len(region.shape) == 3:
        region_gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    else:
        region_gray = region
    
    edges = cv2.Canny(region_gray, 50, 150)
    edge_pixels = np.sum(edges > 0)
    
    edge_density = edge_pixels / mask_area if mask_area > 0 else 0
    
    return edge_density


def compute_region_entropy(region):
    """
    Compute Shannon entropy of the region
    """
    if len(region.shape) == 3:
        region_gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    else:
        region_gray = region
    
    hist, _ = np.histogram(region_gray.flatten(), bins=256, range=(0, 256))
    hist = hist / hist.sum()  # Normalize
    
    region_entropy = entropy(hist + 1e-10)  # Add small epsilon to avoid log(0)
    
    return region_entropy


def compute_color_shift(ref_region, test_region):
    """
    Compute HSV histogram difference between reference and test regions
    """
    # Convert to HSV
    ref_hsv = cv2.cvtColor(ref_region, cv2.COLOR_RGB2HSV)
    test_hsv = cv2.cvtColor(test_region, cv2.COLOR_RGB2HSV)
    
    # Compute histograms for each channel
    hist_ref_h = cv2.calcHist([ref_hsv], [0], None, [180], [0, 180])
    hist_ref_s = cv2.calcHist([ref_hsv], [1], None, [256], [0, 256])
    hist_ref_v = cv2.calcHist([ref_hsv], [2], None, [256], [0, 256])
    
    hist_test_h = cv2.calcHist([test_hsv], [0], None, [180], [0, 180])
    hist_test_s = cv2.calcHist([test_hsv], [1], None, [256], [0, 256])
    hist_test_v = cv2.calcHist([test_hsv], [2], None, [256], [0, 256])
    
    # Normalize
    hist_ref_h = hist_ref_h / (hist_ref_h.sum() + 1e-10)
    hist_ref_s = hist_ref_s / (hist_ref_s.sum() + 1e-10)
    hist_ref_v = hist_ref_v / (hist_ref_v.sum() + 1e-10)
    
    hist_test_h = hist_test_h / (hist_test_h.sum() + 1e-10)
    hist_test_s = hist_test_s / (hist_test_s.sum() + 1e-10)
    hist_test_v = hist_test_v / (hist_test_v.sum() + 1e-10)
    
    # Compute histogram intersection (1 - intersection = shift)
    intersect_h = np.minimum(hist_ref_h, hist_test_h).sum()
    intersect_s = np.minimum(hist_ref_s, hist_test_s).sum()
    intersect_v = np.minimum(hist_ref_v, hist_test_v).sum()
    
    color_shift = 1 - (intersect_h + intersect_s + intersect_v) / 3
    
    return color_shift


def compute_routing_features(ref_img, test_img, refined_mask):
    """
    Compute all routing features
    
    Returns: dict with all features
    """
    # Extract regions
    mask_binary = (refined_mask > 0).astype(np.uint8)
    mask_area = np.sum(mask_binary)
    
    if mask_area == 0:
        return {
            "texture_var": 0.0,
            "edge_density": 0.0,
            "entropy": 0.0,
            "color_shift": 0.0
        }
    
    # Get bounding box for efficiency
    coords = np.argwhere(mask_binary > 0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Extract regions
    ref_region = ref_img[y_min:y_max+1, x_min:x_max+1]
    test_region = test_img[y_min:y_max+1, x_min:x_max+1]
    region_mask = mask_binary[y_min:y_max+1, x_min:x_max+1]
    
    # Apply mask to regions
    ref_region_masked = ref_region * region_mask[:, :, np.newaxis]
    test_region_masked = test_region * region_mask[:, :, np.newaxis]
    
    # Compute features
    texture_var = compute_texture_complexity(test_region_masked)
    edge_density = compute_edge_density(test_region_masked, mask_area)
    region_entropy = compute_region_entropy(test_region_masked)
    color_shift = compute_color_shift(ref_region_masked, test_region_masked)
    
    return {
        "texture_var": float(texture_var),
        "edge_density": float(edge_density),
        "entropy": float(region_entropy),
        "color_shift": float(color_shift)
    }


def route_prediction(features, T_texture=500, T_edge=0.3, T_color=0.15):
    """
    Predict whether task is semantic or anomaly based on routing features
    
    Thresholds (tunable):
    - T_texture: 500 (Laplacian variance threshold)
    - T_edge: 0.3 (edge density threshold)
    - T_color: 0.15 (color shift threshold)
    """
    texture_var = features["texture_var"]
    edge_density = features["edge_density"]
    color_shift = features["color_shift"]
    
    # Decision logic
    if texture_var > T_texture or edge_density > T_edge:
        predicted_type = "anomaly"
        confidence = "high" if texture_var > T_texture * 1.5 else "medium"
    elif color_shift > T_color:
        predicted_type = "semantic"
        confidence = "high" if color_shift > T_color * 1.5 else "medium"
    else:
        predicted_type = "semantic"
        confidence = "low"
    
    return predicted_type, confidence
