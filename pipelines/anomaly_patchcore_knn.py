"""
PatchCore KNN-based Anomaly Detection Pipeline (FrameShift v3.0 approach)
Uses DINOv2 features + KNN comparison for alignment-tolerant anomaly detection
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter
from sklearn.neighbors import NearestNeighbors


def extract_dinov2_features(image, model_name="facebook/dinov2-base"):
    """Extract DINOv2 patch features from image"""
    try:
        from transformers import AutoImageProcessor, AutoModel
        from PIL import Image
    except ImportError:
        print("Warning: transformers not installed, skipping DINOv2 extraction")
        return None
    
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        inputs = processor(img_pil, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state
        
        # Extract patch features (skip CLS token)
        patch_features = features[:, 1:, :].squeeze(0)
        return patch_features
    except Exception as e:
        print(f"Warning: DINOv2 extraction failed: {e}")
        return None


def patchcore_knn_difference(feats_ref, feats_curr, k=9):
    """
    KNN-based anomaly detection
    More robust to alignment than direct spatial comparison
    """
    patches_ref = feats_ref.cpu().numpy()
    patches_curr = feats_curr.cpu().numpy()
    
    # Train KNN on reference features
    knn = NearestNeighbors(n_neighbors=k, metric="cosine")
    knn.fit(patches_ref)
    
    # Find distances in reference space
    distances, _ = knn.kneighbors(patches_curr)
    anomaly_scores = np.mean(distances, axis=1)
    
    # Reshape to grid
    num_patches = len(anomaly_scores)
    grid_size = int(np.sqrt(num_patches))
    heatmap = anomaly_scores.reshape(grid_size, grid_size)
    
    return heatmap, anomaly_scores


def adaptive_statistical_threshold(anomaly_scores, sensitivity="medium"):
    """
    Adaptive thresholding based on percentile
    sensitivity: 'low' (98%), 'medium' (95%), 'high' (90%)
    """
    sensitivity_map = {"low": 98, "medium": 95, "high": 90}
    percentile = sensitivity_map.get(sensitivity, 95)
    threshold = np.percentile(anomaly_scores, percentile)
    
    return threshold, percentile


def refine_binary_mask(heatmap_norm, threshold, anomaly_scores):
    """
    Refine binary mask using morphological operations
    """
    # Blur for smoothing
    heatmap_blurred = cv2.GaussianBlur(heatmap_norm, (21, 21), 0)
    
    # Threshold
    binary_blurred = (heatmap_blurred > (threshold * 255 / anomaly_scores.max())).astype(np.uint8) * 255
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_clean = cv2.morphologyEx(binary_blurred, cv2.MORPH_CLOSE, kernel)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel)
    
    return binary_clean


def find_peak_prompts(heatmap_blurred, threshold, num_peaks=10):
    """
    Find peak points in heatmap for SAM prompts
    """
    footprint = np.ones((20, 20))
    local_max = maximum_filter(heatmap_blurred, footprint=footprint)
    peaks_mask = (heatmap_blurred == local_max) & (heatmap_blurred > threshold)
    
    y_coords, x_coords = np.where(peaks_mask)
    points = [[int(x), int(y)] for x, y in zip(x_coords, y_coords)]
    
    # Limit to top N peaks
    if len(points) > num_peaks:
        # Sort by heatmap value
        values = [heatmap_blurred[int(y), int(x)] for x, y in points]
        sorted_indices = np.argsort(values)[-num_peaks:]
        points = [points[i] for i in sorted_indices]
    
    return points


def assess_severity(area, total_area):
    """
    Severity assessment based on mask area
    Returns severity score 1-5
    """
    area_ratio = area / total_area
    
    if area_ratio > 0.1:  # >10%
        return 5  # Critical
    elif area_ratio > 0.05:  # >5%
        return 4  # High
    elif area_ratio > 0.02:  # >2%
        return 3  # Moderate
    elif area_ratio > 0.01:  # >1%
        return 2  # Low
    else:
        return 1  # Minor


def run_patchcore_knn_pipeline(test_img, refined_mask, ref_img=None, sensitivity="medium"):
    """
    Main pipeline: PatchCore KNN + DINOv2 + Adaptive Thresholding
    
    Args:
        test_img: Test image (preprocessed, 336x336)
        refined_mask: Refined mask from SAM
        ref_img: Reference image (optional, required for comparison)
        sensitivity: 'low', 'medium', 'high'
    
    Returns:
        dict with heatmap, mask_final, overlay, severity
    """
    
    if ref_img is None:
        print("Warning: Reference image required for PatchCore KNN pipeline")
        return None
    
    try:
        print("   [PatchCore KNN] Extracting DINOv2 features...")
        feats_ref = extract_dinov2_features(ref_img)
        feats_curr = extract_dinov2_features(test_img)
        
        if feats_ref is None or feats_curr is None:
            print("   [PatchCore KNN] Feature extraction failed")
            return None
        
        print(f"   [PatchCore KNN] Features: ref={feats_ref.shape}, curr={feats_curr.shape}")
        
        # Ensure same feature dimension
        min_patches = min(len(feats_ref), len(feats_curr))
        feats_ref = feats_ref[:min_patches]
        feats_curr = feats_curr[:min_patches]
        
        print("   [PatchCore KNN] Computing KNN difference...")
        heatmap_small, anomaly_scores = patchcore_knn_difference(
            feats_ref, feats_curr, k=9
        )
        
        # Upsample heatmap to image size
        heatmap_fullsize = cv2.resize(
            heatmap_small, 
            (test_img.shape[1], test_img.shape[0]), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize
        heatmap_norm = cv2.normalize(
            heatmap_fullsize, None, 0, 255, 
            cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        
        print("   [PatchCore KNN] Applying adaptive threshold...")
        threshold, percentile = adaptive_statistical_threshold(
            anomaly_scores, sensitivity
        )
        
        print(f"   [PatchCore KNN] Threshold={threshold:.4f} (percentile={percentile}%)")
        
        # Refine binary mask
        binary_mask = refine_binary_mask(heatmap_norm, threshold, anomaly_scores)
        
        # Calculate severity
        mask_area = np.sum(binary_mask > 0)
        total_area = test_img.shape[0] * test_img.shape[1]
        severity = assess_severity(mask_area, total_area)
        
        print(f"   [PatchCore KNN] Severity: {severity}/5")
        
        # Create heatmap visualization
        heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_HOT)
        
        # Create overlay
        overlay = test_img.copy()
        mask_overlay = binary_mask > 0
        overlay[mask_overlay] = [0, 0, 255]  # Red
        
        result = {
            "heatmap": heatmap_colored,
            "mask_final": binary_mask,
            "overlay": overlay,
            "severity": float(severity),
            "anomaly_scores": anomaly_scores.tolist() if hasattr(anomaly_scores, 'tolist') else anomaly_scores,
            "threshold": float(threshold),
            "sensitivity": sensitivity
        }
        
        print("   [PatchCore KNN] Pipeline complete")
        return result
        
    except Exception as e:
        print(f"   [PatchCore KNN] Error: {e}")
        import traceback
        traceback.print_exc()
        return None
