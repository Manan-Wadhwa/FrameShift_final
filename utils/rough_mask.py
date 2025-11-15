"""
Rough mask generation using SSIM difference
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def generate_rough_mask(ref_img, test_img):
    """
    Generate rough difference mask using SSIM
    
    Steps:
    1. Convert to grayscale
    2. Compute SSIM difference map
    3. Normalize and threshold using Otsu
    
    Returns: binary rough mask
    """
    # Convert to grayscale
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
    
    # Compute SSIM difference map
    score, ssim_map = ssim(ref_gray, test_gray, full=True)
    
    # Convert to difference map (1 - SSIM)
    diff_map = 1 - ssim_map
    
    # Normalize to 0-255
    diff_map = (diff_map * 255).astype(np.uint8)
    
    # Apply Otsu's thresholding
    _, rough_mask = cv2.threshold(diff_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    rough_mask = cv2.morphologyEx(rough_mask, cv2.MORPH_CLOSE, kernel)
    rough_mask = cv2.morphologyEx(rough_mask, cv2.MORPH_OPEN, kernel)
    
    return rough_mask, diff_map
