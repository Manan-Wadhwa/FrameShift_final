"""
Visualization utilities for heatmaps, masks, and overlays
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def create_heatmap(diff_map, colormap='jet'):
    """
    Create colored heatmap from difference map
    """
    # Normalize to 0-1
    normalized = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-10)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(normalized)[:, :, :3]  # Remove alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)
    
    return heatmap


def create_overlay(image, mask, heatmap=None, alpha=0.5):
    """
    Create overlay of image with mask and/or heatmap
    """
    overlay = image.copy()
    
    if heatmap is not None:
        # Blend heatmap
        overlay = cv2.addWeighted(overlay, 1 - alpha, heatmap, alpha, 0)
    
    # Add mask contour
    if mask is not None:
        mask_binary = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    
    return overlay


def visualize_mask(mask):
    """
    Visualize binary mask as RGB image
    """
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    mask_rgb[mask > 0] = [255, 255, 255]
    return mask_rgb
