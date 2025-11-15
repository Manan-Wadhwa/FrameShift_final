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


def prepare_image_for_video(image, target_size=(224, 224)):
    """
    Prepare image tensor for video processing
    Ensures output shape is (3, 224, 224) for video models
    
    Args:
        image: Input image (H, W, C) in RGB format
        target_size: Target spatial dimensions (default 224x224)
    
    Returns:
        tensor: Image tensor with shape (3, H, W)
    """
    # Resize to target size if needed
    if image.shape[:2] != target_size:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Ensure RGB format (H, W, 3)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    
    # Convert to float and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Transpose to (3, H, W) for video models
    image_tensor = np.transpose(image, (2, 0, 1))
    
    return image_tensor


def create_video_frame_batch(images, target_size=(224, 224)):
    """
    Prepare a batch of images for video processing
    
    Args:
        images: List of images in (H, W, C) format
        target_size: Target spatial dimensions
    
    Returns:
        batch: Numpy array with shape (N, 3, H, W)
    """
    batch = []
    for img in images:
        tensor = prepare_image_for_video(img, target_size)
        batch.append(tensor)
    
    return np.stack(batch, axis=0)
