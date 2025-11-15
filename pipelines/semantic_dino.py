"""
SEMANTIC PIPELINE 1: DINOv2 Difference Detection
Extracts DINO patch embeddings and computes L2 difference map
"""
import torch
import cv2
import numpy as np
from utils.visualization import create_heatmap, create_overlay


# Global DINO model (lazy loaded)
_dino_model = None


def get_dino_model():
    """Lazy load DINOv2 model"""
    global _dino_model
    if _dino_model is None:
        try:
            _dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            _dino_model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            _dino_model = _dino_model.to(device)
        except Exception as e:
            print(f"Warning: DINOv2 loading failed: {e}")
            _dino_model = None
    return _dino_model


def extract_dino_features(image):
    """
    Extract DINO patch embeddings
    Returns feature map of shape (H, W, D)
    """
    model = get_dino_model()
    if model is None:
        return None
    
    device = next(model.parameters()).device
    
    # Prepare image (DINO expects 224x224 patches)
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Normalize (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    img_tensor = (img_tensor - mean) / std
    
    with torch.no_grad():
        features = model.forward_features(img_tensor)
        patch_features = features['x_norm_patchtokens']  # (1, num_patches, dim)
    
    # Reshape to spatial grid
    num_patches = int(np.sqrt(patch_features.shape[1]))
    patch_features = patch_features.reshape(1, num_patches, num_patches, -1)
    
    # Upsample to original resolution
    patch_features = patch_features.squeeze(0).cpu().numpy()
    h, w = image.shape[:2]
    
    feature_map = cv2.resize(patch_features, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return feature_map


def compute_dino_difference(ref_features, test_features):
    """
    Compute L2 difference between feature maps
    """
    diff = np.linalg.norm(ref_features - test_features, axis=2)
    
    # Normalize
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-10)
    diff = (diff * 255).astype(np.uint8)
    
    return diff


def run_dino_pipeline(ref_img, test_img, refined_mask):
    """
    Complete DINO pipeline
    
    Returns:
    {
        "heatmap": colored heatmap,
        "mask_final": thresholded mask,
        "overlay": overlay on test image,
        "summary": "semantic",
        "diff_map": raw difference map
    }
    """
    # Extract features
    ref_features = extract_dino_features(ref_img)
    test_features = extract_dino_features(test_img)
    
    if ref_features is None or test_features is None:
        # Fallback to simple pixel difference
        print("Warning: DINO unavailable, using pixel difference")
        diff_map = np.mean(np.abs(ref_img.astype(float) - test_img.astype(float)), axis=2)
        diff_map = (diff_map / diff_map.max() * 255).astype(np.uint8)
    else:
        # Compute difference
        diff_map = compute_dino_difference(ref_features, test_features)
    
    # Create heatmap
    heatmap = create_heatmap(diff_map, colormap='jet')
    
    # Threshold to create mask
    _, mask_binary = cv2.threshold(diff_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Combine with refined mask (intersection)
    mask_final = cv2.bitwise_and(mask_binary, refined_mask)
    
    # Create overlay
    overlay = create_overlay(test_img, mask_final, heatmap, alpha=0.4)
    
    return {
        "heatmap": heatmap,
        "mask_final": mask_final,
        "overlay": overlay,
        "summary": "semantic",
        "diff_map": diff_map
    }
