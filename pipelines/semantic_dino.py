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
    
    # Reduce image size to avoid OOM (DINO with 504x504 creates too many patches)
    # Resize to 336x336 which gives manageable patch count
    h_orig, w_orig = image.shape[:2]
    target_size = 336
    if h_orig > target_size or w_orig > target_size:
        scale = target_size / max(h_orig, w_orig)
        h_new = int(h_orig * scale)
        w_new = int(w_orig * scale)
        image_resized = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    else:
        image_resized = image
        h_new, w_new = h_orig, w_orig
    
    # Prepare image (DINO expects normalized input)
    img_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Normalize (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    img_tensor = (img_tensor - mean) / std
    
    try:
        with torch.no_grad():
            # Clear cache before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            features = model.forward_features(img_tensor)
            patch_features = features['x_norm_patchtokens']  # (1, num_patches, dim)
        
        # Reshape to spatial grid
        num_patches = int(np.sqrt(patch_features.shape[1]))
        patch_features = patch_features.reshape(1, num_patches, num_patches, -1)
        
        # Upsample to original resolution
        patch_features = patch_features.squeeze(0).cpu().numpy()
        
        # Resize back to original size if we resized
        if h_new != h_orig or w_new != w_orig:
            feature_map = cv2.resize(patch_features, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        else:
            feature_map = patch_features
        
        return feature_map
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"Warning: DINO OOM - trying with even smaller size")
            # Clear cache and try with smaller size
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            # Fallback - return None to use pixel difference
            return None
        else:
            raise


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
    # Ensure all inputs have the same size
    h, w = test_img.shape[:2]
    if refined_mask.shape[:2] != (h, w):
        print(f"Warning: Size mismatch - resizing mask from {refined_mask.shape[:2]} to {(h, w)}")
        refined_mask = cv2.resize(refined_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
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
    
    # Ensure diff_map has the same size as test_img
    if diff_map.shape[:2] != (h, w):
        diff_map = cv2.resize(diff_map, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Create heatmap
    heatmap = create_heatmap(diff_map, colormap='jet')
    
    # Threshold to create mask
    _, mask_binary = cv2.threshold(diff_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Ensure mask_binary has the same size as refined_mask
    if mask_binary.shape[:2] != refined_mask.shape[:2]:
        mask_binary = cv2.resize(mask_binary, (refined_mask.shape[1], refined_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    
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
