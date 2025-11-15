"""
SEMANTIC PIPELINE 2: CLIP Patch Difference Detection
Sliding window CLIP embeddings with cosine similarity
"""
import torch
import cv2
import numpy as np
from PIL import Image
from utils.visualization import create_heatmap, create_overlay


# Global CLIP model (lazy loaded)
_clip_model = None
_clip_preprocess = None


def get_clip_model():
    """Lazy load CLIP model"""
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        try:
            import clip
            _clip_model, _clip_preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
            _clip_model.eval()
        except Exception as e:
            print(f"Warning: CLIP loading failed: {e}")
            _clip_model = None
            _clip_preprocess = None
    return _clip_model, _clip_preprocess


def extract_clip_patch_features(image, window_size=64, stride=32):
    """
    Extract CLIP embeddings for sliding window patches
    Returns heatmap of cosine differences
    """
    model, preprocess = get_clip_model()
    if model is None:
        return None
    
    device = next(model.parameters()).device
    h, w = image.shape[:2]
    
    # Initialize feature map
    feature_map = np.zeros((h // stride, w // stride, 512))
    
    y_idx = 0
    for y in range(0, h - window_size, stride):
        x_idx = 0
        for x in range(0, w - window_size, stride):
            # Extract patch
            patch = image[y:y+window_size, x:x+window_size]
            
            # Convert to PIL and preprocess
            patch_pil = Image.fromarray(patch)
            patch_tensor = preprocess(patch_pil).unsqueeze(0).to(device)
            
            try:
                # Extract features
                with torch.no_grad():
                    # Clear cache before each patch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    features = model.encode_image(patch_tensor)
                    features = features / features.norm(dim=-1, keepdim=True)  # Normalize
                
                feature_map[y_idx, x_idx] = features.cpu().numpy()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Warning: CLIP OOM at patch ({y_idx}, {x_idx}), skipping")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    feature_map[y_idx, x_idx] = np.zeros(512)
                else:
                    raise
            
            x_idx += 1
        y_idx += 1
    
    return feature_map


def compute_clip_difference(ref_features, test_features):
    """
    Compute cosine difference heatmap
    """
    # Cosine similarity = dot product of normalized vectors
    similarity = np.sum(ref_features * test_features, axis=2)
    
    # Convert to distance (1 - similarity)
    diff = 1 - similarity
    
    # Normalize
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-10)
    
    return diff


def run_clip_pipeline(ref_img, test_img, refined_mask):
    """
    Complete CLIP pipeline
    
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
    
    # Extract patch features
    ref_features = extract_clip_patch_features(ref_img)
    test_features = extract_clip_patch_features(test_img)
    
    if ref_features is None or test_features is None:
        # Fallback to color histogram difference
        print("Warning: CLIP unavailable, using color difference")
        ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_RGB2HSV)
        test_hsv = cv2.cvtColor(test_img, cv2.COLOR_RGB2HSV)
        diff = np.mean(np.abs(ref_hsv.astype(float) - test_hsv.astype(float)), axis=2)
        diff = diff / diff.max()
    else:
        # Compute difference
        diff = compute_clip_difference(ref_features, test_features)
    
    # Resize to original image size
    diff_resized = cv2.resize(diff, (w, h), interpolation=cv2.INTER_LINEAR)
    diff_map = (diff_resized * 255).astype(np.uint8)
    
    # Create heatmap
    heatmap = create_heatmap(diff_map, colormap='jet')
    
    # Threshold
    _, mask_binary = cv2.threshold(diff_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Ensure mask_binary has the same size as refined_mask
    if mask_binary.shape[:2] != refined_mask.shape[:2]:
        mask_binary = cv2.resize(mask_binary, (refined_mask.shape[1], refined_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Combine with refined mask
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
