"""
ANOMALY PIPELINE 3: PatchCore Detection
Nearest-neighbor distance for anomaly detection
"""
import torch
import cv2
import numpy as np
from utils.visualization import create_heatmap, create_overlay


# Global PatchCore components
_patchcore_model = None
_memory_bank = None


def get_patchcore_model():
    """Lazy load feature extractor (Wide ResNet-50)"""
    global _patchcore_model
    if _patchcore_model is None:
        try:
            import torchvision.models as models
            _patchcore_model = models.wide_resnet50_2(pretrained=True)
            _patchcore_model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            _patchcore_model = _patchcore_model.to(device)
        except Exception as e:
            print(f"Warning: PatchCore model loading failed: {e}")
            _patchcore_model = None
    return _patchcore_model


def extract_patch_features(image):
    """
    Extract patch-level features from intermediate layers
    """
    model = get_patchcore_model()
    if model is None:
        return None
    
    device = next(model.parameters()).device
    
    # Prepare image - resize to 224x224
    import cv2
    img_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    # Convert to tensor: (H, W, C) -> (C, H, W)
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    
    # Ensure shape is exactly (3, 224, 224)
    if img_tensor.shape != (3, 224, 224):
        raise ValueError(f"Expected shape (3, 224, 224), got {img_tensor.shape}")
    
    img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, 3, 224, 224)
    
    # Normalize (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    img_tensor = (img_tensor - mean) / std
    
    features_list = []
    
    def hook_fn(module, input, output):
        features_list.append(output)
    
    # Register hooks for layer2 and layer3
    handles = []
    handles.append(model.layer2.register_forward_hook(hook_fn))
    handles.append(model.layer3.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        _ = model(img_tensor)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Aggregate features
    feature_map = torch.cat([
        torch.nn.functional.interpolate(f, size=features_list[0].shape[2:], mode='bilinear', align_corners=False)
        for f in features_list
    ], dim=1)
    
    # Convert to numpy
    feature_map = feature_map.squeeze(0).cpu().numpy()
    
    return feature_map


def build_memory_bank(ref_features):
    """
    Build memory bank from reference features (normal samples)
    For simplicity, use all patch features
    """
    C, H, W = ref_features.shape
    # Reshape to (H*W, C)
    memory = ref_features.reshape(C, -1).T
    return memory


def compute_anomaly_map(test_features, memory_bank, k=3):
    """
    Compute anomaly score using k-nearest neighbors distance
    """
    C, H, W = test_features.shape
    test_flat = test_features.reshape(C, -1).T  # (H*W, C)
    
    # Compute distances to memory bank
    anomaly_scores = []
    for test_vec in test_flat:
        # L2 distance to all memory vectors
        distances = np.linalg.norm(memory_bank - test_vec, axis=1)
        # Average of k nearest neighbors
        knn_dist = np.partition(distances, k)[:k].mean()
        anomaly_scores.append(knn_dist)
    
    # Reshape to spatial map
    anomaly_map = np.array(anomaly_scores).reshape(H, W)
    
    return anomaly_map


def run_patchcore_pipeline(test_img, refined_mask, ref_img=None):
    """
    Complete PatchCore pipeline
    
    Returns:
    {
        "heatmap": colored heatmap,
        "mask_final": thresholded mask,
        "overlay": overlay on test image,
        "summary": "anomaly",
        "severity": anomaly severity score,
        "diff_map": raw anomaly map
    }
    """
    global _memory_bank
    
    # Ensure all inputs have the same size
    h, w = test_img.shape[:2]
    if refined_mask.shape[:2] != (h, w):
        print(f"Warning: Size mismatch - resizing mask from {refined_mask.shape[:2]} to {(h, w)}")
        refined_mask = cv2.resize(refined_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Extract features
    if ref_img is not None and _memory_bank is None:
        ref_features = extract_patch_features(ref_img)
        if ref_features is not None:
            _memory_bank = build_memory_bank(ref_features)
    
    test_features = extract_patch_features(test_img)
    
    if test_features is None or _memory_bank is None:
        # Fallback to gradient-based anomaly detection
        print("Warning: PatchCore unavailable, using gradient-based detection")
        gray = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
        anomaly_map = cv2.Laplacian(gray, cv2.CV_64F)
        anomaly_map = np.abs(anomaly_map)
    else:
        # Compute anomaly map
        anomaly_map = compute_anomaly_map(test_features, _memory_bank)
    
    # Normalize
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-10)
    
    # Resize to image size
    anomaly_map_resized = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)
    diff_map = (anomaly_map_resized * 255).astype(np.uint8)
    
    # Create heatmap
    heatmap = create_heatmap(diff_map, colormap='hot')
    
    # Threshold
    _, mask_binary = cv2.threshold(diff_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Ensure mask_binary has the same size as refined_mask
    if mask_binary.shape[:2] != refined_mask.shape[:2]:
        mask_binary = cv2.resize(mask_binary, (refined_mask.shape[1], refined_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Combine with refined mask
    mask_final = cv2.bitwise_and(mask_binary, refined_mask)
    
    # Compute severity
    mask_region = (mask_final > 0).astype(np.uint8)
    severity = float(anomaly_map_resized[mask_region > 0].mean()) if mask_region.sum() > 0 else 0.0
    
    # Create overlay
    overlay = create_overlay(test_img, mask_final, heatmap, alpha=0.4)
    
    return {
        "heatmap": heatmap,
        "mask_final": mask_final,
        "overlay": overlay,
        "summary": "anomaly",
        "severity": severity,
        "diff_map": diff_map
    }
