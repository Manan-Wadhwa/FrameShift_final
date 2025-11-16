"""
HYBRID ANOMALY PIPELINE: PatchCore + SAM Fusion
Combines PatchCore feature-based anomaly detection with SAM segmentation refinement
"""
import torch
import cv2
import numpy as np
from utils.visualization import create_heatmap, create_overlay
from utils.sam_refine import sam_refine


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


def extract_patch_features(image, reduce_size=False):
    """
    Extract patch-level features from intermediate layers
    with optional dimensionality reduction for memory efficiency
    """
    model = get_patchcore_model()
    if model is None:
        return None
    
    device = next(model.parameters()).device
    
    # Downscale image if needed to reduce memory
    if reduce_size:
        h, w = image.shape[:2]
        if h > 224 or w > 224:
            image = cv2.resize(image, (224, 224))
    
    # Prepare image
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Normalize (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    img_tensor = (img_tensor - mean) / std
    
    features_list = []
    
    def hook_fn(module, input, output):
        # Immediately move to CPU to free GPU memory
        features_list.append(output.detach().cpu())
    
    # Register hooks for layer3 only (reduces memory vs layer2+layer3)
    handles = []
    handles.append(model.layer3.register_forward_hook(hook_fn))
    
    try:
        with torch.no_grad():
            _ = model(img_tensor)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        if not features_list:
            return None
        
        # Use single feature map instead of concatenating
        feature_map = features_list[0]
        
        # Reduce spatial dimensions if needed
        if feature_map.shape[2] > 32 or feature_map.shape[3] > 32:
            feature_map = torch.nn.functional.adaptive_avg_pool2d(feature_map, (16, 16))
        
        # Convert to numpy on CPU
        feature_map = feature_map.squeeze(0).numpy()
        
        # Clear GPU cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return feature_map
    
    except RuntimeError as e:
        print(f"Memory error during feature extraction: {e}")
        for handle in handles:
            handle.remove()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return None


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
    with memory-efficient batch processing
    """
    C, H, W = test_features.shape
    test_flat = test_features.reshape(C, -1).T  # (H*W, C)
    
    # Process in batches to avoid memory overload
    batch_size = 256
    anomaly_scores = []
    
    for i in range(0, len(test_flat), batch_size):
        batch = test_flat[i:i+batch_size]
        
        # Compute distances to memory bank (batch-wise)
        distances = np.linalg.norm(memory_bank[:, np.newaxis, :] - batch[np.newaxis, :, :], axis=2)
        
        # Average of k nearest neighbors
        k_vals = min(k, len(memory_bank))
        knn_dists = np.partition(distances, k_vals-1, axis=0)[:k_vals, :].mean(axis=0)
        anomaly_scores.extend(knn_dists)
    
    # Reshape to spatial map
    anomaly_map = np.array(anomaly_scores).reshape(H, W)
    
    # Clear memory
    del test_flat, distances
    
    return anomaly_map


def refine_anomaly_with_sam(test_img, anomaly_map, threshold=0.5):
    """
    Refine anomaly detection using SAM
    
    Steps:
    1. Threshold anomaly map to get rough anomaly regions
    2. Use SAM to refine boundaries
    3. Combine both for final mask
    """
    # Normalize and threshold anomaly map
    anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-10)
    _, rough_anomaly_mask = cv2.threshold(
        (anomaly_map_norm * 255).astype(np.uint8),
        int(threshold * 255),
        255,
        cv2.THRESH_BINARY
    )
    
    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    rough_anomaly_mask = cv2.morphologyEx(rough_anomaly_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    rough_anomaly_mask = cv2.morphologyEx(rough_anomaly_mask, cv2.MORPH_OPEN, kernel)
    
    # Use SAM to refine the mask
    refined_mask = sam_refine(test_img, rough_anomaly_mask)
    
    return refined_mask, rough_anomaly_mask, anomaly_map_norm


def run_patchcore_sam_pipeline(test_img, refined_mask=None, ref_img=None):
    """
    Complete PatchCore + SAM hybrid pipeline with memory optimization
    
    Args:
        test_img: Test image (RGB)
        refined_mask: Mask from preprocessing (optional, for reference)
        ref_img: Reference image (RGB, for building memory bank)
    
    Returns:
    {
        "heatmap": colored heatmap,
        "mask_final": thresholded mask refined by SAM,
        "overlay": overlay on test image,
        "summary": "anomaly_hybrid",
        "severity": anomaly severity score,
        "diff_map": raw anomaly map,
        "rough_mask": rough anomaly mask before SAM,
        "refined_mask_sam": mask refined by SAM
    }
    """
    global _memory_bank
    
    try:
        # Ensure all inputs have the same size
        h, w = test_img.shape[:2]
        if refined_mask is not None and refined_mask.shape[:2] != (h, w):
            print(f"Warning: Size mismatch - resizing mask from {refined_mask.shape[:2]} to {(h, w)}")
            refined_mask = cv2.resize(refined_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Extract features with memory optimization
        if ref_img is not None and _memory_bank is None:
            ref_features = extract_patch_features(ref_img, reduce_size=True)
            if ref_features is not None:
                _memory_bank = build_memory_bank(ref_features)
                del ref_features  # Free memory
        
        test_features = extract_patch_features(test_img, reduce_size=True)
        
        if test_features is None or _memory_bank is None:
            # Fallback to gradient-based anomaly detection
            print("Warning: PatchCore unavailable, using gradient-based detection")
            gray = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
            anomaly_map = cv2.Laplacian(gray, cv2.CV_64F)
            anomaly_map = np.abs(anomaly_map)
            anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-10)
            refined_mask_sam = None
            rough_mask = None
        else:
            # Compute anomaly map using PatchCore
            try:
                anomaly_map = compute_anomaly_map(test_features, _memory_bank)
            except Exception as e:
                print(f"Error in anomaly map computation: {e}")
                # Fallback
                gray = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
                anomaly_map = cv2.Laplacian(gray, cv2.CV_64F)
                anomaly_map = np.abs(anomaly_map)
                anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-10)
                refined_mask_sam = None
                rough_mask = None
            else:
                # Upsample anomaly map to match image size (feature map is smaller than image)
                if anomaly_map.shape != (h, w):
                    anomaly_map = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)
                
                # Refine with SAM
                try:
                    refined_mask_sam, rough_mask, anomaly_map = refine_anomaly_with_sam(
                        test_img, 
                        anomaly_map, 
                        threshold=0.5
                    )
                except Exception as e:
                    print(f"SAM refinement failed: {e}")
                    refined_mask_sam = None
                    rough_mask = None
        
        # Normalize anomaly map to [0, 1]
        if anomaly_map.max() > 1.0:
            anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-10)
        
        # Convert to 0-255 range
        diff_map = (anomaly_map * 255).astype(np.uint8)
        
        # Create heatmap
        heatmap = create_heatmap(diff_map, colormap='hot')
        
        # Threshold to create binary mask
        _, mask_binary = cv2.threshold(diff_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use SAM-refined mask if available, otherwise use binary mask
        if refined_mask_sam is not None:
            mask_final = refined_mask_sam
        else:
            mask_final = mask_binary
        
        # Ensure mask_final matches refined_mask if provided
        if refined_mask is not None and mask_final.shape[:2] != refined_mask.shape[:2]:
            mask_final = cv2.resize(mask_final, (refined_mask.shape[1], refined_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            # Combine with preprocessing refined mask
            mask_final = cv2.bitwise_and(mask_final, refined_mask)
        
        # Compute severity
        mask_region = (mask_final > 0).astype(np.uint8)
        severity = float(anomaly_map[mask_region > 0].mean()) if mask_region.sum() > 0 else 0.0
        
        # Create overlay
        overlay = create_overlay(test_img, mask_final, heatmap, alpha=0.4)
        
        return {
            "heatmap": heatmap,
            "mask_final": mask_final,
            "overlay": overlay,
            "summary": "anomaly_hybrid",
            "severity": severity,
            "diff_map": diff_map,
            "rough_mask": rough_mask if rough_mask is not None else mask_binary,
            "refined_mask_sam": refined_mask_sam
        }