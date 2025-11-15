"""
ANOMALY PIPELINE 4: PaDiM Detection
Multivariate Gaussian with Mahalanobis distance
"""
import cv2
import numpy as np
import torch
from scipy.spatial.distance import mahalanobis
from utils.visualization import create_heatmap, create_overlay


# Global PaDiM components
_padim_model = None
_gaussian_params = None


def get_padim_model():
    """Lazy load feature extractor (ResNet-18)"""
    global _padim_model
    if _padim_model is None:
        try:
            import torchvision.models as models
            _padim_model = models.resnet18(pretrained=True)
            _padim_model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            _padim_model = _padim_model.to(device)
        except Exception as e:
            print(f"Warning: PaDiM model loading failed: {e}")
            _padim_model = None
    return _padim_model


def extract_padim_features(image):
    """
    Extract features from multiple layers
    """
    model = get_padim_model()
    if model is None:
        return None
    
    device = next(model.parameters()).device
    
    # Prepare image
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    img_tensor = (img_tensor - mean) / std
    
    features_list = []
    
    def hook_fn(module, input, output):
        features_list.append(output)
    
    # Register hooks for layer1, layer2, layer3
    handles = []
    handles.append(model.layer1.register_forward_hook(hook_fn))
    handles.append(model.layer2.register_forward_hook(hook_fn))
    handles.append(model.layer3.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        _ = model(img_tensor)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Aggregate features (resize all to same spatial size)
    target_size = features_list[0].shape[2:]
    feature_map = torch.cat([
        torch.nn.functional.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
        for f in features_list
    ], dim=1)
    
    feature_map = feature_map.squeeze(0).cpu().numpy()
    
    return feature_map


def fit_gaussian(ref_features):
    """
    Fit multivariate Gaussian to reference features
    Returns mean and covariance for each spatial location
    """
    C, H, W = ref_features.shape
    
    # For single reference, use identity covariance (or small diagonal)
    mean = ref_features.reshape(C, -1).T  # (H*W, C)
    
    # Simple diagonal covariance
    cov = np.eye(C) * 0.01  # Small variance
    
    return mean, cov


def compute_mahalanobis_distance(test_features, mean, cov):
    """
    Compute Mahalanobis distance anomaly map
    """
    C, H, W = test_features.shape
    test_flat = test_features.reshape(C, -1).T  # (H*W, C)
    
    # Compute Mahalanobis distance for each location
    try:
        cov_inv = np.linalg.inv(cov + np.eye(C) * 1e-6)  # Regularization
    except:
        # If singular, use Euclidean distance
        distances = np.linalg.norm(test_flat - mean, axis=1)
        return distances.reshape(H, W)
    
    distances = []
    for i, test_vec in enumerate(test_flat):
        mean_vec = mean[i % len(mean)]  # Cycle through spatial locations
        dist = mahalanobis(test_vec, mean_vec, cov_inv)
        distances.append(dist)
    
    anomaly_map = np.array(distances).reshape(H, W)
    
    return anomaly_map


def run_padim_pipeline(test_img, refined_mask, ref_img=None):
    """
    Complete PaDiM pipeline
    
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
    global _gaussian_params
    
    # Fit Gaussian on reference
    if ref_img is not None and _gaussian_params is None:
        ref_features = extract_padim_features(ref_img)
        if ref_features is not None:
            mean, cov = fit_gaussian(ref_features)
            _gaussian_params = (mean, cov)
    
    # Extract test features
    test_features = extract_padim_features(test_img)
    
    if test_features is None or _gaussian_params is None:
        # Fallback to edge-based anomaly detection
        print("Warning: PaDiM unavailable, using edge-based detection")
        gray = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        anomaly_map = cv2.GaussianBlur(edges, (15, 15), 0)
    else:
        # Compute Mahalanobis distance
        mean, cov = _gaussian_params
        anomaly_map = compute_mahalanobis_distance(test_features, mean, cov)
    
    # Normalize
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-10)
    
    # Resize to image size
    h, w = test_img.shape[:2]
    anomaly_map_resized = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)
    diff_map = (anomaly_map_resized * 255).astype(np.uint8)
    
    # Create heatmap
    heatmap = create_heatmap(diff_map, colormap='hot')
    
    # Threshold
    _, mask_binary = cv2.threshold(diff_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
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
