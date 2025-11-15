"""
ANOMALY PIPELINE 3: PatchCore Detection
Nearest-neighbor distance for anomaly detection
"""
import torch
import cv2
import numpy as np
import logging
import os
from datetime import datetime
from utils.visualization import create_heatmap, create_overlay

# Setup detailed logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"patchcore_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Global PatchCore components
_patchcore_model = None
_memory_bank = None


def get_patchcore_model():
    """Lazy load feature extractor (Wide ResNet-50)"""
    global _patchcore_model
    if _patchcore_model is None:
        try:
            logger.info("Loading PatchCore model (Wide ResNet-50)...")
            import torchvision.models as models
            _patchcore_model = models.wide_resnet50_2(pretrained=True)
            _patchcore_model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            _patchcore_model = _patchcore_model.to(device)
            logger.info(f"PatchCore model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"PatchCore model loading failed: {e}", exc_info=True)
            _patchcore_model = None
    return _patchcore_model


def extract_patch_features(image):
    """
    Extract patch-level features from intermediate layers
    Ensures proper dimension handling for video frames
    """
    logger.debug("="*80)
    logger.debug("EXTRACT_PATCH_FEATURES - Starting")
    logger.debug(f"Input image type: {type(image)}")
    logger.debug(f"Input image dtype: {image.dtype if hasattr(image, 'dtype') else 'N/A'}")
    logger.debug(f"Input image shape: {image.shape}")
    logger.debug(f"Input image min/max: {image.min()}/{image.max()}")
    
    model = get_patchcore_model()
    if model is None:
        logger.error("PatchCore model is None - cannot extract features")
        return None
    
    device = next(model.parameters()).device
    logger.debug(f"Model device: {device}")
    
    # Prepare image - resize to 224x224 (video compatibility)
    import cv2
    
    # Handle grayscale or single-channel images (for video)
    original_shape = image.shape
    logger.debug(f"Original image shape before channel conversion: {original_shape}")
    
    if len(image.shape) == 2:  # (H, W) grayscale
        logger.warning(f"Converting grayscale image {image.shape} to RGB")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        logger.debug(f"After GRAY2RGB conversion: {image.shape}")
    elif image.shape[2] == 1:  # (H, W, 1)
        logger.warning(f"Converting single-channel image {image.shape} to 3-channel")
        image = np.repeat(image, 3, axis=-1)
        logger.debug(f"After repeat conversion: {image.shape}")
    elif image.shape[2] == 4:  # (H, W, 4) RGBA
        logger.warning(f"Converting RGBA image {image.shape} to RGB")
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        logger.debug(f"After BGRA2RGB conversion: {image.shape}")
    elif image.shape[2] == 3:
        logger.debug(f"Image already has 3 channels: {image.shape}")
    
    # Ensure exactly 3 channels for video model
    if image.shape[2] != 3:
        error_msg = f"DIMENSION ERROR: Expected 3 channels after conversion, got {image.shape[2]}. Original shape: {original_shape}, Current shape: {image.shape}"
        logger.error(error_msg)
        logger.error(f"Image dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")
        raise ValueError(error_msg)
    
    logger.debug(f"Pre-resize shape: {image.shape}")
    img_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    logger.debug(f"Post-resize shape: {img_resized.shape}")
    logger.debug(f"Resized dtype: {img_resized.dtype}, min: {img_resized.min()}, max: {img_resized.max()}")
    
    # Convert to float32 and normalize to [0, 1]
    img_resized = img_resized.astype(np.float32) / 255.0
    logger.debug(f"After float32 conversion - dtype: {img_resized.dtype}, min: {img_resized.min()}, max: {img_resized.max()}")
    
    # Convert to tensor: (H, W, C) -> (C, H, W)
    logger.debug(f"Pre-permute shape: {img_resized.shape}")
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1)
    logger.debug(f"Post-permute tensor shape: {img_tensor.shape}")
    
    # Ensure shape is exactly (3, 224, 224) for video
    if img_tensor.shape != (3, 224, 224):
        error_msg = f"DIMENSION ERROR: Expected tensor shape (3, 224, 224), got {img_tensor.shape}. Original input: {original_shape}"
        logger.error(error_msg)
        logger.error(f"Tensor dtype: {img_tensor.dtype}, device: {img_tensor.device}")
        logger.error(f"Permute operation: (H={img_resized.shape[0]}, W={img_resized.shape[1]}, C={img_resized.shape[2]}) -> {img_tensor.shape}")
        raise ValueError(error_msg)
    
    logger.debug(f"Pre-unsqueeze shape: {img_tensor.shape}")
    img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, 3, 224, 224)
    logger.debug(f"Post-unsqueeze shape: {img_tensor.shape}, device: {img_tensor.device}")
    
    # Normalize (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    logger.debug(f"Pre-normalization tensor stats - mean: {img_tensor.mean().item():.4f}, std: {img_tensor.std().item():.4f}")
    img_tensor = (img_tensor - mean) / std
    logger.debug(f"Post-normalization tensor stats - mean: {img_tensor.mean().item():.4f}, std: {img_tensor.std().item():.4f}")
    logger.debug(f"Final tensor shape for model: {img_tensor.shape}")
    
    features_list = []
    
    def hook_fn(module, input, output):
        logger.debug(f"Hook captured output shape: {output.shape}")
        features_list.append(output)
    
    # Register hooks for layer2 and layer3
    handles = []
    handles.append(model.layer2.register_forward_hook(hook_fn))
    handles.append(model.layer3.register_forward_hook(hook_fn))
    logger.debug("Registered hooks on layer2 and layer3")
    
    try:
        with torch.no_grad():
            logger.debug("Running forward pass...")
            _ = model(img_tensor)
            logger.debug(f"Forward pass complete. Captured {len(features_list)} feature maps")
    except Exception as e:
        logger.error(f"FORWARD PASS ERROR: {e}", exc_info=True)
        for handle in handles:
            handle.remove()
        raise
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    logger.debug("Removed hooks")
    
    # Log feature shapes
    for i, f in enumerate(features_list):
        logger.debug(f"Feature map {i} shape: {f.shape}")
    
    # Aggregate features
    logger.debug(f"Aggregating features - target size: {features_list[0].shape[2:]}")
    feature_map = torch.cat([
        torch.nn.functional.interpolate(f, size=features_list[0].shape[2:], mode='bilinear', align_corners=False)
        for f in features_list
    ], dim=1)
    logger.debug(f"Concatenated feature map shape: {feature_map.shape}")
    
    # Convert to numpy
    feature_map = feature_map.squeeze(0).cpu().numpy()
    logger.debug(f"Final feature map shape (numpy): {feature_map.shape}")
    logger.debug(f"Feature map dtype: {feature_map.dtype}, min: {feature_map.min():.4f}, max: {feature_map.max():.4f}")
    logger.debug("EXTRACT_PATCH_FEATURES - Complete")
    logger.debug("="*80)
    
    return feature_map


def build_memory_bank(ref_features):
    """
    Build memory bank from reference features (normal samples)
    For simplicity, use all patch features
    """
    logger.debug("="*80)
    logger.debug("BUILD_MEMORY_BANK - Starting")
    logger.debug(f"Input ref_features shape: {ref_features.shape}")
    logger.debug(f"Input ref_features dtype: {ref_features.dtype}")
    
    C, H, W = ref_features.shape
    logger.debug(f"Feature dimensions - C: {C}, H: {H}, W: {W}")
    
    # Reshape to (H*W, C)
    memory = ref_features.reshape(C, -1).T
    logger.debug(f"Memory bank shape after reshape: {memory.shape}")
    logger.debug(f"Expected shape: ({H*W}, {C})")
    logger.debug(f"Memory bank stats - mean: {memory.mean():.4f}, std: {memory.std():.4f}")
    logger.debug(f"Memory bank min: {memory.min():.4f}, max: {memory.max():.4f}")
    logger.debug("BUILD_MEMORY_BANK - Complete")
    logger.debug("="*80)
    
    return memory


def compute_anomaly_map(test_features, memory_bank, k=3):
    """
    Compute anomaly score using k-nearest neighbors distance
    """
    logger.debug("="*80)
    logger.debug("COMPUTE_ANOMALY_MAP - Starting")
    logger.debug(f"Input test_features shape: {test_features.shape}")
    logger.debug(f"Memory bank shape: {memory_bank.shape}")
    logger.debug(f"k-NN parameter: {k}")
    
    C, H, W = test_features.shape
    logger.debug(f"Test feature dimensions - C: {C}, H: {H}, W: {W}")
    
    test_flat = test_features.reshape(C, -1).T  # (H*W, C)
    logger.debug(f"Test features flattened shape: {test_flat.shape}")
    logger.debug(f"Expected test flat shape: ({H*W}, {C})")
    
    if test_flat.shape[1] != memory_bank.shape[1]:
        error_msg = f"DIMENSION MISMATCH: test_flat channels {test_flat.shape[1]} != memory_bank channels {memory_bank.shape[1]}"
        logger.error(error_msg)
        logger.error(f"Test features original shape: {test_features.shape}")
        logger.error(f"Memory bank shape: {memory_bank.shape}")
        raise ValueError(error_msg)
    
    # Compute distances to memory bank
    logger.debug(f"Computing distances for {len(test_flat)} test vectors against {len(memory_bank)} memory vectors")
    anomaly_scores = []
    for idx, test_vec in enumerate(test_flat):
        if idx % 500 == 0:
            logger.debug(f"Processing test vector {idx}/{len(test_flat)}")
        # L2 distance to all memory vectors
        distances = np.linalg.norm(memory_bank - test_vec, axis=1)
        # Average of k nearest neighbors
        knn_dist = np.partition(distances, min(k, len(distances)-1))[:k].mean()
        anomaly_scores.append(knn_dist)
    
    # Reshape to spatial map
    anomaly_map = np.array(anomaly_scores).reshape(H, W)
    logger.debug(f"Anomaly map shape: {anomaly_map.shape}")
    logger.debug(f"Anomaly map stats - mean: {anomaly_map.mean():.4f}, std: {anomaly_map.std():.4f}")
    logger.debug(f"Anomaly map min: {anomaly_map.min():.4f}, max: {anomaly_map.max():.4f}")
    logger.debug("COMPUTE_ANOMALY_MAP - Complete")
    logger.debug("="*80)
    
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
    
    logger.info("="*80)
    logger.info("RUN_PATCHCORE_PIPELINE - Starting")
    logger.info(f"Test image shape: {test_img.shape}, dtype: {test_img.dtype}")
    logger.info(f"Refined mask shape: {refined_mask.shape}, dtype: {refined_mask.dtype}")
    logger.info(f"Reference image: {'Provided' if ref_img is not None else 'None'}")
    if ref_img is not None:
        logger.info(f"Reference image shape: {ref_img.shape}, dtype: {ref_img.dtype}")
    logger.info(f"Memory bank status: {'Loaded' if _memory_bank is not None else 'Not loaded'}")
    
    # Ensure all inputs have the same size
    h, w = test_img.shape[:2]
    logger.debug(f"Target dimensions: h={h}, w={w}")
    
    if refined_mask.shape[:2] != (h, w):
        logger.warning(f"Size mismatch - resizing mask from {refined_mask.shape[:2]} to {(h, w)}")
        print(f"Warning: Size mismatch - resizing mask from {refined_mask.shape[:2]} to {(h, w)}")
        refined_mask = cv2.resize(refined_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        logger.debug(f"Mask resized to: {refined_mask.shape}")
    
    # Extract features
    if ref_img is not None and _memory_bank is None:
        logger.info("Building memory bank from reference image...")
        ref_features = extract_patch_features(ref_img)
        if ref_features is not None:
            _memory_bank = build_memory_bank(ref_features)
            logger.info(f"Memory bank built successfully: {_memory_bank.shape}")
        else:
            logger.error("Failed to extract reference features")
    
    logger.info("Extracting test image features...")
    test_features = extract_patch_features(test_img)
    
    if test_features is None or _memory_bank is None:
        # Fallback to gradient-based anomaly detection
        logger.warning("PatchCore unavailable, using gradient-based detection")
        print("Warning: PatchCore unavailable, using gradient-based detection")
        gray = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
        anomaly_map = cv2.Laplacian(gray, cv2.CV_64F)
        anomaly_map = np.abs(anomaly_map)
        logger.debug(f"Gradient-based anomaly map shape: {anomaly_map.shape}")
    else:
        # Compute anomaly map
        logger.info("Computing anomaly map with k-NN...")
        anomaly_map = compute_anomaly_map(test_features, _memory_bank)
    
    # Normalize
    logger.debug(f"Anomaly map before normalization - min: {anomaly_map.min():.4f}, max: {anomaly_map.max():.4f}")
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-10)
    logger.debug(f"Anomaly map after normalization - min: {anomaly_map.min():.4f}, max: {anomaly_map.max():.4f}")
    
    # Resize to image size
    logger.debug(f"Resizing anomaly map from {anomaly_map.shape} to ({w}, {h})")
    anomaly_map_resized = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)
    diff_map = (anomaly_map_resized * 255).astype(np.uint8)
    logger.debug(f"Diff map shape: {diff_map.shape}, dtype: {diff_map.dtype}")
    
    # Create heatmap
    logger.debug("Creating heatmap visualization...")
    heatmap = create_heatmap(diff_map, colormap='hot')
    logger.debug(f"Heatmap created: {heatmap.shape}")
    
    # Threshold
    logger.debug("Applying Otsu thresholding...")
    _, mask_binary = cv2.threshold(diff_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    logger.debug(f"Binary mask shape: {mask_binary.shape}, non-zero pixels: {np.count_nonzero(mask_binary)}")
    
    # Ensure mask_binary has the same size as refined_mask
    if mask_binary.shape[:2] != refined_mask.shape[:2]:
        logger.warning(f"Mask size mismatch - resizing from {mask_binary.shape[:2]} to {refined_mask.shape[:2]}")
        mask_binary = cv2.resize(mask_binary, (refined_mask.shape[1], refined_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        logger.debug(f"Mask resized to: {mask_binary.shape}")
    
    # Combine with refined mask
    logger.debug("Combining binary mask with refined mask...")
    mask_final = cv2.bitwise_and(mask_binary, refined_mask)
    logger.debug(f"Final mask shape: {mask_final.shape}, non-zero pixels: {np.count_nonzero(mask_final)}")
    
    # Compute severity
    mask_region = (mask_final > 0).astype(np.uint8)
    severity = float(anomaly_map_resized[mask_region > 0].mean()) if mask_region.sum() > 0 else 0.0
    logger.info(f"Computed severity: {severity:.4f}")
    logger.info(f"Anomaly region size: {mask_region.sum()} pixels ({100*mask_region.sum()/(h*w):.2f}%)")
    
    # Create overlay
    logger.debug("Creating overlay visualization...")
    overlay = create_overlay(test_img, mask_final, heatmap, alpha=0.4)
    logger.debug(f"Overlay created: {overlay.shape}")
    
    logger.info("RUN_PATCHCORE_PIPELINE - Complete")
    logger.info("="*80)
    
    return {
        "heatmap": heatmap,
        "mask_final": mask_final,
        "overlay": overlay,
        "summary": "anomaly",
        "severity": severity,
        "diff_map": diff_map
    }
