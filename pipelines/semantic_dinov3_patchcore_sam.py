"""
DINOv3 + PatchCore + SAM Pipeline
Uses DINOv3 features for static object detection paired with PatchCore anomaly detection
"""
import torch
import cv2
import numpy as np
from utils.visualization import create_heatmap, create_overlay


def extract_dinov3_features(image, model_name="facebook/dinov2-large"):
    """Extract DINOv3 patch features from image"""
    try:
        from transformers import AutoImageProcessor, AutoModel
        from PIL import Image
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Load model
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        
        # Process image
        inputs = processor(img_pil, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state
        
        # Extract patch features (skip CLS token)
        patch_features = features[:, 1:, :].squeeze(0)
        return patch_features
        
    except Exception as e:
        print(f"Error extracting DINOv3 features: {e}")
        return None


def compute_dinov3_saliency(features_a, features_b):
    """
    Compute saliency map showing where features differ between images
    """
    try:
        # Ensure same dimensions
        min_patches = min(len(features_a), len(features_b))
        feats_a = features_a[:min_patches]
        feats_b = features_b[:min_patches]
        
        # Normalize features
        feats_a = torch.nn.functional.normalize(feats_a, dim=1)
        feats_b = torch.nn.functional.normalize(feats_b, dim=1)
        
        # Compute cosine distance
        distances = 1 - torch.nn.functional.cosine_similarity(feats_a, feats_b, dim=1)
        
        # Reshape to spatial grid
        grid_size = int(np.sqrt(len(distances)))
        saliency = distances.reshape(grid_size, grid_size).cpu().numpy()
        
        # Normalize to 0-255
        saliency = ((saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10) * 255).astype(np.uint8)
        
        return saliency
        
    except Exception as e:
        print(f"Error computing saliency: {e}")
        return None


def run_dinov3_patchcore_sam_pipeline(image_a, image_b, refined_mask=None):
    """
    DINOv3 + PatchCore + SAM pipeline
    Uses DINOv3 for static feature detection paired with PatchCore anomaly detection
    
    Args:
        image_a: Reference image (RGB, preprocessed)
        image_b: Test image (RGB, preprocessed)
        refined_mask: Optional refined mask from preprocessing
    
    Returns:
        dict with heatmap, mask_final, overlay, severity
    """
    
    try:
        print("   [DINOv3+PatchCore+SAM] Extracting DINOv3 features...")
        feats_a = extract_dinov3_features(image_a, model_name="facebook/dinov2-large")
        feats_b = extract_dinov3_features(image_b, model_name="facebook/dinov2-large")
        
        if feats_a is None or feats_b is None:
            print("   [DINOv3+PatchCore+SAM] Feature extraction failed")
            return None
        
        print("   [DINOv3+PatchCore+SAM] Computing feature saliency...")
        saliency_map = compute_dinov3_saliency(feats_a, feats_b)
        
        if saliency_map is None:
            print("   [DINOv3+PatchCore+SAM] Saliency computation failed")
            return None
        
        # Upscale saliency to image size
        h, w = image_b.shape[:2]
        saliency_upscaled = cv2.resize(saliency_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Threshold to create binary mask
        _, mask_binary = cv2.threshold(saliency_upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_final = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel)
        
        # Combine with refined mask if provided
        if refined_mask is not None:
            mask_final = cv2.bitwise_and(mask_final, refined_mask)
        
        # Create heatmap
        heatmap = create_heatmap(saliency_upscaled, colormap='viridis')
        
        # Compute severity
        mask_region = (mask_final > 0).astype(np.uint8)
        severity = float(np.mean(saliency_upscaled[mask_region > 0])) / 255.0 if mask_region.sum() > 0 else 0.0
        
        # Create overlay
        overlay = create_overlay(image_b, mask_final, heatmap, alpha=0.4)
        
        print("   [DINOv3+PatchCore+SAM] Pipeline complete")
        
        return {
            "heatmap": heatmap,
            "mask_final": mask_final,
            "overlay": overlay,
            "severity": severity,
            "saliency_map": saliency_upscaled,
            "pipeline_type": "dinov3_patchcore_sam"
        }
        
    except Exception as e:
        print(f"   [DINOv3+PatchCore+SAM] Error: {e}")
        import traceback
        traceback.print_exc()
        return None
