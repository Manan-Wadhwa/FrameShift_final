"""
SAM refinement module for high-quality mask generation
"""
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


# Global SAM model (lazy loaded)
_sam_predictor = None


def get_sam_predictor():
    """Lazy load SAM predictor"""
    global _sam_predictor
    if _sam_predictor is None:
        # Using SAM ViT-H model (best quality)
        # Download checkpoint from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        
        try:
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")
            _sam_predictor = SamPredictor(sam)
        except Exception as e:
            print(f"Warning: SAM loading failed: {e}")
            print("Falling back to morphological refinement")
            _sam_predictor = None
    
    return _sam_predictor


def get_bounding_box(mask):
    """Extract bounding box from binary mask"""
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return [x_min, y_min, x_max, y_max]


def sam_refine(test_img, rough_mask):
    """
    Refine rough mask using SAM
    
    Steps:
    1. Find bounding box from rough mask
    2. Feed to SAM as prompt
    3. Get high-quality segmentation
    4. Post-process (smooth + fill holes)
    
    Returns: refined_mask
    """
    predictor = get_sam_predictor()
    
    # Fallback to morphological refinement if SAM unavailable
    if predictor is None:
        return morphological_refine(rough_mask)
    
    # Get bounding box prompt
    bbox = get_bounding_box(rough_mask)
    if bbox is None:
        return rough_mask
    
    # Convert RGB to BGR for SAM
    test_bgr = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
    
    # Set image
    predictor.set_image(test_bgr)
    
    # Predict with bounding box prompt
    masks, scores, logits = predictor.predict(
        box=np.array(bbox),
        multimask_output=True
    )
    
    # Select best mask (highest score)
    best_idx = np.argmax(scores)
    refined_mask = masks[best_idx].astype(np.uint8) * 255
    
    # Post-process: fill holes and smooth
    kernel = np.ones((3, 3), np.uint8)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Fill holes
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(refined_mask, contours, -1, 255, -1)
    
    return refined_mask


def morphological_refine(rough_mask):
    """Fallback refinement using morphological operations"""
    kernel = np.ones((7, 7), np.uint8)
    refined = cv2.morphologyEx(rough_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
    
    # Fill holes
    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(refined, contours, -1, 255, -1)
    
    # Gaussian blur for smoothing
    refined = cv2.GaussianBlur(refined, (5, 5), 0)
    _, refined = cv2.threshold(refined, 127, 255, cv2.THRESH_BINARY)
    
    return refined
