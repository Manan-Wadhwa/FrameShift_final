"""
SAM refinement module for high-quality mask generation
"""
import cv2
import numpy as np
import threading
import time
from segment_anything import sam_model_registry, SamPredictor


# Global SAM model (lazy loaded)
_sam_predictor = None
_sam_load_timeout = 30  # Timeout in seconds for SAM loading


def get_sam_predictor():
    """Lazy load SAM predictor with timeout"""
    global _sam_predictor
    if _sam_predictor is None:
        # Using SAM ViT-H model (best quality)
        # Download checkpoint from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        
        print("[SAM] Loading model (this may take a minute on first run)...")
        
        def load_sam():
            try:
                import os
                import torch
                
                # Check if checkpoint exists
                if not os.path.exists(sam_checkpoint):
                    print(f"[SAM] Checkpoint not found: {sam_checkpoint}")
                    print(f"[SAM] Place the SAM checkpoint in: {os.path.abspath(sam_checkpoint)}")
                    return None
                
                # Check for CUDA - try multiple methods
                print(f"[SAM] PyTorch version: {torch.__version__}")
                print(f"[SAM] torch.cuda.is_available(): {torch.cuda.is_available()}")
                print(f"[SAM] torch.cuda.device_count(): {torch.cuda.device_count()}")
                
                # Force CUDA if available
                if torch.cuda.is_available():
                    device = "cuda:0"
                    torch.cuda.set_device(0)
                    print(f"[SAM] GPU Device: {torch.cuda.get_device_name(0)}")
                    print(f"[SAM] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                else:
                    # Try using device 0 directly
                    try:
                        test_tensor = torch.zeros(1, device='cuda:0')
                        device = "cuda:0"
                        print(f"[SAM] CUDA device 0 is available!")
                    except RuntimeError:
                        device = "cpu"
                        print(f"[SAM] ⚠️ CUDA not available, falling back to CPU")
                
                print(f"[SAM] Loading model on device: {device}")
                sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                sam.to(device=device)
                
                print(f"[SAM] ✓ Model loaded successfully on {device}")
                return SamPredictor(sam)
            except Exception as e:
                print(f"[SAM] Loading failed: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        # Load with timeout
        result = [None]
        thread = threading.Thread(target=lambda: result.__setitem__(0, load_sam()), daemon=True)
        thread.start()
        thread.join(timeout=_sam_load_timeout)
        
        if thread.is_alive():
            print(f"[SAM] Loading timed out after {_sam_load_timeout}s, using fallback")
            _sam_predictor = None
        else:
            _sam_predictor = result[0]
            if _sam_predictor is None:
                print("[SAM] Falling back to morphological refinement")
    
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
    Refine rough mask using SAM with timeout protection
    
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
        print("[SAM] Using morphological fallback for mask refinement")
        return morphological_refine(rough_mask)
    
    # Get bounding box prompt
    bbox = get_bounding_box(rough_mask)
    if bbox is None:
        print("[SAM] No valid bounding box, returning original mask")
        return rough_mask
    
    try:
        # Convert RGB to BGR for SAM
        test_bgr = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
        
        # Set image with timeout
        print("[SAM] Setting image for prediction...")
        predictor.set_image(test_bgr)
        
        # Predict with bounding box prompt
        print("[SAM] Running mask prediction...")
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
        
        print("[SAM] Mask refinement successful")
        return refined_mask
    except Exception as e:
        print(f"[SAM] Prediction failed: {e}")
        print("[SAM] Using morphological fallback")
        return morphological_refine(rough_mask)


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
