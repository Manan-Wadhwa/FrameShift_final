"""
Preprocessing utilities for F1 Visual Difference Engine
Handles resize, denoise, lighting correction, SIFT alignment, and background removal
"""
import cv2
import numpy as np
from PIL import Image


def resize_image(image, size=(512, 512)):
    """Resize image to standard shape using center crop"""
    return resize_with_center_crop(image, target_size=size)


def resize_with_center_crop(image, target_size=(512, 512)):
    """
    Resize image using center crop (aspect ratio preserving)
    
    Steps:
    1. Scale image to fit target size while preserving aspect ratio
    2. Center crop to exact target size
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling factor to fit the image in the target while preserving aspect ratio
    scale = max(target_h / h, target_w / w)
    
    # Scale image
    scaled_h = int(h * scale)
    scaled_w = int(w * scale)
    scaled = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    
    # Center crop to target size
    start_h = (scaled_h - target_h) // 2
    start_w = (scaled_w - target_w) // 2
    
    cropped = scaled[start_h:start_h + target_h, start_w:start_w + target_w]
    
    return cropped


def remove_background_rembg(image):
    """
    Remove background using rembg
    Input: BGR image
    Output: RGBA image with transparent background
    """
    try:
        import rembg
        
        # Convert BGR to RGB for rembg
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Remove background
        output = rembg.remove(img_pil)
        
        # Convert back to numpy array (RGBA)
        output_np = np.array(output)
        
        return output_np
    except ImportError:
        print("Warning: rembg not installed, skipping background removal")
        return None
    except Exception as e:
        print(f"Warning: Background removal failed: {e}")
        return None


def apply_transparent_bg(image_rgba):
    """
    Apply transparent background as white for further processing
    Input: RGBA image
    Output: RGB image with white background where transparent
    """
    if image_rgba is None or image_rgba.shape[2] != 4:
        return image_rgba
    
    # Extract RGB and alpha channel
    rgb = image_rgba[:, :, :3]
    alpha = image_rgba[:, :, 3:4] / 255.0
    
    # Create white background
    white_bg = np.ones_like(rgb) * 255
    
    # Blend using alpha
    result = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
    
    return result


def median_blur(image, k=3):
    """Apply median blur for denoising"""
    return cv2.medianBlur(image, k)


def gamma_correct(image, gamma=1.2):
    """Apply gamma correction for lighting normalization"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def sift_alignment(ref_img, test_img):
    """
    Align test image to reference using SIFT + Homography
    Returns aligned test image
    """
    # Convert to grayscale for feature detection
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(ref_gray, None)
    kp2, des2 = sift.detectAndCompute(test_gray, None)
    
    if des1 is None or des2 is None:
        print("Warning: No features detected, returning original test image")
        return test_img
    
    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 4:
        print("Warning: Not enough matches for homography, returning original")
        return test_img
    
    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        print("Warning: Homography estimation failed, returning original")
        return test_img
    
    # Warp test image to align with reference
    h, w = ref_img.shape[:2]
    aligned = cv2.warpPerspective(test_img, H, (w, h))
    
    return aligned


def preprocess(image_a, image_b):
    """
    Complete preprocessing pipeline
    Returns: preprocessed_ref, preprocessed_test (both RGB)
    
    Steps:
    1. Resize using center crop to 336x336
    2. Remove background using rembg
    3. Denoise with median blur
    4. Gamma correction
    5. Convert to RGB
    """
    # Step 1: Resize using center crop
    target_size = 336  # Divisible by 14 for transformer compatibility
    ref = resize_with_center_crop(image_a, target_size=(target_size, target_size))
    test = resize_with_center_crop(image_b, target_size=(target_size, target_size))
    
    # Step 2: Background removal using rembg
    print("   [Preprocess] Removing backgrounds using rembg...")
    ref_no_bg = remove_background_rembg(ref)
    test_no_bg = remove_background_rembg(test)
    
    # If background removal succeeded, apply white background; otherwise continue with original
    if ref_no_bg is not None:
        ref = apply_transparent_bg(ref_no_bg)
    if test_no_bg is not None:
        test = apply_transparent_bg(test_no_bg)
    
    # Step 3: Denoise
    ref = median_blur(ref)
    test = median_blur(test)
    
    # Step 4: Gamma correction
    ref = gamma_correct(ref)
    test = gamma_correct(test)
    
    # Step 5: Convert to RGB (if not already)
    if len(ref.shape) == 2:
        ref = cv2.cvtColor(ref, cv2.COLOR_GRAY2RGB)
    elif ref.shape[2] == 4:
        ref = cv2.cvtColor(ref, cv2.COLOR_BGRA2RGB)
    elif ref.shape[2] == 3:
        # Check if it's BGR
        try:
            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
        except:
            pass  # Already RGB
    
    if len(test.shape) == 2:
        test = cv2.cvtColor(test, cv2.COLOR_GRAY2RGB)
    elif test.shape[2] == 4:
        test = cv2.cvtColor(test, cv2.COLOR_BGRA2RGB)
    elif test.shape[2] == 3:
        # Check if it's BGR
        try:
            test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
        except:
            pass  # Already RGB
    
    return ref, test
