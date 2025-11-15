"""
Preprocessing utilities for F1 Visual Difference Engine
Handles resize, denoise, lighting correction, and SIFT alignment
"""
import cv2
import numpy as np


def resize_image(image, size=(512, 512)):
    """Resize image to standard shape"""
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


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
    Returns: preprocessed_ref, preprocessed_test (both RGB and aligned)
    """
    # Step 1: Resize
    ref = resize_image(image_a)
    test = resize_image(image_b)
    
    # Step 2: Denoise
    ref = median_blur(ref)
    test = median_blur(test)
    
    # Step 3: Gamma correction
    ref = gamma_correct(ref)
    test = gamma_correct(test)
    
    # Step 4: Convert to RGB (if not already)
    if len(ref.shape) == 2:
        ref = cv2.cvtColor(ref, cv2.COLOR_GRAY2RGB)
    elif ref.shape[2] == 4:
        ref = cv2.cvtColor(ref, cv2.COLOR_BGRA2RGB)
    else:
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    
    if len(test.shape) == 2:
        test = cv2.cvtColor(test, cv2.COLOR_GRAY2RGB)
    elif test.shape[2] == 4:
        test = cv2.cvtColor(test, cv2.COLOR_BGRA2RGB)
    else:
        test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
    
    # Step 5: SIFT alignment
    test_aligned = sift_alignment(
        cv2.cvtColor(ref, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(test, cv2.COLOR_RGB2BGR)
    )
    test_aligned = cv2.cvtColor(test_aligned, cv2.COLOR_BGR2RGB)
    
    return ref, test_aligned
