"""
Advanced Image Alignment Methods
Multiple techniques for aligning test images to reference frames
"""
import cv2
import numpy as np


def ecc_alignment(ref_img, test_img, warp_mode=cv2.MOTION_EUCLIDEAN, max_iterations=5000):
    """
    Enhanced Correlation Coefficient (ECC) alignment
    
    Best for: Video frames, subtle transformations, camera stabilization
    Speed: Fast
    Handles: Translation, rotation, scale (depending on warp_mode)
    
    Args:
        warp_mode: cv2.MOTION_TRANSLATION, MOTION_EUCLIDEAN, MOTION_AFFINE, MOTION_HOMOGRAPHY
    """
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if len(ref_img.shape) == 3 else ref_img
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY) if len(test_img.shape) == 3 else test_img
    
    # Initialize warp matrix
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, 1e-6)
    
    try:
        # Run ECC
        cc, warp_matrix = cv2.findTransformECC(ref_gray, test_gray, warp_matrix, warp_mode, criteria)
        
        # Apply transformation
        h, w = ref_img.shape[:2]
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            aligned = cv2.warpPerspective(test_img, warp_matrix, (w, h), 
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            aligned = cv2.warpAffine(test_img, warp_matrix, (w, h),
                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        
        return aligned, warp_matrix, cc
    except cv2.error as e:
        print(f"ECC alignment failed: {e}")
        return test_img, None, 0


def optical_flow_alignment(ref_img, test_img, method='farneback'):
    """
    Dense optical flow alignment
    
    Best for: Motion tracking, understanding pixel movements
    Speed: Moderate
    Handles: Complex deformations, non-rigid transformations
    """
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if len(ref_img.shape) == 3 else ref_img
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY) if len(test_img.shape) == 3 else test_img
    
    if method == 'farneback':
        # Dense optical flow (Farneback method)
        flow = cv2.calcOpticalFlowFarneback(
            ref_gray, test_gray,
            None,
            pyr_scale=0.5,      # Pyramid scale
            levels=3,           # Number of pyramid layers
            winsize=15,         # Averaging window size
            iterations=3,       # Iterations at each pyramid level
            poly_n=5,           # Polynomial expansion neighborhood
            poly_sigma=1.2,     # Gaussian standard deviation
            flags=0
        )
    else:
        # Could add other optical flow methods here
        flow = None
    
    if flow is None:
        return test_img, None
    
    # Create remapping coordinates
    h, w = ref_gray.shape
    flow_map = np.zeros((h, w, 2), dtype=np.float32)
    flow_map[..., 0] = np.arange(w)
    flow_map[..., 1] = np.arange(h)[:, np.newaxis]
    
    # Subtract flow (we want to warp test to ref)
    flow_map -= flow
    
    # Remap image
    aligned = cv2.remap(test_img, flow_map, None, cv2.INTER_LINEAR, 
                       borderMode=cv2.BORDER_CONSTANT)
    
    return aligned, flow


def phase_correlation_alignment(ref_img, test_img):
    """
    Phase correlation for translation estimation
    
    Best for: Pure translation, camera shake, fast alignment
    Speed: Very fast
    Handles: Translation only
    """
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if len(ref_img.shape) == 3 else ref_img
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY) if len(test_img.shape) == 3 else test_img
    
    # Convert to float32 for phase correlation
    ref_float = ref_gray.astype(np.float32)
    test_float = test_gray.astype(np.float32)
    
    # Compute phase shift
    shift, response = cv2.phaseCorrelate(ref_float, test_float)
    
    # Create translation matrix
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    
    # Apply translation
    h, w = ref_img.shape[:2]
    aligned = cv2.warpAffine(test_img, M, (w, h))
    
    return aligned, shift, response


def orb_alignment(ref_img, test_img, nfeatures=5000, transform_type='homography'):
    """
    ORB feature-based alignment (faster than SIFT)
    
    Best for: Real-time applications, moderate transformations
    Speed: Fast
    Handles: Rotation, scale, perspective (depending on transform_type)
    
    Args:
        transform_type: 'homography', 'affine', or 'similarity'
    """
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if len(ref_img.shape) == 3 else ref_img
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY) if len(test_img.shape) == 3 else test_img
    
    # ORB detector
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(test_gray, None)
    
    if des1 is None or des2 is None:
        print("ORB: No features detected")
        return test_img, None
    
    # BFMatcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test
    good = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)
    
    min_matches = 4 if transform_type == 'homography' else 3
    if len(good) < min_matches:
        print(f"ORB: Not enough matches ({len(good)} < {min_matches})")
        return test_img, None
    
    # Extract matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
    
    h, w = ref_img.shape[:2]
    
    # Compute transformation
    if transform_type == 'homography':
        src_pts = src_pts.reshape(-1, 1, 2)
        dst_pts = dst_pts.reshape(-1, 1, 2)
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if M is None:
            return test_img, None
        aligned = cv2.warpPerspective(test_img, M, (w, h))
    elif transform_type == 'affine':
        M, inliers = cv2.estimateAffine2D(dst_pts, src_pts, method=cv2.RANSAC)
        if M is None:
            return test_img, None
        aligned = cv2.warpAffine(test_img, M, (w, h))
    elif transform_type == 'similarity':
        M, inliers = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC)
        if M is None:
            return test_img, None
        aligned = cv2.warpAffine(test_img, M, (w, h))
    else:
        return test_img, None
    
    return aligned, M


def similarity_transform_alignment(ref_img, test_img, feature_detector='sift'):
    """
    Similarity transform (scale + rotation + translation)
    
    Best for: Object tracking, constrained transformations
    Speed: Moderate
    Handles: Scale, rotation, translation (NO perspective distortion)
    """
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if len(ref_img.shape) == 3 else ref_img
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY) if len(test_img.shape) == 3 else test_img
    
    # Feature detector
    if feature_detector == 'sift':
        detector = cv2.SIFT_create()
    elif feature_detector == 'orb':
        detector = cv2.ORB_create(nfeatures=5000)
    else:
        detector = cv2.SIFT_create()
    
    kp1, des1 = detector.detectAndCompute(ref_gray, None)
    kp2, des2 = detector.detectAndCompute(test_gray, None)
    
    if des1 is None or des2 is None:
        return test_img, None
    
    # Match features
    if feature_detector == 'orb':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test
    good = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.7 * n.distance:
                good.append(m)
    
    if len(good) < 3:
        return test_img, None
    
    # Extract points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
    
    # Estimate similarity transform (4 DOF: scale, rotation, tx, ty)
    M, inliers = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC)
    
    if M is None:
        return test_img, None
    
    h, w = ref_img.shape[:2]
    aligned = cv2.warpAffine(test_img, M, (w, h))
    
    return aligned, M


def template_matching_alignment(ref_img, test_img, scale_range=(0.8, 1.2), num_scales=20):
    """
    Template matching with multi-scale search
    
    Best for: Known object in scene, scale changes
    Speed: Slow (exhaustive search)
    Handles: Scale + translation
    """
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if len(ref_img.shape) == 3 else ref_img
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY) if len(test_img.shape) == 3 else test_img
    
    best_score = -1
    best_scale = 1.0
    best_loc = (0, 0)
    
    # Multi-scale search
    for scale in np.linspace(scale_range[0], scale_range[1], num_scales):
        h, w = test_gray.shape
        new_h, new_w = int(h * scale), int(w * scale)
        
        if new_h <= 0 or new_w <= 0:
            continue
        
        resized = cv2.resize(test_gray, (new_w, new_h))
        
        # Skip if template larger than image
        if resized.shape[0] > ref_gray.shape[0] or resized.shape[1] > ref_gray.shape[1]:
            continue
        
        # Template matching
        result = cv2.matchTemplate(ref_gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
            best_scale = scale
            best_loc = max_loc
    
    # Apply best transformation
    h, w = test_img.shape[:2]
    scaled = cv2.resize(test_img, (int(w * best_scale), int(h * best_scale)))
    
    # Create aligned image
    h_ref, w_ref = ref_img.shape[:2]
    aligned = np.zeros_like(ref_img)
    
    y, x = best_loc
    h_s, w_s = scaled.shape[:2]
    
    # Ensure we don't exceed bounds
    y_end = min(y + h_s, h_ref)
    x_end = min(x + w_s, w_ref)
    h_crop = y_end - y
    w_crop = x_end - x
    
    aligned[y:y_end, x:x_end] = scaled[:h_crop, :w_crop]
    
    return aligned, (best_scale, best_loc, best_score)


def adaptive_alignment(ref_img, test_img, methods=['ecc', 'orb', 'phase']):
    """
    Try multiple alignment methods and return best result
    
    Automatically selects best method based on success metrics
    """
    results = []
    
    for method in methods:
        try:
            if method == 'ecc':
                aligned, transform, score = ecc_alignment(ref_img, test_img)
                results.append(('ecc', aligned, transform, score))
            elif method == 'orb':
                aligned, transform = orb_alignment(ref_img, test_img)
                if transform is not None:
                    # Compute similarity score
                    score = compute_alignment_quality(ref_img, aligned)
                    results.append(('orb', aligned, transform, score))
            elif method == 'phase':
                aligned, shift, response = phase_correlation_alignment(ref_img, test_img)
                results.append(('phase', aligned, shift, response))
            elif method == 'optical_flow':
                aligned, flow = optical_flow_alignment(ref_img, test_img)
                if flow is not None:
                    score = compute_alignment_quality(ref_img, aligned)
                    results.append(('optical_flow', aligned, flow, score))
            elif method == 'similarity':
                aligned, transform = similarity_transform_alignment(ref_img, test_img)
                if transform is not None:
                    score = compute_alignment_quality(ref_img, aligned)
                    results.append(('similarity', aligned, transform, score))
        except Exception as e:
            print(f"Method {method} failed: {e}")
            continue
    
    if not results:
        return test_img, None, 'none'
    
    # Select best based on score
    best = max(results, key=lambda x: x[3] if len(x) > 3 else 0)
    
    return best[1], best[2], best[0]


def compute_alignment_quality(ref_img, aligned_img):
    """
    Compute alignment quality metric (normalized cross-correlation)
    """
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if len(ref_img.shape) == 3 else ref_img
    aligned_gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY) if len(aligned_img.shape) == 3 else aligned_img
    
    # Normalize
    ref_norm = (ref_gray - ref_gray.mean()) / (ref_gray.std() + 1e-8)
    aligned_norm = (aligned_gray - aligned_gray.mean()) / (aligned_gray.std() + 1e-8)
    
    # Cross-correlation
    ncc = np.mean(ref_norm * aligned_norm)
    
    return ncc
