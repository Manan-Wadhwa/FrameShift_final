"""
Image Differencing with PatchCore + SAM
Compares two images and detects changes using difference maps
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys
import gc
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.preprocess import preprocess
from utils.rough_mask import generate_rough_mask
from utils.sam_refine import sam_refine
from utils.visualization import create_heatmap, create_overlay
from pipelines.anomaly_patchcore_sam import run_patchcore_sam_pipeline


def register_images(img_a, img_b):
    """Align two images using feature matching"""
    try:
        # Convert to grayscale for feature matching
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
        
        # Find SIFT keypoints and descriptors
        sift = cv2.SIFT_create()
        kp_a, des_a = sift.detectAndCompute(gray_a, None)
        kp_b, des_b = sift.detectAndCompute(gray_b, None)
        
        if des_a is None or des_b is None:
            st.warning("Could not find features for alignment, using unaligned comparison")
            return img_a, img_b
        
        # Match features
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_a, des_b, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:
            st.warning("Not enough matches for alignment")
            return img_a, img_b
        
        # Find homography
        src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            st.warning("Could not compute homography")
            return img_a, img_b
        
        # Warp img_b to align with img_a
        h, w = img_a.shape[:2]
        img_b_aligned = cv2.warpPerspective(img_b, H, (w, h))
        
        return img_a, img_b_aligned
        
    except Exception as e:
        st.warning(f"Registration failed: {e}")
        return img_a, img_b


def compute_difference_maps(img_a, img_b):
    """Compute multiple types of difference maps"""
    # Ensure same size
    h, w = img_a.shape[:2]
    img_b_resized = cv2.resize(img_b, (w, h))
    
    # 1. Pixel-wise difference
    diff_pixel = cv2.absdiff(img_a, img_b_resized)
    diff_gray = cv2.cvtColor(diff_pixel, cv2.COLOR_RGB2GRAY)
    
    # 2. SSIM-based difference
    from skimage.metrics import structural_similarity as ssim
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(img_b_resized, cv2.COLOR_RGB2GRAY)
    
    score, diff_ssim = ssim(gray_a, gray_b, full=True)
    diff_ssim = (1 - diff_ssim) * 255
    diff_ssim = diff_ssim.astype(np.uint8)
    
    # 3. Multi-channel difference (color focus)
    diff_r = np.abs(img_a[:,:,0].astype(float) - img_b_resized[:,:,0].astype(float))
    diff_g = np.abs(img_a[:,:,1].astype(float) - img_b_resized[:,:,1].astype(float))
    diff_b_ch = np.abs(img_a[:,:,2].astype(float) - img_b_resized[:,:,2].astype(float))
    diff_color = (diff_r + diff_g + diff_b_ch) / 3.0
    diff_color = (diff_color / 255.0 * 255).astype(np.uint8)
    
    return diff_gray, diff_ssim, diff_color, img_b_resized


def analyze_with_patchcore_sam(img_a, img_b, diff_map):
    """Apply PatchCore + SAM to the difference map"""
    try:
        # Preprocess reference and test images
        ref_proc, test_proc = preprocess(img_a, img_b)
        
        # Generate rough mask from difference
        rough_mask, _ = generate_rough_mask(ref_proc, test_proc)
        
        # Refine with SAM
        refined_mask = sam_refine(test_proc, rough_mask)
        
        # Run PatchCore + SAM pipeline on difference
        # Create difference image (3-channel)
        if len(diff_map.shape) == 2:
            diff_3ch = cv2.cvtColor(diff_map, cv2.COLOR_GRAY2RGB)
        else:
            diff_3ch = diff_map
        
        result = run_patchcore_sam_pipeline(
            test_proc, 
            refined_mask, 
            ref_img=ref_proc
        )
        
        return result, refined_mask, ref_proc, test_proc
        
    except Exception as e:
        st.error(f"PatchCore + SAM analysis failed: {e}")
        return None, None, None, None


def main():
    st.set_page_config(
        page_title="Image Differencing - PatchCore + SAM",
        page_icon="🔄",
        layout="wide"
    )
    
    # Initialize session state
    if 'diff_output' not in st.session_state:
        st.session_state.diff_output = None
    
    st.title("🔄 Image Differencing with PatchCore + SAM")
    st.markdown("**Compare two images and detect changes with proper differencing**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Input Images")
        
        # File uploaders
        uploaded_ref = st.file_uploader("Reference Image (A)", type=["jpg", "jpeg", "png"], key="diff_ref")
        uploaded_test = st.file_uploader("Test Image (B)", type=["jpg", "jpeg", "png"], key="diff_test")
        
        st.markdown("---")
        
        # Options
        st.header("⚙️ Options")
        align_images = st.checkbox("Align images before comparison", value=True)
        show_diff_maps = st.checkbox("Show difference maps", value=True)
        
        st.markdown("---")
        
        # Run button
        run_button = st.button("🚀 Analyze Differences", type="primary", use_container_width=True)
    
    # Main content
    if run_button:
        if uploaded_ref is None or uploaded_test is None:
            st.error("❌ Please upload both images")
            return
        
        # Load images
        ref_img = Image.open(uploaded_ref)
        test_img = Image.open(uploaded_test)
        
        ref_array = np.array(ref_img.convert('RGB'))
        test_array = np.array(test_img.convert('RGB'))
        
        st.success(f"✓ Loaded images: Ref {ref_array.shape}, Test {test_array.shape}")
        
        # Step 1: Image Alignment
        st.header("Step 1️⃣: Image Alignment")
        
        if align_images:
            with st.spinner("Aligning images..."):
                ref_aligned, test_aligned = register_images(ref_array, test_array)
                st.success("✓ Images aligned")
        else:
            ref_aligned = ref_array
            test_aligned = test_array
            st.info("ℹ️ Skipped alignment")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(ref_aligned, caption="Reference (Aligned)", use_column_width=True)
        with col2:
            st.image(test_aligned, caption="Test (Aligned)", use_column_width=True)
        
        st.markdown("---")
        
        # Step 2: Generate Difference Maps
        st.header("Step 2️⃣: Difference Maps")
        
        with st.spinner("Computing difference maps..."):
            diff_gray, diff_ssim, diff_color, test_resized = compute_difference_maps(ref_aligned, test_aligned)
        
        if show_diff_maps:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(diff_gray, caption="Pixel Difference", use_column_width=True, clamp=True)
            
            with col2:
                st.image(diff_ssim, caption="SSIM Difference", use_column_width=True, clamp=True)
            
            with col3:
                st.image(diff_color, caption="Color Difference", use_column_width=True, clamp=True)
            
            st.markdown("---")
        
        # Step 3: PatchCore + SAM Analysis
        st.header("Step 3️⃣: PatchCore + SAM Analysis")
        
        with st.spinner("Running PatchCore + SAM on difference maps..."):
            result, refined_mask, ref_proc, test_proc = analyze_with_patchcore_sam(
                ref_aligned, test_resized, diff_gray
            )
        
        if result is None:
            st.error("Analysis failed")
            gc.collect()
            return
        
        st.success("✓ Analysis complete")
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Heatmap")
            if result.get("heatmap") is not None:
                st.image(result["heatmap"], use_column_width=True)
        
        with col2:
            st.subheader("Detected Changes")
            if result.get("mask_final") is not None:
                mask_rgb = cv2.cvtColor(result["mask_final"], cv2.COLOR_GRAY2RGB)
                st.image(mask_rgb, use_column_width=True)
        
        with col3:
            st.subheader("Overlay")
            if result.get("overlay") is not None:
                st.image(result["overlay"], use_container_width=True)
        
        st.markdown("---")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            severity = result.get("severity", 0.0)
            st.metric("Severity Score", f"{severity:.3f}")
        
        with col2:
            if refined_mask is not None:
                change_pct = (np.sum(refined_mask > 0) / (refined_mask.shape[0] * refined_mask.shape[1])) * 100
                st.metric("Change Area", f"{change_pct:.1f}%")
        
        with col3:
            diff_mean = np.mean(diff_gray)
            st.metric("Avg Difference", f"{diff_mean:.1f}")
        
        st.markdown("---")
        
        # Report
        st.subheader("📄 Analysis Report")
        report = result.get("report", "Report not available")
        with st.expander("View Full Report"):
            st.info(report)
        
        # Cache output
        st.session_state.diff_output = result
        
        # Cleanup
        gc.collect()
    
    else:
        st.info("👈 Upload two images and click 'Analyze Differences' to begin")
        
        with st.expander("ℹ️ How it works"):
            st.markdown("""
            ### Image Differencing Pipeline:
            
            1. **Alignment**: Registers Image B to Image A using SIFT features
            2. **Difference Maps**: Computes pixel-wise, SSIM, and color differences
            3. **PatchCore Analysis**: Detects anomalies in the difference map
            4. **SAM Refinement**: Refines boundaries using Segment Anything Model
            
            ### Results:
            - **Heatmap**: Intensity of detected changes
            - **Mask**: Binary segmentation of changed regions
            - **Overlay**: Changes highlighted on test image
            - **Severity**: Overall change magnitude
            """)


if __name__ == "__main__":
    main()
