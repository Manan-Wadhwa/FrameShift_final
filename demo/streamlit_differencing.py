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


def analyze_with_patchcore_sam(img_a, img_b, diff_map, use_preprocessing=True):
    """Apply PatchCore + SAM to the difference map"""
    try:
        # Preprocess reference and test images
        if use_preprocessing:
            ref_proc, test_proc = preprocess(img_a, img_b)
        else:
            # Skip preprocessing - just resize to standard size
            h, w = 336, 336
            ref_proc = cv2.resize(img_a, (w, h))
            test_proc = cv2.resize(img_b, (w, h))
        
        # Generate rough mask from difference
        rough_mask, _ = generate_rough_mask(ref_proc, test_proc)
        
        # Refine with SAM
        refined_mask = sam_refine(test_proc, rough_mask)
        
        # Run PatchCore + SAM on BOTH images separately to compare heatmaps
        result_a = run_patchcore_sam_pipeline(ref_proc, refined_mask, ref_img=ref_proc)
        result_b = run_patchcore_sam_pipeline(test_proc, refined_mask, ref_img=test_proc)
        
        if result_a and result_b:
            hm_a = result_a.get("heatmap")
            hm_b = result_b.get("heatmap")
            
            if hm_a is not None and hm_b is not None:
                # Resize heatmap_b to match heatmap_a size
                h, w = hm_a.shape[:2]
                hm_b = cv2.resize(hm_b, (w, h))
                
                # Compute heatmap difference
                hm_diff = cv2.absdiff(hm_a, hm_b)
                hm_diff_gray = cv2.cvtColor(hm_diff, cv2.COLOR_RGB2GRAY)
                
                return {
                    "result_a": result_a,
                    "result_b": result_b,
                    "heatmap_a": hm_a,
                    "heatmap_b": hm_b,
                    "heatmap_diff": hm_diff,
                    "heatmap_diff_gray": hm_diff_gray,
                    "refined_mask": refined_mask,
                    "ref_proc": ref_proc,
                    "test_proc": test_proc
                }
        
        return None
        
    except Exception as e:
        st.error(f"PatchCore + SAM analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_with_dinov3_patchcore_sam(img_a, img_b, use_preprocessing=True):
    """Apply DINOv3 + PatchCore + SAM for static feature-based anomaly detection"""
    try:
        # Preprocess reference and test images
        if use_preprocessing:
            ref_proc, test_proc = preprocess(img_a, img_b)
        else:
            # Skip preprocessing - just resize to standard size
            h, w = 336, 336
            ref_proc = cv2.resize(img_a, (w, h))
            test_proc = cv2.resize(img_b, (w, h))
        
        # Generate rough mask from difference
        rough_mask, _ = generate_rough_mask(ref_proc, test_proc)
        
        # Refine with SAM
        refined_mask = sam_refine(test_proc, rough_mask)
        
        # Import the DINOv3 pipeline
        from pipelines.semantic_dinov3_patchcore_sam import run_dinov3_patchcore_sam_pipeline
        
        # Run DINOv3 + PatchCore + SAM pipeline on both images
        result_a = run_dinov3_patchcore_sam_pipeline(ref_proc, test_proc, refined_mask)
        result_b = run_dinov3_patchcore_sam_pipeline(test_proc, ref_proc, refined_mask)
        
        if result_a and result_b:
            hm_a = result_a.get("heatmap")
            hm_b = result_b.get("heatmap")
            
            if hm_a is not None and hm_b is not None:
                # Ensure both heatmaps are same size
                h, w = hm_a.shape[:2]
                if hm_b.shape[:2] != (h, w):
                    hm_b = cv2.resize(hm_b, (w, h))
                
                # Compute heatmap difference
                hm_diff = cv2.absdiff(hm_a, hm_b)
                hm_diff_gray = cv2.cvtColor(hm_diff, cv2.COLOR_RGB2GRAY)
                
                return {
                    "result_a": result_a,
                    "result_b": result_b,
                    "heatmap_a": hm_a,
                    "heatmap_b": hm_b,
                    "heatmap_diff": hm_diff,
                    "heatmap_diff_gray": hm_diff_gray,
                    "refined_mask": refined_mask,
                    "ref_proc": ref_proc,
                    "test_proc": test_proc
                }
        
        return None
        
    except Exception as e:
        st.error(f"DINOv3 + PatchCore + SAM analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


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
        
        # Tab between upload and samples
        tab1, tab2 = st.tabs(["Upload", "Samples"])
        
        uploaded_ref = None
        uploaded_test = None
        
        with tab1:
            uploaded_ref = st.file_uploader("Reference Image (A)", type=["jpg", "jpeg", "png"], key="diff_ref")
            uploaded_test = st.file_uploader("Test Image (B)", type=["jpg", "jpeg", "png"], key="diff_test")
        
        with tab2:
            sample_pairs = {
                "Livery Change (back1/back2)": ("samples/back1.jpeg", "samples/back2.jpeg"),
                "Object Change (copy1/copy2)": ("samples/copy1.jpeg", "samples/copy2.jpeg"),
                "Tire Damage (crack1/crack2)": ("samples/crack1.jpg", "samples/crack2.png"),
                "Subtle Change (side1/side2)": ("samples/side1.jpeg", "samples/side2.jpeg")
            }
            selected_sample = st.selectbox("Choose sample pair:", list(sample_pairs.keys()))
        
        st.markdown("---")
        
        # Options
        st.header("⚙️ Options")
        align_images = st.checkbox("Align images before comparison", value=True)
        use_preprocessing = st.checkbox("Use preprocessing", value=True)
        
        # Pipeline selection
        st.subheader("Pipeline")
        pipeline_choice = st.radio("Choose analysis pipeline:", 
            ["PatchCore + SAM", "DINOv3 + PatchCore + SAM"],
            help="PatchCore+SAM: Anomaly detection | DINOv3+PatchCore+SAM: Static feature detection")
        
        show_diff_maps = st.checkbox("Show difference maps", value=True)
        use_preprocessing = st.checkbox("Use preprocessing", value=True, help="Resize, denoise, gamma correction")
        
        st.markdown("---")
        
        # Run button
        run_button = st.button("🚀 Analyze Differences", type="primary", use_container_width=True)
    
    # Main content
    if run_button:
        # Determine which images to use
        if uploaded_ref is not None and uploaded_test is not None:
            ref_img = Image.open(uploaded_ref)
            test_img = Image.open(uploaded_test)
            ref_array = np.array(ref_img.convert('RGB'))
            test_array = np.array(test_img.convert('RGB'))
        else:
            # Use selected sample
            ref_path, test_path = sample_pairs[selected_sample]
            if not os.path.exists(ref_path) or not os.path.exists(test_path):
                st.error(f"❌ Sample images not found: {ref_path}, {test_path}")
                return
            ref_array = np.array(Image.open(ref_path).convert('RGB'))
            test_array = np.array(Image.open(test_path).convert('RGB'))
        
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
        
        # Step 3: Analysis
        st.header("Step 3️⃣: Analysis")
        
        with st.spinner(f"Running {pipeline_choice}..."):
            if pipeline_choice == "DINOv3 + PatchCore + SAM":
                from pipelines.semantic_dinov3_patchcore_sam import run_dinov3_patchcore_sam_pipeline
                analysis_result = analyze_with_dinov3_patchcore_sam(
                    ref_aligned, test_resized, use_preprocessing
                )
            else:
                analysis_result = analyze_with_patchcore_sam(
                    ref_aligned, test_resized, diff_gray, use_preprocessing=use_preprocessing
                )
        
        if analysis_result is None:
            st.error("Analysis failed")
            gc.collect()
            return
        
        st.success("✓ Analysis complete")
        
        # Display heatmap comparison
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Heatmap A (Reference)")
            if analysis_result.get("heatmap_a") is not None:
                st.image(analysis_result["heatmap_a"], use_container_width=True)
        
        with col2:
            st.subheader("Heatmap B (Test)")
            if analysis_result.get("heatmap_b") is not None:
                st.image(analysis_result["heatmap_b"], use_container_width=True)
        
        with col3:
            st.subheader("Heatmap Difference")
            if analysis_result.get("heatmap_diff") is not None:
                st.image(analysis_result["heatmap_diff"], use_container_width=True)
        
        st.markdown("---")
        
        # Display detection masks
        st.header("Detected Changes")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Image A Mask")
            result_a = analysis_result.get("result_a")
            if result_a and result_a.get("mask_final") is not None:
                mask_a_rgb = cv2.cvtColor(result_a["mask_final"], cv2.COLOR_GRAY2RGB)
                st.image(mask_a_rgb, use_container_width=True)
        
        with col2:
            st.subheader("Image B Mask")
            result_b = analysis_result.get("result_b")
            if result_b and result_b.get("mask_final") is not None:
                mask_b_rgb = cv2.cvtColor(result_b["mask_final"], cv2.COLOR_GRAY2RGB)
                st.image(mask_b_rgb, use_container_width=True)
        
        st.markdown("---")
        
        # Display overlays
        st.header("Overlays")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Image A Overlay")
            if result_a and result_a.get("overlay") is not None:
                st.image(result_a["overlay"], use_container_width=True)
        
        with col2:
            st.subheader("Image B Overlay")
            if result_b and result_b.get("overlay") is not None:
                st.image(result_b["overlay"], use_container_width=True)
        
        st.markdown("---")
        
        # Metrics
        st.header("📊 Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            result_a = analysis_result.get("result_a", {})
            severity_a = result_a.get("severity", 0.0)
            st.metric("Severity (Image A)", f"{severity_a:.3f}")
        
        with col2:
            result_b = analysis_result.get("result_b", {})
            severity_b = result_b.get("severity", 0.0)
            st.metric("Severity (Image B)", f"{severity_b:.3f}")
        
        with col3:
            refined_mask = analysis_result.get("refined_mask")
            if refined_mask is not None:
                change_pct = (np.sum(refined_mask > 0) / (refined_mask.shape[0] * refined_mask.shape[1])) * 100
                st.metric("Change Area", f"{change_pct:.1f}%")
        
        with col4:
            diff_mean = np.mean(diff_gray)
            st.metric("Avg Difference", f"{diff_mean:.1f}")
        
        st.markdown("---")
        
        # Cache output
        st.session_state.diff_output = analysis_result
        
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
