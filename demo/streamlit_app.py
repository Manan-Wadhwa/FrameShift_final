"""
Streamlit UI for F1 Visual Difference Engine
Interactive dashboard for comparing all 4 pipelines
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from main_pipeline import run_all_pipelines


def display_pipeline_result(pipeline_name, result, col):
    """Display results for a single pipeline in a column"""
    with col:
        st.subheader(pipeline_name)
        
        # Display heatmap
        st.markdown("**Heatmap**")
        st.image(result["heatmap"], use_container_width=True)
        
        # Display mask
        st.markdown("**Segmentation Mask**")
        mask_rgb = cv2.cvtColor(result["mask_final"], cv2.COLOR_GRAY2RGB)
        st.image(mask_rgb, use_container_width=True)
        
        # Display overlay
        st.markdown("**Overlay**")
        st.image(result["overlay"], use_container_width=True)
        
        # Display metrics
        if "severity" in result:
            st.metric("Severity Score", f"{result['severity']:.3f}")
        
        # Display report
        st.markdown("**Analysis Report**")
        st.info(result.get("report", "No report generated"))


def main():
    st.set_page_config(
        page_title="F1 Visual Difference Engine",
        page_icon="🏎️",
        layout="wide"
    )
    
    # Header
    st.title("🏎️ F1 Visual Difference Engine")
    st.markdown("**Multi-Pipeline Visual Change Detection System**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Input Images")
        
        # Option 1: Upload custom images
        st.subheader("Upload Custom Images")
        uploaded_ref = st.file_uploader("Reference Image (A)", type=["jpg", "jpeg", "png"])
        uploaded_test = st.file_uploader("Test Image (B)", type=["jpg", "jpeg", "png"])
        
        st.markdown("---")
        
        # Option 2: Select from samples
        st.subheader("Or Select Sample Pair")
        sample_pairs = {
            "Livery Change (back1/back2)": ("samples/back1.jpeg", "samples/back2.jpeg"),
            "Object Change (copy1/copy2)": ("samples/copy1.jpeg", "samples/copy2.jpeg"),
            "Tire Damage (crack1/crack2)": ("samples/crack1.jpg", "samples/crack2.png"),
            "Subtle Change (side1/side2)": ("samples/side1.jpeg", "samples/side2.jpeg")
        }
        
        selected_sample = st.selectbox("Choose sample pair:", list(sample_pairs.keys()))
        
        st.markdown("---")
        
        # Run button
        run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    # Main content
    if run_button:
        # Determine which images to use
        if uploaded_ref is not None and uploaded_test is not None:
            # Save uploaded files temporarily
            ref_path = "temp_ref.jpg"
            test_path = "temp_test.jpg"
            
            with open(ref_path, "wb") as f:
                f.write(uploaded_ref.getbuffer())
            with open(test_path, "wb") as f:
                f.write(uploaded_test.getbuffer())
        else:
            # Use selected sample
            ref_path, test_path = sample_pairs[selected_sample]
        
        # Display input images
        st.header("📷 Input Images")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Reference Image (A)")
            ref_img = Image.open(ref_path)
            st.image(ref_img, use_container_width=True)
        
        with col2:
            st.subheader("Test Image (B)")
            test_img = Image.open(test_path)
            st.image(test_img, use_container_width=True)
        
        st.markdown("---")
        
        # Run pipeline
        with st.spinner("Running all 4 detection pipelines..."):
            try:
                output = run_all_pipelines(ref_path, test_path)
            except Exception as e:
                st.error(f"Error running pipeline: {str(e)}")
                return
        
        # Display routing results
        st.header("🧭 Routing Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Predicted Type", output["predicted_type"].upper())
        with col2:
            st.metric("Confidence", output["confidence"].upper())
        with col3:
            st.metric("Texture Variance", f"{output['features']['texture_var']:.1f}")
        with col4:
            st.metric("Color Shift", f"{output['features']['color_shift']:.3f}")
        
        # Display feature details
        with st.expander("📊 Detailed Routing Features"):
            st.json(output["features"])
        
        st.markdown("---")
        
        # Display preprocessing results
        st.header("🔧 Preprocessing Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Aligned Reference")
            st.image(output["preprocessed"]["ref"], use_container_width=True)
        
        with col2:
            st.subheader("Rough Mask (SSIM)")
            rough_rgb = cv2.cvtColor(output["preprocessed"]["rough_mask"], cv2.COLOR_GRAY2RGB)
            st.image(rough_rgb, use_container_width=True)
        
        with col3:
            st.subheader("Refined Mask (SAM)")
            refined_rgb = cv2.cvtColor(output["preprocessed"]["refined_mask"], cv2.COLOR_GRAY2RGB)
            st.image(refined_rgb, use_container_width=True)
        
        st.markdown("---")
        
        # Display all pipeline results
        st.header("🔬 Pipeline Comparison")
        
        # SEMANTIC PIPELINES
        st.subheader("🎨 Semantic Pipelines")
        col1, col2 = st.columns(2)
        
        display_pipeline_result("DINO", output["results"]["dino"], col1)
        display_pipeline_result("CLIP", output["results"]["clip"], col2)
        
        st.markdown("---")
        
        # ANOMALY PIPELINES
        st.subheader("⚠️ Anomaly Pipelines")
        col1, col2 = st.columns(2)
        
        display_pipeline_result("PatchCore", output["results"]["patchcore"], col1)
        display_pipeline_result("PaDiM", output["results"]["padim"], col2)
        
        st.markdown("---")
        
        # Manual selection
        st.header("✅ Select Top 2 Pipelines")
        st.markdown("Based on visual inspection, select the 2 best-performing pipelines:")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            dino_selected = st.checkbox("DINO")
        with col2:
            clip_selected = st.checkbox("CLIP")
        with col3:
            patchcore_selected = st.checkbox("PatchCore")
        with col4:
            padim_selected = st.checkbox("PaDiM")
        
        selected_pipelines = []
        if dino_selected:
            selected_pipelines.append("DINO")
        if clip_selected:
            selected_pipelines.append("CLIP")
        if patchcore_selected:
            selected_pipelines.append("PatchCore")
        if padim_selected:
            selected_pipelines.append("PaDiM")
        
        if len(selected_pipelines) == 2:
            st.success(f"✅ Selected pipelines: {', '.join(selected_pipelines)}")
        elif len(selected_pipelines) > 2:
            st.warning("⚠️ Please select exactly 2 pipelines")
        else:
            st.info("ℹ️ Please select 2 pipelines for final comparison")
        
        # Clean up temp files
        if uploaded_ref is not None and uploaded_test is not None:
            if os.path.exists("temp_ref.jpg"):
                os.remove("temp_ref.jpg")
            if os.path.exists("temp_test.jpg"):
                os.remove("temp_test.jpg")
    
    else:
        # Initial state
        st.info("👈 Upload images or select a sample pair from the sidebar, then click 'Run Analysis'")
        
        # Show example
        st.header("📖 About")
        st.markdown("""
        The **F1 Visual Difference Engine** is a comprehensive multi-pipeline system for detecting 
        visual changes between two images of Formula 1 cars.
        
        ### 🔍 Features:
        - **4 Detection Pipelines**: DINO, CLIP, PatchCore, PaDiM
        - **Intelligent Routing**: Automatically predicts semantic vs anomaly type
        - **SAM Refinement**: High-quality segmentation masks
        - **LLaVA Reports**: Natural language explanations
        
        ### 📋 Pipeline Types:
        - **Semantic (DINO, CLIP)**: Livery changes, color modifications, design updates
        - **Anomaly (PatchCore, PaDiM)**: Structural damage, tire defects, surface anomalies
        
        ### 🎯 Use Cases:
        1. **Livery Change Detection**: Compare car designs across seasons
        2. **Damage Assessment**: Identify tire or bodywork damage
        3. **Quality Control**: Detect manufacturing defects
        4. **Compliance Verification**: Ensure regulatory conformance
        """)


if __name__ == "__main__":
    main()
