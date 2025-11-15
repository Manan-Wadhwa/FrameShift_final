"""
Streamlit UI for F1 Visual Difference Engine
Interactive dashboard for comparing all 4 pipelines
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys
import gc
from pathlib import Path

# Add parent directory to path to import main_pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))
from main_pipeline import run_all_pipelines

# Configure Streamlit for memory efficiency
st.set_page_config(
    page_title="F1 Visual Difference Engine",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)


def display_pipeline_result(pipeline_name, result, col):
    """Display results for a single pipeline in a column"""
    if result is None:
        with col:
            st.warning(f"⚠️ {pipeline_name} result not available")
        return
    
    with col:
        st.subheader(pipeline_name)
        
        try:
            # Display heatmap (compress to reduce memory)
            st.markdown("**Heatmap**")
            if "heatmap" in result and result["heatmap"] is not None:
                heatmap = result["heatmap"]
                # Compress if needed
                if heatmap.nbytes > 5e6:
                    scale = int(np.sqrt(5e6 / heatmap.nbytes))
                    heatmap = cv2.resize(heatmap, (heatmap.shape[1]//scale, heatmap.shape[0]//scale))
                st.image(heatmap, width='stretch', use_column_width=True)
            else:
                st.warning("Heatmap not available")
            
            # Display mask
            st.markdown("**Segmentation Mask**")
            if "mask_final" in result and result["mask_final"] is not None:
                mask_rgb = cv2.cvtColor(result["mask_final"], cv2.COLOR_GRAY2RGB)
                st.image(mask_rgb, width='stretch', use_column_width=True)
            else:
                st.warning("Mask not available")
            
            # Display overlay (compress if needed)
            st.markdown("**Overlay**")
            if "overlay" in result and result["overlay"] is not None:
                overlay = result["overlay"]
                if overlay.nbytes > 5e6:
                    scale = int(np.sqrt(5e6 / overlay.nbytes))
                    overlay = cv2.resize(overlay, (overlay.shape[1]//scale, overlay.shape[0]//scale))
                st.image(overlay, width='stretch', use_column_width=True)
            else:
                st.warning("Overlay not available")
            
            # Display metrics
            if "severity" in result and result["severity"] is not None:
                st.metric("Severity Score", f"{result['severity']:.3f}")
            
            # Display report (with expander to save space)
            st.markdown("**Analysis Report**")
            report_text = result.get("report", None)
            if report_text:
                with st.expander("📄 View Full Report"):
                    st.info(report_text)
            else:
                st.warning("Report not available")
        except Exception as e:
            st.error(f"Error displaying {pipeline_name}: {str(e)}")


def main():
    # Initialize session state for memory management
    if 'last_analysis' not in st.session_state:
        st.session_state.last_analysis = None
    if 'current_output' not in st.session_state:
        st.session_state.current_output = None
    
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
        
        # Model Selection
        st.header("🎯 Select Models to Run")
        st.markdown("Choose which pipelines to execute:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎨 Semantic")
            dino_enabled = st.checkbox("DINO", value=True, key="dino_check")
            clip_enabled = st.checkbox("CLIP", value=True, key="clip_check")
        
        with col2:
            st.subheader("⚠️ Anomaly")
            patchcore_enabled = st.checkbox("PatchCore", value=True, key="patchcore_check")
            padim_enabled = st.checkbox("PaDiM", value=True, key="padim_check")
        
        st.subheader("🔬 Advanced")
        patchcore_sam_enabled = st.checkbox("PatchCore + SAM", value=False, key="patchcore_sam_check")
        patchcore_knn_enabled = st.checkbox("PatchCore KNN (v3.0)", value=False, key="patchcore_knn_check")
        
        st.markdown("---")
        
        # Run button
        run_button = st.button("🚀 Run Analysis", type="primary", width='stretch')
    
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
            st.image(ref_img, width='stretch')
        
        with col2:
            st.subheader("Test Image (B)")
            test_img = Image.open(test_path)
            st.image(test_img, width='stretch')
        
        st.markdown("---")
        
        # Run pipeline
        with st.spinner("Running selected detection pipelines..."):
            try:
                # Build list of pipelines to run based on checkbox selections
                pipelines_to_run = []
                if dino_enabled:
                    pipelines_to_run.append("dino")
                if clip_enabled:
                    pipelines_to_run.append("clip")
                if patchcore_enabled:
                    pipelines_to_run.append("patchcore")
                if padim_enabled:
                    pipelines_to_run.append("padim")
                if patchcore_sam_enabled:
                    pipelines_to_run.append("patchcore_sam")
                if patchcore_knn_enabled:
                    pipelines_to_run.append("patchcore_knn")
                
                if not pipelines_to_run:
                    st.error("❌ Please select at least one model to run!")
                    return
                
                output = run_all_pipelines(ref_path, test_path, pipelines_to_run=pipelines_to_run)
                st.session_state.current_output = output
                
            except Exception as e:
                st.error(f"Error running pipeline: {str(e)}")
                # Cleanup on error
                gc.collect()
                return
        
        # Use cached output
        output = st.session_state.current_output
        if output is None:
            st.error("Pipeline execution failed")
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
        
        # Original images
        st.subheader("Step 1: Original Images")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Reference Image (Original)**")
            st.image(output["preprocessing_steps"]["original"][0], width='stretch')
        with col2:
            st.markdown("**Test Image (Original)**")
            st.image(output["preprocessing_steps"]["original"][1], width='stretch')
        
        # After preprocess (resize with center crop, background removal, denoise, gamma correction)
        st.subheader("Step 2: After Preprocessing (Resize, BG Removal, Denoise, Gamma Correction)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Reference Image**")
            st.image(output["preprocessing_steps"]["after_preprocess"][0], width='stretch')
        with col2:
            st.markdown("**Test Image**")
            st.image(output["preprocessing_steps"]["after_preprocess"][1], width='stretch')
        
        # SAM refinement
        st.subheader("Step 3: Rough Mask & SAM Refinement")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Rough Mask (SSIM)**")
            rough_rgb = cv2.cvtColor(output["preprocessed"]["rough_mask"], cv2.COLOR_GRAY2RGB)
            st.image(rough_rgb, width='stretch')
        
        with col2:
            st.markdown("**Refined Mask (SAM)**")
            refined_rgb = cv2.cvtColor(output["preprocessed"]["refined_mask"], cv2.COLOR_GRAY2RGB)
            st.image(refined_rgb, width='stretch')
        
        with col3:
            st.markdown("**Final Processed Images**")
            st.image(output["preprocessing_steps"]["after_sam_refinement"][0], width='stretch')
        
        st.markdown("---")
        
        # Display all pipeline results
        st.header("🔬 Pipeline Comparison")
        
        # SEMANTIC PIPELINES
        if dino_enabled or clip_enabled:
            st.subheader("🎨 Semantic Pipelines")
            semantic_cols = []
            if dino_enabled:
                semantic_cols.append(("DINO", output["results"].get("dino")))
            if clip_enabled:
                semantic_cols.append(("CLIP", output["results"].get("clip")))
            
            if semantic_cols:
                col1, col2 = st.columns(len(semantic_cols)) if len(semantic_cols) == 2 else (st.columns(1)[0], None)
                for idx, (pipeline_name, result) in enumerate(semantic_cols):
                    if len(semantic_cols) == 2 and idx == 0:
                        display_pipeline_result(pipeline_name, result, col1)
                    elif len(semantic_cols) == 2 and idx == 1:
                        display_pipeline_result(pipeline_name, result, col2)
                    elif len(semantic_cols) == 1:
                        display_pipeline_result(pipeline_name, result, col1 if idx == 0 else col2)
            
            st.markdown("---")
        
        # ANOMALY PIPELINES
        if patchcore_enabled or padim_enabled:
            st.subheader("⚠️ Anomaly Pipelines")
            anomaly_cols = []
            if patchcore_enabled:
                anomaly_cols.append(("PatchCore", output["results"].get("patchcore")))
            if padim_enabled:
                anomaly_cols.append(("PaDiM", output["results"].get("padim")))
            
            if anomaly_cols:
                col1, col2 = st.columns(len(anomaly_cols)) if len(anomaly_cols) == 2 else (st.columns(1)[0], None)
                for idx, (pipeline_name, result) in enumerate(anomaly_cols):
                    if len(anomaly_cols) == 2 and idx == 0:
                        display_pipeline_result(pipeline_name, result, col1)
                    elif len(anomaly_cols) == 2 and idx == 1:
                        display_pipeline_result(pipeline_name, result, col2)
                    elif len(anomaly_cols) == 1:
                        display_pipeline_result(pipeline_name, result, col1 if idx == 0 else col2)
            
            st.markdown("---")
        
        # HYBRID PIPELINE
        if patchcore_sam_enabled:
            st.subheader("🔬 Hybrid Pipeline (PatchCore + SAM)")
            if "patchcore_sam" in output["results"] and output["results"]["patchcore_sam"] is not None:
                col1 = st.columns(1)[0]
                display_pipeline_result("PatchCore + SAM", output["results"]["patchcore_sam"], col1)
            else:
                st.warning("⚠️ PatchCore + SAM pipeline result not available")
            
            st.markdown("---")
        
        # ADVANCED PIPELINE (v3.0)
        if patchcore_knn_enabled:
            st.subheader("🚀 Advanced Pipeline (PatchCore KNN - FrameShift v3.0)")
            if "patchcore_knn" in output["results"] and output["results"]["patchcore_knn"] is not None:
                col1 = st.columns(1)[0]
                display_pipeline_result("PatchCore KNN (DINOv2 + KNN)", output["results"]["patchcore_knn"], col1)
            else:
                st.warning("⚠️ PatchCore KNN pipeline result not available")
            
            st.markdown("---")
        
        # Manual selection
        st.header("✅ Select Top 2 Pipelines")
        st.markdown("Based on visual inspection, select the 2 best-performing pipelines:")
        
        available_pipelines = []
        if dino_enabled and "dino" in output["results"] and output["results"]["dino"] is not None:
            available_pipelines.append("DINO")
        if clip_enabled and "clip" in output["results"] and output["results"]["clip"] is not None:
            available_pipelines.append("CLIP")
        if patchcore_enabled and "patchcore" in output["results"] and output["results"]["patchcore"] is not None:
            available_pipelines.append("PatchCore")
        if padim_enabled and "padim" in output["results"] and output["results"]["padim"] is not None:
            available_pipelines.append("PaDiM")
        if patchcore_sam_enabled and "patchcore_sam" in output["results"] and output["results"]["patchcore_sam"] is not None:
            available_pipelines.append("PatchCore+SAM")
        if patchcore_knn_enabled and "patchcore_knn" in output["results"] and output["results"]["patchcore_knn"] is not None:
            available_pipelines.append("PatchCore KNN")
        
        if not available_pipelines:
            st.warning("⚠️ No pipelines were executed successfully")
        else:
            num_cols = min(len(available_pipelines), 5)
            cols = st.columns(num_cols)
            
            dino_selected = False
            clip_selected = False
            patchcore_selected = False
            padim_selected = False
            patchcore_sam_selected = False
            patchcore_knn_selected = False
            
            col_idx = 0
            if "DINO" in available_pipelines:
                with cols[col_idx]:
                    dino_selected = st.checkbox("DINO", key="select_dino")
                col_idx += 1
            if "CLIP" in available_pipelines:
                with cols[col_idx]:
                    clip_selected = st.checkbox("CLIP", key="select_clip")
                col_idx += 1
            if "PatchCore" in available_pipelines:
                with cols[col_idx]:
                    patchcore_selected = st.checkbox("PatchCore", key="select_patchcore")
                col_idx += 1
            if "PaDiM" in available_pipelines:
                with cols[col_idx]:
                    padim_selected = st.checkbox("PaDiM", key="select_padim")
                col_idx += 1
            if "PatchCore+SAM" in available_pipelines:
                with cols[col_idx]:
                    patchcore_sam_selected = st.checkbox("PatchCore+SAM", key="select_patchcore_sam")
                col_idx += 1
            if "PatchCore KNN" in available_pipelines:
                with cols[min(col_idx, len(cols)-1)]:
                    patchcore_knn_selected = st.checkbox("PatchCore KNN", key="select_patchcore_knn")
            
            selected_pipelines = []
            if dino_selected:
                selected_pipelines.append("DINO")
            if clip_selected:
                selected_pipelines.append("CLIP")
            if patchcore_selected:
                selected_pipelines.append("PatchCore")
            if padim_selected:
                selected_pipelines.append("PaDiM")
            if patchcore_sam_selected:
                selected_pipelines.append("PatchCore+SAM")
            if patchcore_knn_selected:
                selected_pipelines.append("PatchCore KNN")
            
            if len(selected_pipelines) == 2:
                st.success(f"✅ Selected pipelines: {', '.join(selected_pipelines)}")
            elif len(selected_pipelines) > 2:
                st.warning("⚠️ Please select exactly 2 pipelines")
            else:
                st.info("ℹ️ Please select 2 pipelines for final comparison")
        
        # Clean up temp files and memory
        if uploaded_ref is not None and uploaded_test is not None:
            if os.path.exists("temp_ref.jpg"):
                os.remove("temp_ref.jpg")
            if os.path.exists("temp_test.jpg"):
                os.remove("temp_test.jpg")
        
        # Cleanup memory periodically
        gc.collect()
    
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
