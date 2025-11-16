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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars

# Add parent directory to path to import main_pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))
from main_pipeline import run_all_pipelines
from utils.heatmap_metrics import HeatmapMetrics
from utils.gemini_analyzer import GeminiAnalyzer
import plotly.graph_objects as go
import plotly.express as px

# Configure Streamlit for memory efficiency
st.set_page_config(
    page_title="F1 Visual Difference Engine",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)


def display_comprehensive_metrics(metrics, ai_analysis=None):
    """Display comprehensive metrics in Streamlit with charts and visualizations"""
    
    st.subheader("📊 Comprehensive Metrics Dashboard")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Anomaly Coverage", f"{metrics['anomaly_percentage']:.2f}%")
    with col2:
        st.metric("Affected Pixels", f"{metrics['total_anomaly_pixels']:,}")
    with col3:
        st.metric("Detected Regions", f"{metrics['num_regions']}")
    with col4:
        st.metric("Max Intensity", f"{metrics['max_anomaly_score']:.2f}")
    
    # Severity Distribution Chart
    st.markdown("### ⚠️ Severity Distribution")
    severity_data = {
        'Level': ['Low', 'Medium', 'High'],
        'Pixels': [
            metrics['low_severity_pixels'],
            metrics['medium_severity_pixels'],
            metrics['high_severity_pixels']
        ],
        'Color': ['#28a745', '#ffc107', '#dc3545']
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=severity_data['Level'],
            y=severity_data['Pixels'],
            marker_color=severity_data['Color'],
            text=[f"{p:,}" for p in severity_data['Pixels']],
            textposition='outside'
        )
    ])
    fig.update_layout(
        title="Pixel Count by Severity Level",
        xaxis_title="Severity",
        yaxis_title="Pixels",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Spatial Distribution Heatmap
    st.markdown("### 🗺️ Spatial Distribution (3x3 Grid)")
    spatial_dist = metrics.get('spatial_distribution', {})
    if spatial_dist:
        grid_data = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                key = f'cell_{i}_{j}'
                grid_data[i, j] = spatial_dist.get(key, 0) * 100
        
        fig = go.Figure(data=go.Heatmap(
            z=grid_data,
            text=[[f'{val:.1f}%' for val in row] for row in grid_data],
            texttemplate='%{text}',
            textfont={"size": 16},
            colorscale='Reds',
            showscale=True
        ))
        fig.update_layout(
            title="Anomaly Density Across Image Regions",
            height=400,
            xaxis={'title': 'Column', 'tickvals': [0, 1, 2]},
            yaxis={'title': 'Row', 'tickvals': [0, 1, 2]}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top Regions Table
    st.markdown("### 🎯 Top Anomaly Regions")
    regions = metrics.get('regions', [])
    if regions:
        sorted_regions = sorted(regions, key=lambda r: r['area'], reverse=True)[:10]
        region_data = []
        for i, region in enumerate(sorted_regions, 1):
            x, y, w, h = region['bbox']
            region_data.append({
                'Rank': i,
                'Area (px)': f"{region['area']:,}",
                'Bounding Box': f"({x}, {y}, {w}, {h})",
                'Max Intensity': f"{region['max_intensity']:.3f}",
                'Mean Intensity': f"{region['mean_intensity']:.3f}"
            })
        
        import pandas as pd
        df = pd.DataFrame(region_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Coverage Comparison
    st.markdown("### 🔄 Coverage Comparison")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("A → B Coverage", f"{metrics.get('coverage_a_to_b', 0):.2f}%")
    with col2:
        st.metric("B → A Coverage", f"{metrics.get('coverage_b_to_a', 0):.2f}%")
    with col3:
        st.metric("Coverage Difference", f"{metrics.get('coverage_difference', 0):.2f}%")
    
    # AI Analysis Section
    if ai_analysis:
        st.markdown("---")
        st.markdown("### 🤖 AI-Powered Structural Analysis")
        
        # Summary
        st.info(f"**Summary:** {ai_analysis.get('summary', 'No summary available')}")
        
        # Structural Issues
        issues = ai_analysis.get('structural_issues', [])
        if issues:
            st.markdown(f"#### ⚠️ Detected {len(issues)} Structural Issue(s)")
            
            for i, issue in enumerate(issues, 1):
                severity = issue.get('severity', 'low').lower()
                
                # Color-code by severity
                if severity == 'critical' or severity == 'high':
                    color = '🔴'
                elif severity == 'medium':
                    color = '🟡'
                else:
                    color = '🟢'
                
                with st.expander(f"{color} Issue {i}: {issue.get('component', 'Unknown Component')} - {severity.upper()}"):
                    st.markdown(f"**Issue Type:** {issue.get('issue_type', 'Unknown')}")
                    st.markdown(f"**Location:** {issue.get('location', 'Not specified')}")
                    st.markdown(f"**Description:** {issue.get('description', 'No description')}")
                    st.markdown(f"**Safety Impact:** {issue.get('safety_impact', 'Unknown')}")
                    st.markdown(f"**Recommendation:** {issue.get('recommendation', 'No recommendation')}")
        else:
            st.success("✓ No significant structural issues detected.")
        
        # Critical Actions
        actions = ai_analysis.get('critical_actions', [])
        if actions:
            st.markdown("#### 🔧 Critical Actions Required")
            for action in actions:
                st.warning(f"• {action}")


def display_pipeline_result(pipeline_name, result, col, image_a=None, image_b=None, use_gemini=False):
    """Display results for a single pipeline in a column with comprehensive reporting"""
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
            
            # Generate comprehensive metrics if heatmaps available
            if pipeline_name == "PatchCore + SAM" and image_a is not None and image_b is not None:
                if st.button(f"📊 Show Comprehensive Report", key=f"report_{pipeline_name}"):
                    with st.spinner("Computing comprehensive metrics..."):
                        # Get heatmaps
                        heatmap_a2b = result.get('heatmap_a2b') or result.get('diff_map')
                        heatmap_b2a = result.get('heatmap_b2a') or result.get('diff_map')
                        
                        if heatmap_a2b is not None and heatmap_b2a is not None:
                            # Compute metrics
                            metrics = HeatmapMetrics.compute_comprehensive_metrics(
                                heatmap_a2b,
                                heatmap_b2a,
                                image_a,
                                image_b
                            )
                            
                            # AI Analysis if enabled
                            ai_analysis = None
                            if use_gemini:
                                try:
                                    gemini = GeminiAnalyzer()
                                    union_colored = cv2.applyColorMap(
                                        metrics['union_heatmap'].astype(np.uint8),
                                        cv2.COLORMAP_JET
                                    )
                                    ai_analysis = gemini.analyze_heatmaps(
                                        image_a, image_b, union_colored, metrics
                                    )
                                except Exception as e:
                                    st.warning(f"⚠️ AI analysis failed: {e}")
                            
                            # Display comprehensive metrics
                            display_comprehensive_metrics(metrics, ai_analysis)
            
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
    if 'page' not in st.session_state:
        st.session_state.page = "Multi-Pipeline"
    
    st.title("🏎️ F1 Visual Difference Engine")
    st.markdown("**Multi-Pipeline Visual Change Detection System**")
    
    # Page selector
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔬 Multi-Pipeline", use_container_width=True):
            st.session_state.page = "Multi-Pipeline"
    with col2:
        if st.button("🔄 Image Differencing", use_container_width=True):
            st.session_state.page = "Differencing"
    
    st.markdown("---")
    
    # Route to selected page
    if st.session_state.page == "Differencing":
        import importlib
        differencing_module = importlib.import_module('streamlit_differencing')
        differencing_module.main()
    else:
        # Original multi-pipeline page
        render_multi_pipeline()


def render_multi_pipeline():
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
        
        # AI Analysis Option
        st.header("🤖 AI Analysis")
        use_gemini = st.checkbox("Enable Gemini AI Analysis", value=False, key="gemini_check",
                                 help="Requires GEMINI_API_KEY environment variable")
        
        if use_gemini:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                st.success("✓ Gemini API key detected")
            else:
                st.warning("⚠️ GEMINI_API_KEY not set. AI analysis will be skipped.")
        
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
        
        # Load images as numpy arrays for metrics
        img_a_cv = cv2.imread(ref_path)
        img_a_rgb = cv2.cvtColor(img_a_cv, cv2.COLOR_BGR2RGB) if img_a_cv is not None else None
        img_b_cv = cv2.imread(test_path)
        img_b_rgb = cv2.cvtColor(img_b_cv, cv2.COLOR_BGR2RGB) if img_b_cv is not None else None
        
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
                        display_pipeline_result(pipeline_name, result, col1, img_a_rgb, img_b_rgb, use_gemini)
                    elif len(semantic_cols) == 2 and idx == 1:
                        display_pipeline_result(pipeline_name, result, col2, img_a_rgb, img_b_rgb, use_gemini)
                    elif len(semantic_cols) == 1:
                        display_pipeline_result(pipeline_name, result, col1 if idx == 0 else col2, img_a_rgb, img_b_rgb, use_gemini)
            
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
                        display_pipeline_result(pipeline_name, result, col1, img_a_rgb, img_b_rgb, use_gemini)
                    elif len(anomaly_cols) == 2 and idx == 1:
                        display_pipeline_result(pipeline_name, result, col2, img_a_rgb, img_b_rgb, use_gemini)
                    elif len(anomaly_cols) == 1:
                        display_pipeline_result(pipeline_name, result, col1 if idx == 0 else col2, img_a_rgb, img_b_rgb, use_gemini)
            
            st.markdown("---")
        
        # HYBRID PIPELINE
        if patchcore_sam_enabled:
            st.subheader("🔬 Hybrid Pipeline (PatchCore + SAM)")
            if "patchcore_sam" in output["results"] and output["results"]["patchcore_sam"] is not None:
                col1 = st.columns(1)[0]
                display_pipeline_result("PatchCore + SAM", output["results"]["patchcore_sam"], col1, img_a_rgb, img_b_rgb, use_gemini)
            else:
                st.warning("⚠️ PatchCore + SAM pipeline result not available")
            
            st.markdown("---")
        
        # ADVANCED PIPELINE (v3.0)
        if patchcore_knn_enabled:
            st.subheader("🚀 Advanced Pipeline (PatchCore KNN - FrameShift v3.0)")
            if "patchcore_knn" in output["results"] and output["results"]["patchcore_knn"] is not None:
                col1 = st.columns(1)[0]
                display_pipeline_result("PatchCore KNN (DINOv2 + KNN)", output["results"]["patchcore_knn"], col1, img_a_rgb, img_b_rgb, use_gemini)
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
