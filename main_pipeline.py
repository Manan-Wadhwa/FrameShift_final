"""
Main Pipeline Executor for F1 Visual Difference Engine
Orchestrates all 4 pipelines and routing logic
"""
import cv2
import numpy as np
from utils.preprocess import preprocess, resize_image
from utils.rough_mask import generate_rough_mask
from utils.sam_refine import sam_refine
from utils.routing_features import compute_routing_features, route_prediction
from pipelines.semantic_dino import run_dino_pipeline
from pipelines.semantic_clip import run_clip_pipeline
from pipelines.anomaly_patchcore import run_patchcore_pipeline
from pipelines.anomaly_padim import run_padim_pipeline
from pipelines.anomaly_patchcore_sam import run_patchcore_sam_pipeline
from pipelines.anomaly_patchcore_knn import run_patchcore_knn_pipeline


def run_all_pipelines(image_a_path, image_b_path, pipelines_to_run=None):
    """
    Complete pipeline execution for F1 Visual Difference Engine
    
    Args:
        image_a_path: Path to reference image
        image_b_path: Path to test image
        pipelines_to_run: List of pipeline names to run. 
                         If None, runs all: ["dino", "clip", "patchcore", "padim", "patchcore_sam", "patchcore_knn"]
    
    Returns:
        {
            "predicted_type": "semantic" or "anomaly",
            "confidence": "high", "medium", or "low",
            "features": routing feature dict,
            "results": {
                "dino": pipeline result dict,
                "clip": pipeline result dict,
                "patchcore": pipeline result dict,
                "padim": pipeline result dict,
                "patchcore_sam": pipeline result dict,
                "patchcore_knn": pipeline result dict
            }
        }
    """
    # Default to all pipelines if not specified
    if pipelines_to_run is None:
        pipelines_to_run = ["dino", "clip", "patchcore", "padim", "patchcore_sam", "patchcore_knn"]
    print("=" * 60)
    print("F1 VISUAL DIFFERENCE ENGINE")
    print("=" * 60)
    
    # Load images
    print("\n[1/7] Loading images...")
    image_a = cv2.imread(image_a_path)
    image_b = cv2.imread(image_b_path)
    
    if image_a is None or image_b is None:
        raise ValueError("Failed to load one or both images")
    
    # Convert BGR to RGB for display
    image_a_rgb = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB)
    image_b_rgb = cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB)
    
    # Preprocessing
    print("[2/7] Preprocessing (resize with center crop, background removal, denoise, gamma correction)...")
    ref, test = preprocess(image_a, image_b)
    
    # Store preprocessing steps
    preprocessing_steps = {
        "original": (image_a_rgb, image_b_rgb),
        "after_preprocess": (ref.copy(), test.copy()),
    }
    
    # Rough mask generation
    print("[3/7] Generating rough mask using SSIM...")
    rough_mask, diff_map = generate_rough_mask(ref, test)
    
    # SAM refinement
    print("[4/7] Refining mask using SAM...")
    refined_mask = sam_refine(test, rough_mask)
    
    preprocessing_steps["after_sam_refinement"] = (ref.copy(), test.copy(), refined_mask.copy())
    
    # Compute routing features
    print("[5/7] Computing routing features...")
    features = compute_routing_features(ref, test, refined_mask)
    
    print(f"\n   Texture Variance: {features['texture_var']:.2f}")
    print(f"   Edge Density: {features['edge_density']:.4f}")
    print(f"   Entropy: {features['entropy']:.4f}")
    print(f"   Color Shift: {features['color_shift']:.4f}")
    
    # Route prediction
    predicted_type, confidence = route_prediction(features)
    print(f"\n   >> PREDICTED TYPE: {predicted_type.upper()} (confidence: {confidence})")
    
    # Run all 4 pipelines
    print("\n[6/7] Running selected detection pipelines...")
    
    results = {}
    
    if "dino" in pipelines_to_run:
        print("   - Running DINO pipeline (semantic)...")
        results["dino"] = run_dino_pipeline(ref, test, refined_mask)
    
    if "clip" in pipelines_to_run:
        print("   - Running CLIP pipeline (semantic)...")
        results["clip"] = run_clip_pipeline(ref, test, refined_mask)
    
    if "patchcore" in pipelines_to_run:
        print("   - Running PatchCore pipeline (anomaly)...")
        results["patchcore"] = run_patchcore_pipeline(test, refined_mask, ref_img=ref)
    
    if "padim" in pipelines_to_run:
        print("   - Running PaDiM pipeline (anomaly)...")
        results["padim"] = run_padim_pipeline(test, refined_mask, ref_img=ref)
    
    if "patchcore_sam" in pipelines_to_run:
        print("   - Running PatchCore + SAM hybrid pipeline (anomaly)...")
        try:
            results["patchcore_sam"] = run_patchcore_sam_pipeline(test, refined_mask, ref_img=ref)
        except Exception as e:
            print(f"Warning: PatchCore + SAM pipeline failed: {e}")
            print("Skipping hybrid pipeline")
            results["patchcore_sam"] = None
    
    if "patchcore_knn" in pipelines_to_run:
        print("   - Running PatchCore KNN pipeline (DINOv2 + KNN, FrameShift v3.0 approach)...")
        try:
            results["patchcore_knn"] = run_patchcore_knn_pipeline(test, refined_mask, ref_img=ref, sensitivity="medium")
        except Exception as e:
            print(f"Warning: PatchCore KNN pipeline failed: {e}")
            print("Skipping PatchCore KNN pipeline")
            results["patchcore_knn"] = None
    
    # Generate LLaVA reports
    print("\n[7/7] Generating analysis reports...")
    
    # Use rule-based reports (no LLaVA model download)
    for pipeline_key in results.keys():
        if results[pipeline_key] is not None:
            pipeline_name = pipeline_key.upper()
            if "dino" in pipeline_key.lower():
                pipeline_name = "DINOv2 Semantic Detection"
            elif "clip" in pipeline_key.lower():
                pipeline_name = "CLIP Semantic Detection"
            elif "patchcore" in pipeline_key.lower() and "knn" not in pipeline_key.lower() and "sam" not in pipeline_key.lower():
                pipeline_name = "PatchCore Anomaly Detection"
            elif "padim" in pipeline_key.lower():
                pipeline_name = "PaDiM Anomaly Detection"
            elif "patchcore_sam" in pipeline_key.lower():
                pipeline_name = "PatchCore + SAM Hybrid Detection"
            elif "patchcore_knn" in pipeline_key.lower():
                pipeline_name = "PatchCore KNN (DINOv2 + KNN, FrameShift v3.0)"
            
            print(f"   - Generating report for {pipeline_name}...")
            
            # Generate rule-based report (fast, no model downloads)
            from llava.llava_report import generate_rule_based_report
            
            if "semantic" in pipeline_name.lower():
                pipeline_type = "semantic"
            else:
                pipeline_type = "anomaly"
            
            results[pipeline_key]["report"] = generate_rule_based_report(
                pipeline_type=pipeline_type,
                severity=results[pipeline_key].get("severity", None),
                features=features,
                mask_area=np.sum(results[pipeline_key].get("mask_final", np.array([])) > 0),
                pipeline_name=pipeline_name
            )
    
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 60)
    
    return {
        "predicted_type": predicted_type,
        "confidence": confidence,
        "features": features,
        "results": results,
        "preprocessed": {
            "ref": ref,
            "test": test,
            "rough_mask": rough_mask,
            "refined_mask": refined_mask
        },
        "preprocessing_steps": preprocessing_steps
    }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python main_pipeline.py <image_a_path> <image_b_path>")
        sys.exit(1)
    
    image_a_path = sys.argv[1]
    image_b_path = sys.argv[2]
    
    output = run_all_pipelines(image_a_path, image_b_path)
    
    print(f"\nPredicted Type: {output['predicted_type']}")
    print(f"Confidence: {output['confidence']}")
    print(f"\nFeatures: {output['features']}")
