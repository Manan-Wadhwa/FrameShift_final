"""
Main Pipeline Executor for F1 Visual Difference Engine
Orchestrates all 4 pipelines and routing logic
"""
import cv2
import numpy as np
from utils.preprocess import preprocess
from utils.rough_mask import generate_rough_mask
from utils.sam_refine import sam_refine
from utils.routing_features import compute_routing_features, route_prediction
from pipelines.semantic_dino import run_dino_pipeline
from pipelines.semantic_clip import run_clip_pipeline
from pipelines.anomaly_patchcore import run_patchcore_pipeline
from pipelines.anomaly_padim import run_padim_pipeline
from llava.llava_report import llava_generate


def run_all_pipelines(image_a_path, image_b_path):
    """
    Complete pipeline execution for F1 Visual Difference Engine
    
    Args:
        image_a_path: Path to reference image
        image_b_path: Path to test image
    
    Returns:
        {
            "predicted_type": "semantic" or "anomaly",
            "confidence": "high", "medium", or "low",
            "features": routing feature dict,
            "results": {
                "dino": pipeline result dict,
                "clip": pipeline result dict,
                "patchcore": pipeline result dict,
                "padim": pipeline result dict
            }
        }
    """
    print("=" * 60)
    print("F1 VISUAL DIFFERENCE ENGINE")
    print("=" * 60)
    
    # Load images
    print("\n[1/7] Loading images...")
    image_a = cv2.imread(image_a_path)
    image_b = cv2.imread(image_b_path)
    
    if image_a is None or image_b is None:
        raise ValueError("Failed to load one or both images")
    
    # Preprocessing
    print("[2/7] Preprocessing (resize, denoise, gamma correction, SIFT alignment)...")
    ref, test = preprocess(image_a, image_b)
    
    # Rough mask generation
    print("[3/7] Generating rough mask using SSIM...")
    rough_mask, diff_map = generate_rough_mask(ref, test)
    
    # SAM refinement
    print("[4/7] Refining mask using SAM...")
    refined_mask = sam_refine(test, rough_mask)
    
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
    print("\n[6/7] Running all 4 detection pipelines...")
    
    results = {}
    
    print("   - Running DINO pipeline (semantic)...")
    results["dino"] = run_dino_pipeline(ref, test, refined_mask)
    
    print("   - Running CLIP pipeline (semantic)...")
    results["clip"] = run_clip_pipeline(ref, test, refined_mask)
    
    print("   - Running PatchCore pipeline (anomaly)...")
    results["patchcore"] = run_patchcore_pipeline(test, refined_mask, ref_img=ref)
    
    print("   - Running PaDiM pipeline (anomaly)...")
    results["padim"] = run_padim_pipeline(test, refined_mask, ref_img=ref)
    
    # Generate LLaVA reports
    print("\n[7/7] Generating analysis reports...")
    
    print("   - Generating report for DINO...")
    results["dino"]["report"] = llava_generate(
        ref, test, 
        results["dino"]["heatmap"], 
        results["dino"]["mask_final"],
        pipeline_type="semantic",
        features=features,
        pipeline_name="DINOv2 Semantic Detection"
    )
    
    print("   - Generating report for CLIP...")
    results["clip"]["report"] = llava_generate(
        ref, test,
        results["clip"]["heatmap"],
        results["clip"]["mask_final"],
        pipeline_type="semantic",
        features=features,
        pipeline_name="CLIP Semantic Detection"
    )
    
    print("   - Generating report for PatchCore...")
    results["patchcore"]["report"] = llava_generate(
        ref, test,
        results["patchcore"]["heatmap"],
        results["patchcore"]["mask_final"],
        pipeline_type="anomaly",
        severity=results["patchcore"].get("severity", 0.0),
        features=features,
        pipeline_name="PatchCore Anomaly Detection"
    )
    
    print("   - Generating report for PaDiM...")
    results["padim"]["report"] = llava_generate(
        ref, test,
        results["padim"]["heatmap"],
        results["padim"]["mask_final"],
        pipeline_type="anomaly",
        severity=results["padim"].get("severity", 0.0),
        features=features,
        pipeline_name="PaDiM Anomaly Detection"
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
        }
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
