"""
Quick test of advanced models
"""
import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipelines.anomaly_patchcore_sam import run_patchcore_sam_pipeline
from pipelines.anomaly_patchcore_knn import run_patchcore_knn_pipeline
from utils.preprocess import preprocess

# Load sample images
ref_path = "samples/back1.jpeg"
test_path = "samples/back2.jpeg"

print("Loading images...")
ref_img = cv2.imread(ref_path)
test_img = cv2.imread(test_path)

if ref_img is None or test_img is None:
    print("ERROR: Could not load sample images")
    sys.exit(1)

print(f"Original shapes: ref={ref_img.shape}, test={test_img.shape}")

# Preprocess
print("\nPreprocessing...")
ref_processed, test_processed = preprocess(ref_img, test_img)
print(f"Processed shapes: ref={ref_processed.shape}, test={test_processed.shape}")

# Create dummy refined mask
refined_mask = np.ones((test_processed.shape[0], test_processed.shape[1]), dtype=np.uint8) * 255

print("\n" + "="*60)
print("Testing PatchCore + SAM Pipeline...")
print("="*60)
try:
    result_sam = run_patchcore_sam_pipeline(test_processed, refined_mask, ref_img=ref_processed)
    if result_sam:
        print("✓ PatchCore + SAM SUCCESS")
        print(f"  - heatmap shape: {result_sam['heatmap'].shape}")
        print(f"  - mask_final shape: {result_sam['mask_final'].shape}")
        print(f"  - severity: {result_sam.get('severity', 'N/A')}")
    else:
        print("✗ PatchCore + SAM returned None")
except Exception as e:
    print(f"✗ PatchCore + SAM FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Testing PatchCore KNN Pipeline...")
print("="*60)
try:
    result_knn = run_patchcore_knn_pipeline(test_processed, refined_mask, ref_img=ref_processed, sensitivity="medium")
    if result_knn:
        print("✓ PatchCore KNN SUCCESS")
        print(f"  - heatmap shape: {result_knn['heatmap'].shape}")
        print(f"  - mask_final shape: {result_knn['mask_final'].shape}")
        print(f"  - severity: {result_knn.get('severity', 'N/A')}")
    else:
        print("✗ PatchCore KNN returned None")
except Exception as e:
    print(f"✗ PatchCore KNN FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Test Complete")
print("="*60)
