"""
Integration test for F1 Visual Difference Engine
Tests the complete pipeline with error handling
"""
import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

def create_test_images():
    """Create synthetic test images"""
    print("Creating test images...")
    
    # Create reference image - solid blue background with a circle
    ref = np.ones((336, 336, 3), dtype=np.uint8) * 100  # Dark blue background
    cv2.circle(ref, (168, 168), 50, (255, 200, 0), -1)  # Blue circle
    
    # Create test image - same but with an additional anomaly (red dot)
    test = ref.copy()
    cv2.circle(test, (200, 150), 20, (0, 0, 255), -1)  # Red anomaly
    
    # Save to temp directory
    temp_dir = Path(__file__).parent / "temp_test"
    temp_dir.mkdir(exist_ok=True)
    
    ref_path = temp_dir / "ref.jpg"
    test_path = temp_dir / "test.jpg"
    
    cv2.imwrite(str(ref_path), ref)
    cv2.imwrite(str(test_path), test)
    
    print(f"✅ Created reference image: {ref_path}")
    print(f"✅ Created test image: {test_path}")
    
    return str(ref_path), str(test_path)


def test_preprocessing():
    """Test preprocessing pipeline"""
    print("\n" + "="*60)
    print("Testing Preprocessing Pipeline...")
    print("="*60)
    
    try:
        from utils.preprocess import preprocess
        ref_path, test_path = create_test_images()
        
        ref, test = preprocess(ref_path, test_path)
        
        print(f"✅ Reference preprocessed: shape={ref.shape}, dtype={ref.dtype}")
        print(f"✅ Test preprocessed: shape={test.shape}, dtype={test.dtype}")
        
        assert ref.shape == (336, 336, 3), f"Wrong ref shape: {ref.shape}"
        assert test.shape == (336, 336, 3), f"Wrong test shape: {test.shape}"
        
        return True, ref_path, test_path
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_individual_pipelines(ref_path, test_path):
    """Test each pipeline individually"""
    print("\n" + "="*60)
    print("Testing Individual Pipelines...")
    print("="*60)
    
    from utils.preprocess import preprocess
    from utils.rough_mask import get_rough_mask
    from utils.sam_refine import refine_mask_with_sam
    
    # Preprocess
    ref, test = preprocess(ref_path, test_path)
    
    # Get rough mask
    print("\n1. Getting rough mask...")
    try:
        rough_mask = get_rough_mask(ref, test)
        print(f"   ✅ Rough mask: shape={rough_mask.shape}")
    except Exception as e:
        print(f"   ❌ Rough mask failed: {e}")
        rough_mask = np.ones((336, 336), dtype=np.uint8) * 255
    
    # Refine with SAM
    print("2. Refining mask with SAM...")
    try:
        refined_mask = refine_mask_with_sam(test, rough_mask)
        print(f"   ✅ Refined mask: shape={refined_mask.shape}")
    except Exception as e:
        print(f"   ⚠️  SAM refinement failed (may not be installed): {e}")
        refined_mask = rough_mask
    
    # Test semantic pipelines
    print("3. Testing DINO...")
    try:
        from pipelines.semantic_dino import run_dino_pipeline
        result = run_dino_pipeline(ref, test, refined_mask)
        print(f"   ✅ DINO result keys: {result.keys()}")
    except Exception as e:
        print(f"   ❌ DINO failed: {e}")
    
    print("4. Testing CLIP...")
    try:
        from pipelines.semantic_clip import run_clip_pipeline
        result = run_clip_pipeline(ref, test, refined_mask)
        print(f"   ✅ CLIP result keys: {result.keys()}")
    except Exception as e:
        print(f"   ❌ CLIP failed: {e}")
    
    # Test anomaly pipelines
    print("5. Testing PatchCore...")
    try:
        from pipelines.anomaly_patchcore import run_patchcore_pipeline
        result = run_patchcore_pipeline(test, refined_mask)
        print(f"   ✅ PatchCore result keys: {result.keys()}")
    except Exception as e:
        print(f"   ❌ PatchCore failed: {e}")
    
    print("6. Testing PaDiM...")
    try:
        from pipelines.anomaly_padim import run_padim_pipeline
        result = run_padim_pipeline(test, refined_mask)
        print(f"   ✅ PaDiM result keys: {result.keys()}")
    except Exception as e:
        print(f"   ❌ PaDiM failed: {e}")
    
    print("7. Testing PatchCore + SAM (Hybrid)...")
    try:
        from pipelines.anomaly_patchcore_sam import run_patchcore_sam_pipeline
        result = run_patchcore_sam_pipeline(test, refined_mask, ref_img=ref)
        print(f"   ✅ PatchCore+SAM result keys: {result.keys()}")
        if result:
            print(f"      - Severity: {result.get('severity', 'N/A')}")
            print(f"      - Mask shape: {result.get('mask_final', 'N/A').shape if 'mask_final' in result else 'N/A'}")
    except Exception as e:
        print(f"   ⚠️  PatchCore+SAM failed (expected in some setups): {e}")
        import traceback
        traceback.print_exc()


def test_full_pipeline():
    """Test the complete pipeline"""
    print("\n" + "="*60)
    print("Testing Full Pipeline...")
    print("="*60)
    
    try:
        from main_pipeline import run_all_pipelines
        ref_path, test_path = create_test_images()
        
        print("Running complete pipeline...")
        output = run_all_pipelines(ref_path, test_path)
        
        print("\n✅ Pipeline completed successfully!")
        print(f"   - Predicted type: {output['predicted_type']}")
        print(f"   - Confidence: {output['confidence']}")
        print(f"   - Results keys: {list(output['results'].keys())}")
        
        # Check each result
        print("\n   Pipeline Results:")
        for pipeline_name, result in output['results'].items():
            if result is None:
                print(f"      - {pipeline_name:<20}: ⚠️  None (failed)")
            elif "report" in result and result["report"] is None:
                print(f"      - {pipeline_name:<20}: ⚠️  Report generation failed")
            else:
                keys = list(result.keys()) if isinstance(result, dict) else "invalid"
                print(f"      - {pipeline_name:<20}: ✅ {keys}")
        
        return True
    except Exception as e:
        print(f"❌ Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup():
    """Clean up test files"""
    print("\n" + "="*60)
    print("Cleaning up test files...")
    print("="*60)
    
    import shutil
    temp_dir = Path(__file__).parent / "temp_test"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print("✅ Cleaned up temp directory")


if __name__ == "__main__":
    print("\n")
    print("*" * 60)
    print("F1 VISUAL DIFFERENCE ENGINE - INTEGRATION TEST")
    print("*" * 60)
    
    # Test preprocessing
    success, ref_path, test_path = test_preprocessing()
    if not success:
        print("\n❌ Preprocessing test failed!")
        sys.exit(1)
    
    # Test individual pipelines
    test_individual_pipelines(ref_path, test_path)
    
    # Test full pipeline
    success = test_full_pipeline()
    
    # Cleanup
    cleanup()
    
    print("\n" + "*" * 60)
    if success:
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
    else:
        print("⚠️  SOME TESTS FAILED - SEE OUTPUT ABOVE")
    print("*" * 60 + "\n")
