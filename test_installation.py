"""
Test script to verify F1 Visual Difference Engine installation
Run this to check if all components are working correctly
"""
import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("="*60)
    print("Testing Package Imports...")
    print("="*60)
    
    tests = [
        ("OpenCV", "import cv2"),
        ("NumPy", "import numpy as np"),
        ("Matplotlib", "import matplotlib.pyplot as plt"),
        ("scikit-image", "from skimage.metrics import structural_similarity"),
        ("scipy", "import scipy"),
        ("PIL", "from PIL import Image"),
        ("PyTorch", "import torch"),
        ("torchvision", "import torchvision"),
    ]
    
    passed = 0
    failed = 0
    
    for name, import_cmd in tests:
        try:
            exec(import_cmd)
            print(f"✅ {name:<20} - OK")
            passed += 1
        except ImportError as e:
            print(f"❌ {name:<20} - FAILED: {e}")
            failed += 1
    
    print(f"\n{passed}/{len(tests)} core packages available")
    return failed == 0


def test_optional_imports():
    """Test optional advanced models"""
    print("\n" + "="*60)
    print("Testing Optional Models...")
    print("="*60)
    
    # SAM
    try:
        from segment_anything import sam_model_registry
        print("✅ SAM (Segment Anything) - Available")
    except ImportError:
        print("⚠️  SAM - Not installed (will use morphological fallback)")
    
    # CLIP
    try:
        import clip
        print("✅ CLIP - Available")
    except ImportError:
        print("⚠️  CLIP - Not installed (will use color difference fallback)")
    
    # LLaVA dependencies
    try:
        from transformers import LlavaNextProcessor
        print("✅ LLaVA (transformers) - Available")
    except ImportError:
        print("⚠️  LLaVA - Not installed (will use rule-based reports)")


def test_project_structure():
    """Check if all required files and directories exist"""
    print("\n" + "="*60)
    print("Testing Project Structure...")
    print("="*60)
    
    required_items = [
        ("utils/", "dir"),
        ("pipelines/", "dir"),
        ("llava/", "dir"),
        ("demo/", "dir"),
        ("samples/", "dir"),
        ("main_pipeline.py", "file"),
        ("requirements.txt", "file"),
        ("README.md", "file"),
        ("utils/preprocess.py", "file"),
        ("utils/rough_mask.py", "file"),
        ("utils/sam_refine.py", "file"),
        ("utils/routing_features.py", "file"),
        ("utils/visualization.py", "file"),
        ("pipelines/semantic_dino.py", "file"),
        ("pipelines/semantic_clip.py", "file"),
        ("pipelines/anomaly_patchcore.py", "file"),
        ("pipelines/anomaly_padim.py", "file"),
        ("llava/llava_report.py", "file"),
        ("demo/streamlit_app.py", "file"),
        ("demo/app.ipynb", "file"),
    ]
    
    passed = 0
    failed = 0
    
    for item, item_type in required_items:
        if item_type == "dir":
            exists = os.path.isdir(item)
        else:
            exists = os.path.isfile(item)
        
        if exists:
            print(f"✅ {item}")
            passed += 1
        else:
            print(f"❌ {item} - MISSING")
            failed += 1
    
    print(f"\n{passed}/{len(required_items)} required items found")
    return failed == 0


def test_sample_images():
    """Check if sample images exist"""
    print("\n" + "="*60)
    print("Testing Sample Images...")
    print("="*60)
    
    samples = [
        "samples/back1.jpeg",
        "samples/back2.jpeg",
        "samples/side1.jpeg",
        "samples/side2.jpeg",
        "samples/crack1.jpg",
        "samples/crack2.png",
        "samples/copy1.jpeg",
        "samples/copy2.jpeg",
    ]
    
    passed = 0
    for sample in samples:
        if os.path.isfile(sample):
            print(f"✅ {sample}")
            passed += 1
        else:
            print(f"❌ {sample} - MISSING")
    
    print(f"\n{passed}/{len(samples)} sample images found")
    return passed == len(samples)


def test_basic_functionality():
    """Test basic preprocessing functionality"""
    print("\n" + "="*60)
    print("Testing Basic Functionality...")
    print("="*60)
    
    try:
        import cv2
        import numpy as np
        from utils.preprocess import resize_image, median_blur, gamma_correct
        
        # Create test image
        test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Test resize
        resized = resize_image(test_img, (128, 128))
        assert resized.shape == (128, 128, 3), "Resize failed"
        print("✅ Image resize - OK")
        
        # Test median blur
        blurred = median_blur(test_img, k=3)
        assert blurred.shape == test_img.shape, "Median blur failed"
        print("✅ Median blur - OK")
        
        # Test gamma correction
        corrected = gamma_correct(test_img, gamma=1.2)
        assert corrected.shape == test_img.shape, "Gamma correction failed"
        print("✅ Gamma correction - OK")
        
        print("\n✅ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Functionality test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("F1 VISUAL DIFFERENCE ENGINE - SYSTEM TEST")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Package Imports", test_imports()))
    test_optional_imports()
    results.append(("Project Structure", test_project_structure()))
    results.append(("Sample Images", test_sample_images()))
    results.append(("Basic Functionality", test_basic_functionality()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL CRITICAL TESTS PASSED!")
        print("="*60)
        print("\nYou can now run:")
        print("  • streamlit run demo/streamlit_app.py")
        print("  • jupyter notebook demo/app.ipynb")
        print("  • python main_pipeline.py samples/back1.jpeg samples/back2.jpeg")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("="*60)
        print("\nPlease:")
        print("  1. Run: pip install -r requirements.txt")
        print("  2. Check that all files are present")
        print("  3. Verify sample images are in samples/ folder")
    
    print("\n")
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
