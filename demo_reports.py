"""
Quick demonstration of PatchCore + SAM reporting system.
This script shows all features in action.

Run this to test the system without actual images.
"""

import numpy as np
import cv2
from pathlib import Path


def create_sample_images():
    """Create synthetic sample images for testing."""
    print("🎨 Creating synthetic sample images...")
    
    # Create base image (simulated F1 car)
    h, w = 600, 800
    base = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Add some patterns
    base[100:500, 100:300] = [50, 50, 150]  # "Body"
    base[200:400, 400:700] = [150, 50, 50]  # "Wing"
    cv2.circle(base, (150, 450), 60, (80, 80, 80), -1)  # "Wheel"
    cv2.circle(base, (650, 450), 60, (80, 80, 80), -1)  # "Wheel"
    
    # Add some noise
    noise = np.random.randint(0, 30, (h, w, 3), dtype=np.uint8)
    image_a = cv2.add(base, noise)
    
    # Create modified image (with "damage")
    image_b = image_a.copy()
    
    # Simulate tire crack
    cv2.line(image_b, (130, 440), (160, 470), (200, 200, 200), 3)
    cv2.line(image_b, (135, 445), (165, 475), (220, 220, 220), 2)
    
    # Simulate wing deformation
    pts = np.array([[420, 250], [450, 260], [480, 270], [450, 280]], np.int32)
    cv2.fillPoly(image_b, [pts], (180, 70, 70))
    
    # Add more noise to damaged area
    damage_noise = np.random.randint(0, 50, (100, 100, 3), dtype=np.uint8)
    image_b[400:500, 400:500] = cv2.add(image_b[400:500, 400:500], damage_noise)
    
    print("✓ Sample images created")
    return image_a, image_b


def demo_basic_reports():
    """Demonstrate basic reporting without AI."""
    print("\n" + "="*80)
    print("📊 DEMO 1: BASIC REPORTS (No AI)")
    print("="*80)
    
    from pipelines.anomaly_patchcore_sam import run_patchcore_sam_pipeline
    
    # Create sample images
    image_a, image_b = create_sample_images()
    
    # Run pipeline
    print("\n🔬 Running PatchCore + SAM pipeline...")
    result = run_patchcore_sam_pipeline(
        test_img=image_b,
        ref_img=image_a,
        generate_reports=True,
        output_dir='demo_reports_basic/',
        use_gemini=False
    )
    
    if result and 'metrics' in result:
        m = result['metrics']
        print("\n📊 Results:")
        print(f"   Anomaly Coverage: {m['anomaly_percentage']:.2f}%")
        print(f"   Total Regions: {m['num_regions']}")
        print(f"   High Severity Pixels: {m['high_severity_pixels']:,}")
        
        if 'reports' in result:
            print("\n📁 Generated Reports:")
            for name, path in result['reports'].items():
                print(f"   {name}: {path}")
    
    print("\n✅ Basic demo complete!")
    return result


def demo_ai_reports():
    """Demonstrate AI-powered reporting."""
    print("\n" + "="*80)
    print("🤖 DEMO 2: AI-POWERED REPORTS (Gemini)")
    print("="*80)
    
    import os
    
    # Check if API key is set
    if not os.getenv('GEMINI_API_KEY'):
        print("\n⚠️  GEMINI_API_KEY not set!")
        print("   This demo requires Gemini API access.")
        print("\n   To enable:")
        print("   1. Get API key: https://aistudio.google.com/app/apikey")
        print("   2. Set environment variable:")
        print('      $env:GEMINI_API_KEY="your-api-key-here"')
        print("   3. Run: python setup_gemini.py")
        print("\n   Skipping AI demo...")
        return None
    
    from pipelines.anomaly_patchcore_sam import run_patchcore_sam_pipeline
    
    # Create sample images
    image_a, image_b = create_sample_images()
    
    # Run pipeline with AI
    print("\n🔬 Running PatchCore + SAM pipeline with Gemini AI...")
    result = run_patchcore_sam_pipeline(
        test_img=image_b,
        ref_img=image_a,
        generate_reports=True,
        output_dir='demo_reports_ai/',
        use_gemini=True
    )
    
    if result:
        if 'metrics' in result:
            m = result['metrics']
            print("\n📊 Metrics:")
            print(f"   Anomaly Coverage: {m['anomaly_percentage']:.2f}%")
            print(f"   Total Regions: {m['num_regions']}")
        
        if 'ai_analysis' in result:
            ai = result['ai_analysis']
            print("\n🤖 AI Analysis:")
            print(f"   Summary: {ai.get('summary', 'N/A')[:100]}...")
            
            issues = ai.get('structural_issues', [])
            if issues:
                print(f"\n   Detected {len(issues)} issue(s):")
                for i, issue in enumerate(issues[:3], 1):  # Show first 3
                    print(f"   {i}. {issue.get('component', 'Unknown')}: {issue.get('issue_type', 'Unknown')}")
        
        if 'reports' in result:
            print("\n📁 Generated Reports:")
            for name, path in result['reports'].items():
                print(f"   {name}: {path}")
    
    print("\n✅ AI demo complete!")
    return result


def demo_metrics_only():
    """Demonstrate metrics calculation only."""
    print("\n" + "="*80)
    print("📐 DEMO 3: METRICS CALCULATION")
    print("="*80)
    
    from utils.heatmap_metrics import HeatmapMetrics
    
    # Create sample heatmaps
    print("\n🎨 Creating sample heatmaps...")
    h, w = 256, 256
    
    # Heatmap A→B (gradient with some hot spots)
    heatmap_a2b = np.zeros((h, w), dtype=np.float32)
    heatmap_a2b[50:150, 50:150] = 0.8  # High severity region
    heatmap_a2b[100:200, 150:220] = 0.5  # Medium severity
    heatmap_a2b = cv2.GaussianBlur(heatmap_a2b, (21, 21), 0)
    heatmap_a2b = (heatmap_a2b * 255).astype(np.uint8)
    
    # Heatmap B→A (different pattern)
    heatmap_b2a = np.zeros((h, w), dtype=np.float32)
    heatmap_b2a[80:180, 100:200] = 0.6  # Medium severity
    heatmap_b2a[150:200, 50:100] = 0.9  # High severity
    heatmap_b2a = cv2.GaussianBlur(heatmap_b2a, (21, 21), 0)
    heatmap_b2a = (heatmap_b2a * 255).astype(np.uint8)
    
    # Compute metrics
    print("📊 Computing comprehensive metrics...")
    metrics = HeatmapMetrics.compute_comprehensive_metrics(
        heatmap_a2b,
        heatmap_b2a
    )
    
    print("\n📈 Results:")
    print(f"   Union Method: Maximum")
    print(f"   Anomaly Coverage: {metrics['anomaly_percentage']:.2f}%")
    print(f"   Detected Regions: {metrics['num_regions']}")
    print(f"   Severity Breakdown:")
    print(f"      Low:    {metrics['low_severity_pixels']:,} pixels")
    print(f"      Medium: {metrics['medium_severity_pixels']:,} pixels")
    print(f"      High:   {metrics['high_severity_pixels']:,} pixels")
    print(f"   Coverage Difference: {metrics['coverage_difference']:.2f}%")
    
    # Test different union methods
    print("\n🔀 Testing union methods:")
    for method in ['max', 'mean', 'weighted']:
        union = HeatmapMetrics.compute_union_heatmap(heatmap_a2b, heatmap_b2a, method=method)
        coverage = np.sum(union > 0.3 * 255) / (h * w) * 100
        print(f"   {method:10s}: {coverage:.2f}% coverage")
    
    print("\n✅ Metrics demo complete!")
    return metrics


def main():
    """Run all demos."""
    print("="*80)
    print("🚀 PATCHCORE + SAM REPORTING SYSTEM - COMPREHENSIVE DEMO")
    print("="*80)
    print()
    print("This demo will:")
    print("  1. Create synthetic F1 car images with simulated damage")
    print("  2. Run PatchCore + SAM anomaly detection")
    print("  3. Generate comprehensive reports")
    print("  4. (Optional) Use Gemini AI for structural analysis")
    print()
    
    try:
        # Demo 1: Basic reports
        demo_basic_reports()
        
        # Demo 2: Metrics calculation
        demo_metrics_only()
        
        # Demo 3: AI reports (if API key available)
        demo_ai_reports()
        
        print("\n" + "="*80)
        print("✅ ALL DEMOS COMPLETE")
        print("="*80)
        print()
        print("📁 Check the following directories:")
        print("   - demo_reports_basic/  (Basic metrics)")
        print("   - demo_reports_ai/     (AI analysis, if Gemini key was set)")
        print()
        print("📖 Next steps:")
        print("   1. Review generated HTML reports in your browser")
        print("   2. Check metrics_visualization.jpg for visual overview")
        print("   3. See REPORTS_README.md for detailed documentation")
        print("   4. See GEMINI_SETUP.md for AI analysis setup")
        print()
        print("🎯 Try with real images:")
        print("   python example_patchcore_sam_reports.py \\")
        print("       --image-a your_ref.jpg \\")
        print("       --image-b your_test.jpg \\")
        print("       --output-dir reports/ \\")
        print("       --use-gemini")
        print()
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
