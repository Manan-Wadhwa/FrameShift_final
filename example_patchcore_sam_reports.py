"""
Example script demonstrating PatchCore + SAM with comprehensive reporting and Gemini AI analysis.

This script shows how to:
1. Run PatchCore + SAM anomaly detection
2. Generate bidirectional heatmaps (A→B and B→A)
3. Compute union, intersection, and difference heatmaps
4. Calculate comprehensive metrics (severity levels, regions, spatial distribution)
5. Generate basic reports (JSON, TXT, HTML)
6. Use Gemini AI for structural analysis (tire cracks, wing damage, etc.)

Usage:
    # Without AI analysis:
    python example_patchcore_sam_reports.py --image-a samples/ref.jpg --image-b samples/test.jpg --output-dir reports/

    # With Gemini AI analysis:
    export GEMINI_API_KEY="your-api-key-here"
    python example_patchcore_sam_reports.py --image-a samples/ref.jpg --image-b samples/test.jpg --output-dir reports/ --use-gemini
"""

import cv2
import argparse
from pathlib import Path
from pipelines.anomaly_patchcore_sam import run_patchcore_sam_pipeline


def main():
    parser = argparse.ArgumentParser(description='PatchCore + SAM with comprehensive reporting')
    parser.add_argument('--image-a', type=str, required=True, help='Reference image (Image A)')
    parser.add_argument('--image-b', type=str, required=True, help='Test image (Image B)')
    parser.add_argument('--output-dir', type=str, default='reports/', help='Output directory for reports')
    parser.add_argument('--use-gemini', action='store_true', help='Use Gemini AI for structural analysis')
    parser.add_argument('--visualize', action='store_true', help='Display results in window')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🏎️  F1 PATCHCORE + SAM ANOMALY DETECTION WITH COMPREHENSIVE REPORTING")
    print("="*80)
    
    # Load images
    print(f"\n📂 Loading images...")
    image_a = cv2.imread(args.image_a)
    image_b = cv2.imread(args.image_b)
    
    if image_a is None:
        print(f"❌ Error: Could not load image A: {args.image_a}")
        return
    if image_b is None:
        print(f"❌ Error: Could not load image B: {args.image_b}")
        return
    
    # Convert BGR to RGB
    image_a_rgb = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB)
    image_b_rgb = cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB)
    
    print(f"✓ Image A loaded: {image_a.shape}")
    print(f"✓ Image B loaded: {image_b.shape}")
    
    # Run pipeline with reporting
    print(f"\n🔬 Running PatchCore + SAM pipeline...")
    print(f"   - Generating bidirectional heatmaps (A→B and B→A)")
    print(f"   - Computing comprehensive metrics")
    print(f"   - Creating basic reports (JSON, TXT, HTML)")
    if args.use_gemini:
        print(f"   - Sending to Gemini AI for structural analysis")
    
    result = run_patchcore_sam_pipeline(
        test_img=image_b_rgb,
        ref_img=image_a_rgb,
        generate_reports=True,
        output_dir=args.output_dir,
        use_gemini=args.use_gemini
    )
    
    if result is None:
        print("❌ Pipeline failed")
        return
    
    # Print summary
    print("\n" + "="*80)
    print("📊 DETECTION SUMMARY")
    print("="*80)
    print(f"Severity Score:        {result['severity']:.4f}")
    
    if 'metrics' in result:
        metrics = result['metrics']
        print(f"Anomaly Coverage:      {metrics['anomaly_percentage']:.2f}%")
        print(f"Total Anomaly Pixels:  {metrics['total_anomaly_pixels']:,}")
        print(f"High Severity Pixels:  {metrics['high_severity_pixels']:,}")
        print(f"Medium Severity:       {metrics['medium_severity_pixels']:,}")
        print(f"Low Severity:          {metrics['low_severity_pixels']:,}")
        print(f"Number of Regions:     {metrics['num_regions']}")
        print(f"Largest Region:        {metrics['largest_region_area']:,} pixels")
    
    if 'ai_analysis' in result:
        ai = result['ai_analysis']
        print("\n🤖 AI STRUCTURAL ANALYSIS")
        print("="*80)
        print(f"Summary: {ai.get('summary', 'N/A')}")
        
        issues = ai.get('structural_issues', [])
        if issues:
            print(f"\n⚠️  Detected {len(issues)} structural issue(s):")
            for i, issue in enumerate(issues, 1):
                print(f"\n  {i}. {issue.get('component', 'Unknown')}")
                print(f"     Issue: {issue.get('issue_type', 'Unknown')}")
                print(f"     Severity: {issue.get('severity', 'Unknown').upper()}")
                print(f"     Location: {issue.get('location', 'Not specified')}")
                print(f"     Recommendation: {issue.get('recommendation', 'No recommendation')}")
        
        actions = ai.get('critical_actions', [])
        if actions:
            print(f"\n🔧 Critical Actions Required:")
            for action in actions:
                print(f"   - {action}")
    
    # Print report paths
    if 'reports' in result:
        print("\n" + "="*80)
        print("📁 GENERATED REPORTS")
        print("="*80)
        for report_type, path in result['reports'].items():
            print(f"   {report_type:25s}: {path}")
    
    print("="*80)
    
    # Visualize if requested
    if args.visualize:
        print("\n👁️  Displaying results (press any key to close)...")
        
        # Create visualization
        overlay = result['overlay']
        heatmap = result['heatmap']
        
        # Convert RGB to BGR for display
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        
        # Resize for display
        display_height = 600
        h, w = overlay_bgr.shape[:2]
        scale = display_height / h
        display_width = int(w * scale)
        
        overlay_display = cv2.resize(overlay_bgr, (display_width, display_height))
        heatmap_display = cv2.resize(heatmap, (display_width, display_height))
        
        # Stack side by side
        combined = cv2.hconcat([overlay_display, heatmap_display])
        
        cv2.imshow('PatchCore + SAM Results | Overlay (left) | Heatmap (right)', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
