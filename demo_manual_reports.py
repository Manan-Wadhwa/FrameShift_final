"""
Demo script to test manual report generation
Shows the detailed reports that are generated when LLaVA is not available
"""
import sys
sys.path.append('.')

from llava.llava_report import generate_rule_based_report

print("="*80)
print("F1 VISUAL DIFFERENCE ENGINE - MANUAL REPORT DEMO")
print("="*80)
print("\nThis demonstrates the comprehensive rule-based reports generated")
print("when LLaVA vision-language model is not available.\n")

# Example 1: Semantic Change Report (Livery Change)
print("\n" + "="*80)
print("EXAMPLE 1: SEMANTIC CHANGE (Livery Modification)")
print("="*80)

semantic_features = {
    'texture_var': 245.3,
    'edge_density': 0.18,
    'color_shift': 0.42,
    'entropy': 4.8
}

semantic_report = generate_rule_based_report(
    pipeline_type="semantic",
    severity=None,
    features=semantic_features,
    mask_area=15234,  # pixels
    pipeline_name="DINOv2 Semantic Detection"
)

print(semantic_report)

# Example 2: High Severity Anomaly Report (Tire Damage)
print("\n\n" + "="*80)
print("EXAMPLE 2: ANOMALY DETECTION - HIGH SEVERITY (Tire Damage)")
print("="*80)

anomaly_features_high = {
    'texture_var': 892.5,
    'edge_density': 0.47,
    'color_shift': 0.08,
    'entropy': 6.2
}

anomaly_report_high = generate_rule_based_report(
    pipeline_type="anomaly",
    severity=0.78,
    features=anomaly_features_high,
    mask_area=8456,
    pipeline_name="PatchCore Anomaly Detection"
)

print(anomaly_report_high)

# Example 3: Low Severity Anomaly Report
print("\n\n" + "="*80)
print("EXAMPLE 3: ANOMALY DETECTION - LOW SEVERITY (Minor Surface Irregularity)")
print("="*80)

anomaly_features_low = {
    'texture_var': 325.1,
    'edge_density': 0.22,
    'color_shift': 0.05,
    'entropy': 4.1
}

anomaly_report_low = generate_rule_based_report(
    pipeline_type="anomaly",
    severity=0.28,
    features=anomaly_features_low,
    mask_area=2341,
    pipeline_name="PaDiM Anomaly Detection"
)

print(anomaly_report_low)

# Example 4: Subtle Semantic Change
print("\n\n" + "="*80)
print("EXAMPLE 4: SEMANTIC CHANGE - SUBTLE (Minor Design Update)")
print("="*80)

semantic_features_subtle = {
    'texture_var': 198.7,
    'edge_density': 0.12,
    'color_shift': 0.18,
    'entropy': 4.2
}

semantic_report_subtle = generate_rule_based_report(
    pipeline_type="semantic",
    severity=None,
    features=semantic_features_subtle,
    mask_area=4523,
    pipeline_name="CLIP Semantic Detection"
)

print(semantic_report_subtle)

print("\n\n" + "="*80)
print("DEMO COMPLETE")
print("="*80)
print("\nThese detailed reports are automatically generated when LLaVA is unavailable.")
print("They provide comprehensive analysis based on quantitative features.")
print("\nTo see these reports in action, run:")
print("  python main_pipeline.py samples/back1.jpeg samples/back2.jpeg")
print("\n")
