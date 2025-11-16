# 📊 Comprehensive Reporting System for PatchCore + SAM

This system generates detailed anomaly detection reports with AI-powered structural analysis using Google Gemini.

## 🎯 Features

### 1. **Bidirectional Heatmap Analysis**
- **A→B Heatmap**: Changes from reference to test image
- **B→A Heatmap**: Changes from test to reference image
- **Union Heatmap**: Combined anomalies (maximum from both directions)
- **Intersection Heatmap**: Common anomalies in both directions
- **Difference Heatmap**: Asymmetric anomalies

### 2. **Comprehensive Metrics**
- Total anomaly coverage (percentage)
- Severity breakdown (low/medium/high)
- Region analysis (count, areas, bounding boxes)
- Spatial distribution (3x3 grid)
- Coverage comparison (A→B vs B→A)

### 3. **Multi-Format Reports**
- **JSON** (`basic_metrics.json`): Machine-readable data
- **TXT** (`basic_metrics.txt`): Human-readable text
- **HTML** (`basic_metrics.html`): Interactive visual report
- **Visualization** (`metrics_visualization.jpg`): 6-panel heatmap overview

### 4. **AI-Powered Structural Analysis** (Optional)
- Uses Google Gemini to analyze heatmaps
- Focuses on **structural issues only** (not cosmetic):
  - Tire cracks and rubber degradation
  - Wing damage and deformation
  - Suspension damage
  - Brake disc wear
  - Bodywork cracks and panel damage
- Provides severity assessment and recommendations
- Ignores logos, stickers, paint, sponsor decals

## 🚀 Quick Start

### Setup

1. **Install dependencies:**
   ```bash
   pip install google-generativeai opencv-python numpy torch torchvision
   ```

2. **Set up Gemini API** (optional, for AI analysis):
   ```powershell
   # Get API key from: https://aistudio.google.com/app/apikey
   $env:GEMINI_API_KEY="your-api-key-here"
   
   # Verify setup
   python setup_gemini.py
   ```

### Basic Usage

```python
import cv2
from pipelines.anomaly_patchcore_sam import run_patchcore_sam_pipeline

# Load images
image_a = cv2.cvtColor(cv2.imread('ref.jpg'), cv2.COLOR_BGR2RGB)
image_b = cv2.cvtColor(cv2.imread('test.jpg'), cv2.COLOR_BGR2RGB)

# Run with basic reports (no AI)
result = run_patchcore_sam_pipeline(
    test_img=image_b,
    ref_img=image_a,
    generate_reports=True,
    output_dir='reports/',
    use_gemini=False
)

# Access metrics
print(f"Anomaly coverage: {result['metrics']['anomaly_percentage']:.2f}%")
print(f"Number of regions: {result['metrics']['num_regions']}")
```

### With AI Analysis

```python
# Run with Gemini AI analysis
result = run_patchcore_sam_pipeline(
    test_img=image_b,
    ref_img=image_a,
    generate_reports=True,
    output_dir='reports/',
    use_gemini=True  # Requires GEMINI_API_KEY
)

# Access AI insights
for issue in result['ai_analysis']['structural_issues']:
    print(f"{issue['component']}: {issue['issue_type']} - {issue['severity']}")
```

### Command Line

```bash
# Basic reports only
python example_patchcore_sam_reports.py \
    --image-a samples/ref.jpg \
    --image-b samples/test.jpg \
    --output-dir reports/

# With AI analysis
python example_patchcore_sam_reports.py \
    --image-a samples/ref.jpg \
    --image-b samples/test.jpg \
    --output-dir reports/ \
    --use-gemini

# With visualization
python example_patchcore_sam_reports.py \
    --image-a samples/ref.jpg \
    --image-b samples/test.jpg \
    --output-dir reports/ \
    --use-gemini \
    --visualize
```

## 📁 Generated Files

After running with `generate_reports=True`, you'll get:

```
reports/
├── basic_metrics.json          # Machine-readable metrics
├── basic_metrics.txt           # Human-readable text report
├── basic_metrics.html          # Interactive HTML report
├── metrics_visualization.jpg   # 6-panel heatmap visualization
└── ai_analysis_report.html     # AI-powered structural analysis (if use_gemini=True)
```

## 📊 Report Contents

### Basic Metrics Report

**Key Metrics:**
- Anomaly coverage percentage
- Total affected pixels
- Number of anomaly regions
- Maximum anomaly intensity

**Severity Distribution:**
- Low severity pixels (0.3-0.6)
- Medium severity pixels (0.6-0.8)
- High severity pixels (0.8-1.0)

**Region Analysis:**
- Top 10 largest regions
- Bounding boxes
- Max/mean intensity per region

**Spatial Distribution:**
- 3x3 grid showing anomaly density
- Color-coded by intensity

### AI Analysis Report (with Gemini)

**Structural Issues:**
Each issue includes:
- Component (e.g., "Front left tire")
- Issue type (e.g., "Surface cracking")
- Severity (critical/high/medium/low)
- Location (specific area)
- Description
- Safety impact
- Recommendation

**Critical Actions:**
- Prioritized list of required actions
- Immediate vs. monitoring recommendations

**AI Summary:**
- Overall assessment
- Key findings

## 🎨 Metrics Visualization

The `metrics_visualization.jpg` contains 6 panels:

```
+-------------------+-------------------+
|    Image A        |    Image B        |
|  (Reference)      |   (Current)       |
+-------------------+-------------------+
|  Union Heatmap    | Union Overlay     |
|   (Combined)      | + Region Boxes    |
+-------------------+-------------------+
| Intersection      |   Difference      |
|   Heatmap         |    Heatmap        |
+-------------------+-------------------+
```

## 🔧 Customization

### Adjust Severity Thresholds

Edit `utils/heatmap_metrics.py`:

```python
severity = HeatmapMetrics.compute_severity_levels(
    union_norm,
    low_threshold=0.3,      # Change this
    medium_threshold=0.6,   # Change this
    high_threshold=0.8      # Change this
)
```

### Customize Gemini Prompt

Edit `utils/gemini_analyzer.py` in `create_structural_analysis_prompt()`:

```python
prompt = f"""You are analyzing anomaly detection heatmaps...

**YOUR CUSTOM INSTRUCTIONS HERE**
- Focus on specific components
- Add domain knowledge
- Adjust severity criteria
...
```

### Change Union Method

Edit pipeline call:

```python
# In utils/heatmap_metrics.py
union_heatmap = HeatmapMetrics.compute_union_heatmap(
    heatmap_a2b, 
    heatmap_b2a, 
    method='max'  # Options: 'max', 'mean', 'weighted'
)
```

## 🧪 Example Output

```
================================================================================
🏎️  F1 PATCHCORE + SAM ANOMALY DETECTION WITH COMPREHENSIVE REPORTING
================================================================================

📂 Loading images...
✓ Image A loaded: (1080, 1920, 3)
✓ Image B loaded: (1080, 1920, 3)

🔬 Running PatchCore + SAM pipeline...
   - Generating bidirectional heatmaps (A→B and B→A)
   - Computing comprehensive metrics
   - Creating basic reports (JSON, TXT, HTML)
   - Sending to Gemini AI for structural analysis

================================================================================
📋 GENERATING COMPREHENSIVE REPORTS
================================================================================
📊 Computing comprehensive heatmap metrics...
✓ Metrics computed: 12.45% anomaly coverage, 8 regions

📊 Generating basic metrics reports...
✓ JSON report saved to: reports/basic_metrics.json
✓ Text report saved to: reports/basic_metrics.txt
✓ HTML report saved to: reports/basic_metrics.html

🤖 Generating AI-powered structural analysis with Gemini...
📤 Sending heatmaps to Gemini for analysis...
✓ Received response from Gemini
✓ Report saved to: reports/ai_analysis_report.html
✓ AI analysis report generated

================================================================================
✅ REPORTS GENERATED SUCCESSFULLY
================================================================================
📁 Output directory: reports
📊 Basic reports: JSON, TXT, HTML
🤖 AI analysis: reports/ai_analysis_report.html
================================================================================

================================================================================
📊 DETECTION SUMMARY
================================================================================
Severity Score:        0.3421
Anomaly Coverage:      12.45%
Total Anomaly Pixels:  258,048
High Severity Pixels:  45,120
Medium Severity:       89,344
Low Severity:          123,584
Number of Regions:     8
Largest Region:        38,456 pixels

🤖 AI STRUCTURAL ANALYSIS
================================================================================
Summary: Front left tire shows significant surface degradation with multiple 
radial cracks. Front wing endplate has minor deformation. Recommend immediate 
tire replacement and wing inspection.

⚠️  Detected 2 structural issue(s):

  1. Front left tire
     Issue: Surface cracking
     Severity: HIGH
     Location: Outer shoulder region
     Recommendation: Immediate replacement required

  2. Front wing endplate
     Issue: Minor deformation
     Severity: MEDIUM
     Location: Left side, lower section
     Recommendation: Inspect for structural integrity, monitor for progression

🔧 Critical Actions Required:
   - Replace front left tire before next session
   - Inspect front wing endplate for cracks
   - Monitor tire wear on remaining tires

================================================================================
```

## 🔍 Interpreting Results

### Heatmap Colors

- **Red/Hot**: High anomaly (critical issues)
- **Yellow/Orange**: Medium anomaly (significant wear)
- **Blue/Green**: Low anomaly (minor changes)
- **Dark Blue**: No anomaly

### Severity Levels

- **Low (0.3-0.6)**: Normal wear, minor changes, monitoring recommended
- **Medium (0.6-0.8)**: Significant wear, plan replacement/repair
- **High (0.8-1.0)**: Critical damage, immediate attention required

### Coverage Metrics

- **A→B Coverage**: Anomalies when using A as baseline (what changed from reference)
- **B→A Coverage**: Anomalies when using B as baseline (new features in test)
- **Difference**: Large difference suggests asymmetric changes (additions or removals)

## 🐛 Troubleshooting

### "google.generativeai not found"
```bash
pip install google-generativeai
```

### "GEMINI_API_KEY not set"
```powershell
$env:GEMINI_API_KEY="your-key-here"
python setup_gemini.py  # Verify setup
```

### "Out of memory" during processing
- Reduce image size before processing
- Use `reduce_size=True` in feature extraction (default)
- Process images in smaller batches

### AI analysis returns empty results
- Check API key is valid
- Verify internet connection
- Check Gemini API quota: https://console.cloud.google.com/

## 📚 Additional Resources

- **Gemini API Setup**: See `GEMINI_SETUP.md`
- **API Documentation**: https://ai.google.dev/docs
- **Example Script**: `example_patchcore_sam_reports.py`
- **Setup Checker**: `setup_gemini.py`

## 💡 Tips

1. **Always use bidirectional analysis** (`ref_img` parameter) for best results
2. **Enable AI analysis** for structural insights (requires API key)
3. **Review HTML reports** for easiest interpretation
4. **Check spatial distribution** to identify localized vs. distributed damage
5. **Compare severity levels** across multiple test runs to track degradation

## 🔐 Security

- Never commit API keys to version control
- Use environment variables for API keys
- Rotate API keys regularly
- Restrict API key usage in Google Cloud Console
