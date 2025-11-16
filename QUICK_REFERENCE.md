# 🚀 Quick Reference - PatchCore + SAM Reporting

## 30-Second Setup

```powershell
# 1. Install packages
pip install google-generativeai opencv-python numpy torch torchvision

# 2. Get Gemini API key from: https://aistudio.google.com/app/apikey

# 3. Set environment variable
$env:GEMINI_API_KEY="your-api-key-here"

# 4. Test setup
python setup_gemini.py

# 5. Run demo
python demo_reports.py
```

## One-Line Usage

```python
# Basic (no AI)
result = run_patchcore_sam_pipeline(test_img, ref_img=ref_img, generate_reports=True, output_dir='reports/')

# With AI
result = run_patchcore_sam_pipeline(test_img, ref_img=ref_img, generate_reports=True, output_dir='reports/', use_gemini=True)
```

## Command Line

```bash
# Without AI
python example_patchcore_sam_reports.py --image-a ref.jpg --image-b test.jpg --output-dir reports/

# With AI
python example_patchcore_sam_reports.py --image-a ref.jpg --image-b test.jpg --output-dir reports/ --use-gemini --visualize
```

## What You Get

```
reports/
├── basic_metrics.json          # Data for programs
├── basic_metrics.txt           # Human-readable summary
├── basic_metrics.html          # Interactive visual report ⭐
├── metrics_visualization.jpg   # 6-panel heatmap overview ⭐
└── ai_analysis_report.html     # AI structural analysis ⭐⭐⭐
```

## Key Metrics

- **Anomaly Coverage**: % of image with anomalies
- **Severity Levels**: Low (0.3-0.6), Medium (0.6-0.8), High (0.8-1.0)
- **Regions**: Detected anomaly regions with bounding boxes
- **Spatial Distribution**: 3x3 grid showing anomaly density

## AI Analysis Focus

✅ **Analyzes:**
- Tire cracks and degradation
- Wing damage and deformation
- Suspension damage
- Brake disc wear
- Bodywork cracks
- Structural integrity

❌ **Ignores:**
- Logos and stickers
- Paint changes
- Sponsor decals
- Cosmetic differences

## Common Commands

```python
# Access metrics
result['metrics']['anomaly_percentage']     # 12.45
result['metrics']['num_regions']            # 8
result['metrics']['high_severity_pixels']   # 45120

# Access AI analysis
result['ai_analysis']['summary']
result['ai_analysis']['structural_issues']
result['ai_analysis']['critical_actions']

# Access reports
result['reports']['basic_html']
result['reports']['ai_analysis']
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No AI analysis | Set `GEMINI_API_KEY` environment variable |
| Import error | `pip install google-generativeai` |
| Out of memory | Images auto-scaled, reduce size if needed |
| Slow processing | Normal - PatchCore uses deep features |

## Files to Open

1. **`reports/basic_metrics.html`** - Start here! Visual overview
2. **`reports/ai_analysis_report.html`** - AI insights (if enabled)
3. **`reports/metrics_visualization.jpg`** - Quick visual check

## Customization Hotspots

```python
# Change severity thresholds
HeatmapMetrics.compute_severity_levels(
    heatmap, low_threshold=0.3, medium_threshold=0.6, high_threshold=0.8
)

# Change union method
HeatmapMetrics.compute_union_heatmap(a2b, b2a, method='max')  # or 'mean', 'weighted'

# Adjust Gemini prompt
# Edit: utils/gemini_analyzer.py -> create_structural_analysis_prompt()
```

## API Limits (Free Tier)

- **Requests**: 15/minute
- **Tokens**: 1M/minute
- **Images**: 3000/request
- **Cost**: FREE ✨

## Next Steps

1. **Read**: `REPORTS_README.md` for full docs
2. **Setup**: `GEMINI_SETUP.md` for AI config
3. **Examples**: Run `demo_reports.py`
4. **Integrate**: Add to your pipeline

## Code Snippet Library

### Basic Integration
```python
from pipelines.anomaly_patchcore_sam import run_patchcore_sam_pipeline
import cv2

img_a = cv2.cvtColor(cv2.imread('ref.jpg'), cv2.COLOR_BGR2RGB)
img_b = cv2.cvtColor(cv2.imread('test.jpg'), cv2.COLOR_BGR2RGB)

result = run_patchcore_sam_pipeline(
    test_img=img_b,
    ref_img=img_a,
    generate_reports=True,
    output_dir='reports/',
    use_gemini=True
)

print(f"Coverage: {result['metrics']['anomaly_percentage']:.2f}%")
```

### Metrics Only
```python
from utils.heatmap_metrics import HeatmapMetrics

metrics = HeatmapMetrics.compute_comprehensive_metrics(
    heatmap_a2b, heatmap_b2a, img_a, img_b
)
```

### Custom Report
```python
from utils.basic_report import BasicReportGenerator

BasicReportGenerator.generate_html_report(metrics, 'custom.html')
```

### Standalone AI
```python
from utils.gemini_analyzer import GeminiAnalyzer

gemini = GeminiAnalyzer()
analysis = gemini.analyze_heatmaps(img_a, img_b, heatmap, metrics)
gemini.generate_report(metrics, analysis, 'ai_report.html')
```

## Quick Tests

```bash
# Test Gemini setup
python setup_gemini.py

# Run full demo
python demo_reports.py

# Test with real images
python example_patchcore_sam_reports.py --image-a ref.jpg --image-b test.jpg --output-dir test_reports/ --use-gemini
```

## Performance Tips

- Images auto-scale for efficiency
- GPU used if available (CUDA)
- Reports generate in < 1 second
- Gemini analysis: 2-5 seconds
- Total pipeline: ~10-30 seconds

## Support

- **Documentation**: `REPORTS_README.md`, `GEMINI_SETUP.md`
- **Implementation**: `IMPLEMENTATION_SUMMARY.md`
- **Examples**: `example_patchcore_sam_reports.py`, `demo_reports.py`
- **Setup**: `setup_gemini.py`

---

**Remember**: Open `reports/basic_metrics.html` in your browser for the best experience! 🌟
