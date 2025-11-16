# Gemini AI Integration for Structural Analysis

This guide explains how to set up and use Google Gemini AI for analyzing F1 anomaly detection heatmaps.

## What Gemini Does

Gemini AI analyzes the anomaly heatmaps and provides:
- **Structural damage detection** (tire cracks, wing damage, carbon fiber cracks)
- **Component-specific analysis** (front/rear wings, tires, suspension, bodywork)
- **Severity assessment** (critical/high/medium/low based on heatmap colors)
- **Safety impact evaluation** (potential failures, performance degradation)
- **Actionable recommendations** (immediate repairs, replacements, monitoring)

**What Gemini IGNORES:**
- Cosmetic changes (logos, stickers, paint, sponsor decals)
- Non-structural differences

## Getting Your Gemini API Key

1. **Go to Google AI Studio:**
   - Visit: https://aistudio.google.com/app/apikey

2. **Sign in with Google account**

3. **Create API Key:**
   - Click "Get API Key" or "Create API Key"
   - Select a Google Cloud project (or create new one)
   - Copy the generated API key

4. **Set Environment Variable:**

   **Windows (PowerShell):**
   ```powershell
   $env:GEMINI_API_KEY="your-api-key-here"
   ```

   **Windows (Command Prompt):**
   ```cmd
   set GEMINI_API_KEY=your-api-key-here
   ```

   **Linux/Mac (Bash):**
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

   **Permanent Setup (Windows):**
   ```powershell
   [System.Environment]::SetEnvironmentVariable('GEMINI_API_KEY', 'your-api-key-here', 'User')
   ```

   **Permanent Setup (Linux/Mac):**
   Add to `~/.bashrc` or `~/.zshrc`:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

## Installing Required Packages

```bash
pip install google-generativeai
```

## Usage Examples

### 1. Basic Usage (No AI Analysis)

Generates metrics only:

```python
from pipelines.anomaly_patchcore_sam import run_patchcore_sam_pipeline
import cv2

image_a = cv2.cvtColor(cv2.imread('ref.jpg'), cv2.COLOR_BGR2RGB)
image_b = cv2.cvtColor(cv2.imread('test.jpg'), cv2.COLOR_BGR2RGB)

result = run_patchcore_sam_pipeline(
    test_img=image_b,
    ref_img=image_a,
    generate_reports=True,
    output_dir='reports/',
    use_gemini=False  # No AI analysis
)

# Access basic metrics
print(f"Anomaly coverage: {result['metrics']['anomaly_percentage']:.2f}%")
```

**Generates:**
- `basic_metrics.json` - Machine-readable metrics
- `basic_metrics.txt` - Human-readable text report
- `basic_metrics.html` - Visual HTML report
- `metrics_visualization.jpg` - Visualization of heatmaps and regions

### 2. With Gemini AI Analysis

Adds AI-powered structural analysis:

```python
result = run_patchcore_sam_pipeline(
    test_img=image_b,
    ref_img=image_a,
    generate_reports=True,
    output_dir='reports/',
    use_gemini=True  # Enable AI analysis
)

# Access AI analysis
ai = result['ai_analysis']
print(f"Summary: {ai['summary']}")

for issue in ai['structural_issues']:
    print(f"Component: {issue['component']}")
    print(f"Issue: {issue['issue_type']}")
    print(f"Severity: {issue['severity']}")
    print(f"Recommendation: {issue['recommendation']}")
```

**Generates everything from Basic + AI report:**
- `ai_analysis_report.html` - Comprehensive report with AI insights

### 3. Command Line Usage

```bash
# Basic metrics only
python example_patchcore_sam_reports.py --image-a ref.jpg --image-b test.jpg --output-dir reports/

# With Gemini AI analysis
python example_patchcore_sam_reports.py --image-a ref.jpg --image-b test.jpg --output-dir reports/ --use-gemini

# With visualization
python example_patchcore_sam_reports.py --image-a ref.jpg --image-b test.jpg --output-dir reports/ --use-gemini --visualize
```

## Understanding the Reports

### Basic Metrics Report (`basic_metrics.html`)

Shows:
- **Key Metrics**: Total anomaly coverage, affected pixels, number of regions
- **Severity Distribution**: Bar chart showing low/medium/high severity pixels
- **Top Regions**: Table of largest anomaly regions with bounding boxes
- **Spatial Distribution**: 3x3 grid showing anomaly density across image

### AI Analysis Report (`ai_analysis_report.html`)

Adds:
- **Structural Issues List**: Detailed analysis of each structural problem
  - Component affected (tire, wing, etc.)
  - Issue type (crack, deformation, wear)
  - Severity level (critical/high/medium/low)
  - Safety impact assessment
  - Specific recommendations
- **Critical Actions**: Prioritized list of required actions
- **AI Summary**: Overall assessment from Gemini

### Metrics Visualization (`metrics_visualization.jpg`)

6-panel visualization:
- **Row 1**: Image A (reference) | Image B (current)
- **Row 2**: Union Heatmap | Union Overlay with Region Boxes
- **Row 3**: Intersection Heatmap | Difference Heatmap

## Gemini Prompt Customization

The Gemini prompt is optimized for F1 structural analysis. To customize it, edit `utils/gemini_analyzer.py`:

```python
def create_structural_analysis_prompt(self, metrics: Dict) -> str:
    prompt = f"""You are analyzing anomaly detection heatmaps from an F1 race car inspection system.

**CRITICAL INSTRUCTIONS:**
- Focus ONLY on structural and mechanical changes that affect safety/performance
- IGNORE cosmetic changes like logos, stickers, paint, sponsor decals
- Report tire cracks, rubber degradation, surface damage
...
```

## API Costs and Limits

**Gemini 1.5 Flash** (used by default):
- **Free tier**: 15 requests per minute, 1 million tokens per minute
- **Pricing**: Free for moderate usage
- **Image limit**: Up to 3000 images per request

For high-volume usage, monitor your usage at:
https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas

## Troubleshooting

### "Import google.generativeai could not be resolved"
```bash
pip install google-generativeai
```

### "Gemini API key not found"
Make sure you've set the environment variable:
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```

### "API quota exceeded"
You've hit the free tier limit. Wait a minute or upgrade to paid tier.

### "Model not found"
Update the package:
```bash
pip install --upgrade google-generativeai
```

## Security Best Practices

1. **Never commit API keys to Git:**
   ```bash
   echo "GEMINI_API_KEY=*" >> .gitignore
   ```

2. **Use environment variables** (not hardcoded keys)

3. **Rotate keys regularly** (create new key, delete old one)

4. **Restrict API key** in Google Cloud Console:
   - Limit to specific APIs (Generative Language API only)
   - Set application restrictions
   - Add IP restrictions if needed

## Example Output

When you run with `--use-gemini`, you'll see:

```
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
```

## Advanced Features

### Bidirectional Heatmap Analysis

The system automatically computes:
- **A→B Heatmap**: Anomalies when comparing A to B (reference to test)
- **B→A Heatmap**: Anomalies when comparing B to A (test to reference)
- **Union Heatmap**: Combined anomalies (max of A→B and B→A)
- **Intersection Heatmap**: Common anomalies in both directions
- **Difference Heatmap**: Asymmetric anomalies

### Severity Levels

Based on heatmap intensity:
- **Low** (0.3-0.6): Minor wear, slight changes
- **Medium** (0.6-0.8): Moderate damage, significant wear
- **High** (0.8-1.0): Critical damage, immediate attention required

### Region Analysis

Automatically detects and analyzes:
- Connected anomaly regions
- Bounding boxes
- Centroid positions
- Max/mean intensity per region
- Region areas

### Spatial Distribution

3x3 grid analysis showing:
- Anomaly concentration in different image areas
- Helps identify localized vs distributed damage
- Useful for component-specific analysis (e.g., front wing damage)
