# 🎨 Streamlit App - Comprehensive Reporting Guide

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
pip install -r streamlit_requirements.txt
```

### 2. Set Up Gemini API (Optional)

For AI-powered structural analysis:

```powershell
# Get API key from: https://aistudio.google.com/app/apikey
$env:GEMINI_API_KEY="your-api-key-here"
```

### 3. Run Streamlit App

```powershell
cd demo
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 New Reporting Features

### In-App Comprehensive Reports

When you run **PatchCore + SAM** pipeline, you'll now see a button:

**"📊 Show Comprehensive Report"**

Click it to see:

### 1. **Metrics Dashboard**
- 🎯 Key metrics cards (Coverage, Pixels, Regions, Max Intensity)
- 📊 Interactive severity distribution chart
- 🗺️ Spatial distribution heatmap (3x3 grid)
- 📋 Top anomaly regions table

### 2. **Severity Distribution Chart**
- Color-coded bar chart (Green/Yellow/Red)
- Shows pixel counts for Low/Medium/High severity
- Interactive hover tooltips

### 3. **Spatial Heatmap**
- 3x3 grid showing anomaly density
- Percentage values per cell
- Red color scale (higher = more anomalies)

### 4. **Top Regions Table**
- Ranked by area
- Bounding box coordinates
- Max and mean intensity values

### 5. **AI Analysis** (if enabled)
- 🤖 Summary from Gemini AI
- ⚠️ Detected structural issues
  - Component (tire, wing, etc.)
  - Issue type (crack, deformation, etc.)
  - Severity level
  - Safety impact
  - Recommendations
- 🔧 Critical actions list

## 🎯 How to Use

### Step 1: Upload Images

**Option A: Upload Custom Images**
- Click "Reference Image (A)" → Upload
- Click "Test Image (B)" → Upload

**Option B: Use Sample Pairs**
- Select from dropdown:
  - Livery Change
  - Object Change
  - Tire Damage
  - Subtle Change

### Step 2: Select Pipelines

**Semantic Models:**
- ✓ DINO
- ✓ CLIP

**Anomaly Models:**
- ✓ PatchCore
- ✓ PaDiM

**Advanced:**
- ✓ PatchCore + SAM (← **Enable this for comprehensive reports!**)
- ✓ PatchCore KNN

### Step 3: Enable AI Analysis (Optional)

- ✓ Check "Enable Gemini AI Analysis"
- Make sure `GEMINI_API_KEY` is set

### Step 4: Run Analysis

- Click **"🚀 Run Analysis"**
- Wait for processing (10-30 seconds)

### Step 5: View Results

Scroll down to the **PatchCore + SAM** section and click:

**"📊 Show Comprehensive Report"**

## 📈 Report Components

### Key Metrics Cards

```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ Anomaly Coverage│ Affected Pixels │ Detected Regions│  Max Intensity  │
│     12.45%      │    258,048      │        8        │      0.89       │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

### Severity Distribution

Interactive Plotly bar chart:
- **Green bar**: Low severity (0.3-0.6)
- **Yellow bar**: Medium severity (0.6-0.8)
- **Red bar**: High severity (0.8-1.0)

Hover to see exact pixel counts!

### Spatial Heatmap

3x3 grid showing where anomalies are located:

```
Top-Left     Top-Center    Top-Right
  5.2%          8.1%          3.4%

Mid-Left     Mid-Center    Mid-Right
  12.3%        15.8%         9.7%

Bottom-Left  Bottom-Center Bottom-Right
  7.1%          11.2%         6.5%
```

**Darker red = More anomalies in that region**

### Top Regions Table

| Rank | Area (px) | Bounding Box      | Max Intensity | Mean Intensity |
|------|-----------|-------------------|---------------|----------------|
| 1    | 38,456    | (120, 340, 85, 92)| 0.923         | 0.678          |
| 2    | 24,132    | (450, 210, 65, 71)| 0.854         | 0.601          |
| ...  | ...       | ...               | ...           | ...            |

### AI Analysis Section (with Gemini)

**Summary Box:**
```
ℹ️ Summary: Front left tire shows significant surface degradation 
with multiple radial cracks. Front wing endplate has minor 
deformation. Recommend immediate tire replacement and wing inspection.
```

**Detected Issues:**

```
🔴 Issue 1: Front left tire - HIGH
   Issue Type: Surface cracking
   Location: Outer shoulder region
   Description: Multiple radial cracks visible in tire rubber
   Safety Impact: Critical - potential tire failure
   Recommendation: Immediate replacement required

🟡 Issue 2: Front wing endplate - MEDIUM
   Issue Type: Minor deformation
   Location: Left side, lower section
   Description: Slight bending detected in endplate structure
   Safety Impact: Moderate - may affect aerodynamics
   Recommendation: Inspect for structural integrity
```

**Critical Actions:**
```
⚠️ • Replace front left tire before next session
⚠️ • Inspect front wing endplate for cracks
⚠️ • Monitor tire wear on remaining tires
```

## 🎨 Visual Features

### Interactive Charts

All charts are powered by **Plotly**:
- 🖱️ Hover for details
- 🔍 Zoom and pan
- 💾 Download as PNG
- 📊 Responsive to window size

### Expandable Sections

AI issues are in **expandable cards**:
- Click to expand/collapse
- Color-coded by severity
- Easy to scan multiple issues

### Clean Layout

- Metrics in organized columns
- Clear section headers
- Color-coded severity (🔴 🟡 🟢)
- Professional styling

## 🔧 Customization

### Change Severity Thresholds

The app uses default thresholds:
- Low: 0.3 - 0.6
- Medium: 0.6 - 0.8
- High: 0.8 - 1.0

To customize, edit `utils/heatmap_metrics.py`

### Adjust Spatial Grid

Default: 3x3 grid

To change to 4x4 or 5x5, edit the `compute_spatial_distribution` call

### Modify Gemini Prompt

To focus on specific components, edit:
`utils/gemini_analyzer.py` → `create_structural_analysis_prompt()`

## 🐛 Troubleshooting

### "plotly not found"
```powershell
pip install plotly pandas
```

### "GEMINI_API_KEY not set" warning
AI analysis is **optional**. The app works fine without it. To enable:
```powershell
$env:GEMINI_API_KEY="your-key-here"
```

### "Report button does nothing"
Make sure you've:
1. Selected **PatchCore + SAM** pipeline
2. Clicked **Run Analysis**
3. Waited for processing to complete

### Charts not showing
Check browser console for JavaScript errors. Try refreshing the page.

### Out of memory
- Use smaller images
- Run fewer pipelines at once
- Restart the Streamlit app

## 💡 Tips

1. **Always enable PatchCore + SAM** for comprehensive reports
2. **Enable Gemini AI** for structural insights (needs API key)
3. **Use interactive charts** - hover, zoom, download
4. **Check spatial heatmap** to see where anomalies concentrate
5. **Review AI recommendations** for actionable next steps

## 📊 Metrics Explained

### Anomaly Coverage
Percentage of image pixels with anomalies above threshold (0.3)

### Severity Levels
- **Low (0.3-0.6)**: Minor wear, monitoring needed
- **Medium (0.6-0.8)**: Significant wear, plan action
- **High (0.8-1.0)**: Critical damage, immediate attention

### Spatial Distribution
Shows which parts of the image have most anomalies:
- Helps identify localized damage vs. widespread issues
- Useful for component-specific analysis

### Coverage Comparison
- **A→B**: Changes from reference perspective
- **B→A**: Changes from test perspective
- **Difference**: Asymmetric changes (additions/removals)

## 🎯 Best Practices

1. **Start with sample pairs** to understand the system
2. **Enable all pipelines** for first run, then select best 2
3. **Use Gemini AI** for structural analysis
4. **Review spatial heatmap** to identify problem areas
5. **Check critical actions** for immediate next steps
6. **Download charts** for reports/presentations

## 🔐 Security

- API keys stored in environment variables only
- No data sent to external services (except Gemini if enabled)
- Images processed locally
- Reports generated in-browser

## 📚 Additional Resources

- **Setup Guide**: `GEMINI_SETUP.md`
- **Full Documentation**: `REPORTS_README.md`
- **API Reference**: `IMPLEMENTATION_SUMMARY.md`
- **Quick Start**: `QUICK_REFERENCE.md`

---

**Enjoy the comprehensive reporting system! 🎉**

For questions or issues, check the documentation files or review the example outputs.
