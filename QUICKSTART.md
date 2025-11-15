# 🚀 Quick Start Guide - F1 Visual Difference Engine

## Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Optional: Install Advanced Models**
   ```bash
   # SAM (Segment Anything)
   pip install git+https://github.com/facebookresearch/segment-anything.git
   
   # CLIP
   pip install git+https://github.com/openai/CLIP.git
   ```

3. **Download SAM Checkpoint** (Optional but recommended)
   - Download: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   - Place in project root directory

## Usage

### Option 1: Streamlit Dashboard (Recommended)
```bash
streamlit run demo/streamlit_app.py
```
Then:
- Select a sample pair or upload your own images
- Click "Run Analysis"
- Compare all 4 pipelines side-by-side

### Option 2: Jupyter Notebook
```bash
jupyter notebook demo/app.ipynb
```
Run all cells to see comprehensive analysis of all 4 sample pairs.

### Option 3: Command Line
```bash
python main_pipeline.py samples/back1.jpeg samples/back2.jpeg
```

## Sample Image Pairs

The `samples/` folder contains 4 test cases:

1. **Livery Change** - `back1.jpeg` ↔ `back2.jpeg`
   - Type: Semantic
   - Best Pipelines: DINO, CLIP

2. **Object Change** - `copy1.jpeg` ↔ `copy2.jpeg`
   - Type: Mixed
   - Best Pipelines: CLIP, PatchCore

3. **Tire Damage** - `crack1.jpg` ↔ `crack2.png`
   - Type: Anomaly
   - Best Pipelines: PatchCore, PaDiM

4. **Subtle Change** - `side1.jpeg` ↔ `side2.jpeg`
   - Type: Semantic
   - Best Pipelines: DINO

## Understanding the Output

Each pipeline returns:
- **Heatmap**: Colored visualization of differences
- **Mask**: Binary segmentation of changed region
- **Overlay**: Heatmap + mask on test image
- **Report**: LLaVA-generated natural language explanation
- **Severity** (anomaly pipelines): Numerical score 0-1

## Routing Features

The system automatically predicts whether the change is **semantic** or **anomaly** based on:

| Feature | Semantic Indicator | Anomaly Indicator |
|---------|-------------------|-------------------|
| Texture Variance | Low (<500) | High (>500) |
| Edge Density | Low (<0.3) | High (>0.3) |
| Color Shift | High (>0.15) | Low |

## Troubleshooting

**Models not loading?**
- System will fall back to classical CV methods
- Results will still work but with lower accuracy

**Out of memory?**
- Use smaller SAM model (`sam_vit_b_01ec64.pth`)
- Run on CPU instead of GPU

**Streamlit not opening?**
- Check that port 8501 is available
- Use: `streamlit run demo/streamlit_app.py --server.port 8502`

## Next Steps

1. Test all 4 sample pairs
2. Upload your own F1 car images
3. Compare pipeline performances
4. Adjust routing thresholds in `utils/routing_features.py`
5. Fine-tune preprocessing in `utils/preprocess.py`

---

**Happy Racing! 🏎️💨**
