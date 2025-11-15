# Image Alignment Methods - Quick Reference

## Overview
Instead of expanding features or resizing blindly, these methods **align the object/frame** based on **visual similarity** before comparison.

---

## Method Comparison

| Method | Speed | Handles | Best For | Limitations |
|--------|-------|---------|----------|-------------|
| **ECC** | ⚡⚡⚡ Fast | Translation, Rotation, Scale | Video frames, camera shake | Requires similar appearance |
| **Optical Flow** | ⚡⚡ Moderate | Non-rigid deformations | Motion tracking, complex movements | Computationally intensive |
| **Phase Correlation** | ⚡⚡⚡⚡ Very Fast | Translation only | Camera shake, pure shifts | Only handles translation |
| **ORB** | ⚡⚡⚡ Fast | Rotation, Scale, Perspective | Real-time apps, mobile | Less accurate than SIFT |
| **SIFT** (existing) | ⚡⚡ Moderate | Rotation, Scale, Perspective | High accuracy needs | Slower, patented |
| **Similarity Transform** | ⚡⚡ Moderate | Scale, Rotation, Translation | Object tracking | No perspective |
| **Template Matching** | ⚡ Slow | Scale, Translation | Known objects | Exhaustive search |

---

## 1. ECC (Enhanced Correlation Coefficient) ✅ RECOMMENDED FOR VIDEO

**Best for:** Sequential video frames, camera stabilization, F1 onboard footage

```python
from utils.alignment import ecc_alignment

# Align test frame to reference
aligned, transform, score = ecc_alignment(ref_frame, test_frame, 
                                          warp_mode=cv2.MOTION_EUCLIDEAN)

# Warp modes:
# - MOTION_TRANSLATION: Only x,y shift
# - MOTION_EUCLIDEAN: Translation + rotation
# - MOTION_AFFINE: Translation + rotation + scale + shear
# - MOTION_HOMOGRAPHY: Full perspective transform
```

**Why ECC?**
- Fast convergence for similar frames
- Handles camera motion naturally
- Works well with grayscale
- Iterative refinement (5000 iterations)

---

## 2. Optical Flow - Motion Understanding

**Best for:** Understanding pixel-level motion, tracking driver movements

```python
from utils.alignment import optical_flow_alignment

# Dense optical flow
aligned, flow = optical_flow_alignment(ref_frame, test_frame)

# Flow is 2-channel (dx, dy) showing pixel movements
# Can visualize motion vectors
```

**Why Optical Flow?**
- Shows WHERE pixels moved
- Handles non-rigid transformations
- Can track deformations (driver's body)
- Provides motion magnitude map

---

## 3. Phase Correlation - Ultra Fast Translation

**Best for:** Pure camera shake, vibration compensation

```python
from utils.alignment import phase_correlation_alignment

# Fast translation estimation
aligned, shift, confidence = phase_correlation_alignment(ref_frame, test_frame)

# shift = (dx, dy) in pixels
# confidence = correlation peak strength
```

**Why Phase Correlation?**
- Fastest method (FFT-based)
- Sub-pixel accuracy
- Robust to noise
- Perfect for translation-only cases

---

## 4. ORB Features - Real-time Alternative

**Best for:** Real-time processing, embedded systems, mobile apps

```python
from utils.alignment import orb_alignment

# Fast feature-based alignment
aligned, transform = orb_alignment(ref_frame, test_frame, 
                                  nfeatures=5000,
                                  transform_type='similarity')

# transform_type options:
# - 'homography': Full perspective (8 DOF)
# - 'affine': Affine transform (6 DOF)
# - 'similarity': Scale + rotation + translation (4 DOF)
```

**Why ORB?**
- 100x faster than SIFT
- Binary descriptors (efficient)
- Rotation invariant
- Free (no patent issues)

---

## 5. Similarity Transform - Constrained Alignment

**Best for:** Object tracking, known transformations, preventing overfitting

```python
from utils.alignment import similarity_transform_alignment

# Constrained to scale + rotation + translation
aligned, transform = similarity_transform_alignment(ref_frame, test_frame, 
                                                    feature_detector='sift')
```

**Why Similarity?**
- Only 4 parameters (s, θ, tx, ty)
- Prevents overfitting to noise
- Preserves shape better
- Good for objects that don't distort

---

## 6. Template Matching - Multi-Scale Search

**Best for:** Finding scaled objects, exhaustive search scenarios

```python
from utils.alignment import template_matching_alignment

# Search across scales
aligned, (scale, location, score) = template_matching_alignment(
    ref_frame, test_frame,
    scale_range=(0.8, 1.2),
    num_scales=20
)
```

**Why Template Matching?**
- No feature detection needed
- Finds best scale automatically
- Returns confidence score
- Works with texture-less regions

---

## 7. Adaptive Alignment - Auto-Select Best Method

**Best for:** Unknown transformation types, robustness

```python
from utils.alignment import adaptive_alignment

# Try multiple methods, return best
aligned, transform, method_used = adaptive_alignment(
    ref_frame, test_frame,
    methods=['ecc', 'orb', 'phase']
)

print(f"Best method: {method_used}")
```

**Why Adaptive?**
- Automatic method selection
- Fallback if one fails
- Quality-based ranking
- Handles diverse cases

---

## Integration with PatchCore

**Updated F1 Motion Tracker:**

```python
# Create detector with alignment
detector = PatchCoreDetector(
    frame_height=1080,
    frame_width=1920,
    alignment_method='ecc'  # ← NEW PARAMETER
)

# Options:
# - 'ecc': Enhanced Correlation Coefficient (RECOMMENDED)
# - 'orb': ORB features (fast)
# - 'phase': Phase correlation (translation only)
# - 'optical_flow': Dense optical flow
# - 'none': No alignment (original behavior)
```

**What happens:**
1. First frame → Build memory bank (reference)
2. Subsequent frames → Align to reference using ECC
3. Extract patches from **aligned** frame
4. Compare aligned patches to reference patches
5. Generate spatial anomaly map

---

## Performance Comparison (1920x1080 frame)

| Method | Time (ms) | Memory | GPU? |
|--------|-----------|--------|------|
| ECC | ~50-100 | Low | Optional |
| Optical Flow | ~150-300 | Medium | Optional |
| Phase Correlation | ~20-40 | Low | No |
| ORB | ~80-150 | Low | No |
| SIFT | ~200-400 | Medium | No |
| Similarity | ~100-200 | Low | No |
| Template Match | ~500-2000 | High | No |

---

## Recommended Settings by Use Case

### F1 Onboard Video (Your Case)
```python
alignment_method='ecc'
warp_mode=cv2.MOTION_EUCLIDEAN  # Translation + rotation
```
**Why:** Camera shake, driver movement, similar frames

### Driver Motion Tracking
```python
alignment_method='optical_flow'
```
**Why:** Non-rigid body movements, deformations

### Camera Shake Only
```python
alignment_method='phase'
```
**Why:** Fastest, sufficient for translation

### General Purpose
```python
alignment_method='ecc'
warp_mode=cv2.MOTION_AFFINE
```
**Why:** Handles most transformations, balanced speed/accuracy

---

## How Alignment Fixes Your Issue

**BEFORE (Resizing):**
```
Reference: [Driver at position A, orientation X]
Current:   [Driver at position B, orientation Y]
         ↓
Patches extracted from different spatial locations
         ↓
Feature comparison fails (comparing background to driver)
         ↓
False positives everywhere
```

**AFTER (Alignment):**
```
Reference: [Driver at position A, orientation X]
Current:   [Driver at position B, orientation Y]
         ↓
Align current to reference (ECC warp)
         ↓
Current becomes: [Driver at position A, orientation X]
         ↓
Patches extracted from SAME spatial locations
         ↓
Feature comparison works (driver to driver, background to background)
         ↓
True anomalies detected (actual motion)
```

---

## Testing Your Setup

```python
# Test different alignment methods
from utils.alignment import (
    ecc_alignment,
    orb_alignment,
    phase_correlation_alignment,
    compute_alignment_quality
)

# Load two frames
ref = cv2.imread('frame_000.jpg')
test = cv2.imread('frame_001.jpg')

# Compare methods
methods = {
    'ECC': ecc_alignment(ref, test)[0],
    'ORB': orb_alignment(ref, test)[0],
    'Phase': phase_correlation_alignment(ref, test)[0]
}

# Evaluate quality
for name, aligned in methods.items():
    quality = compute_alignment_quality(ref, aligned)
    print(f"{name}: {quality:.4f}")
```

---

## Tips for Best Results

1. **Use grayscale** for alignment (faster, often better)
2. **Downsample** large frames before alignment (speed)
3. **Cache transforms** if frames are sequential
4. **Validate alignment** before patch extraction
5. **Fallback to 'none'** if alignment fails

## Summary

✅ **For F1 video:** Use `alignment_method='ecc'`  
✅ **For speed:** Use `alignment_method='phase'`  
✅ **For robustness:** Use `alignment_method='orb'`  
✅ **For motion analysis:** Use `alignment_method='optical_flow'`

The alignment happens **before patch extraction**, ensuring patches are spatially consistent between reference and current frames.
