"""
Configuration file for F1 Visual Difference Engine
Adjust these settings to tune pipeline performance
"""

# ============================
# PREPROCESSING SETTINGS
# ============================
PREPROCESSING = {
    "resize": (512, 512),          # Standard image size
    "median_blur_kernel": 3,       # Denoising kernel size
    "gamma": 1.2,                  # Lighting correction factor
}

# ============================
# ROUGH MASK SETTINGS
# ============================
ROUGH_MASK = {
    "morphology_kernel": (5, 5),   # Kernel for morphological operations
    "morph_iterations": 2,         # Number of closing/opening iterations
}

# ============================
# SAM SETTINGS
# ============================
SAM = {
    "checkpoint": "sam_vit_h_4b8939.pth",  # SAM model checkpoint path
    "model_type": "vit_h",                  # Options: vit_h, vit_l, vit_b
    "use_sam": True,                        # Set False to use morphological fallback
}

# ============================
# ROUTING THRESHOLDS
# ============================
ROUTING = {
    "texture_threshold": 500,      # Laplacian variance threshold
    "edge_threshold": 0.3,         # Edge density threshold
    "color_threshold": 0.15,       # Color shift threshold
}

# ============================
# PIPELINE SETTINGS
# ============================
PIPELINES = {
    "dino": {
        "model": "dinov2_vits14",  # DINOv2 model variant
        "enabled": True,
    },
    "clip": {
        "model": "ViT-B/32",       # CLIP model variant
        "window_size": 64,         # Sliding window size
        "stride": 32,              # Sliding window stride
        "enabled": True,
    },
    "patchcore": {
        "model": "wide_resnet50_2", # Feature extractor
        "k_neighbors": 3,           # K for k-NN distance
        "enabled": True,
    },
    "padim": {
        "model": "resnet18",        # Feature extractor
        "enabled": True,
    }
}

# ============================
# LLAVA SETTINGS
# ============================
LLAVA = {
    "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
    "max_tokens": 200,
    "use_llava": True,  # Set False to use rule-based reports
}

# ============================
# VISUALIZATION SETTINGS
# ============================
VISUALIZATION = {
    "heatmap_colormap": "jet",     # Semantic heatmap colormap
    "anomaly_colormap": "hot",     # Anomaly heatmap colormap
    "overlay_alpha": 0.4,          # Transparency for overlays
    "contour_color": (0, 255, 0),  # Mask contour color (Green)
    "contour_thickness": 2,        # Mask contour line thickness
}

# ============================
# PERFORMANCE SETTINGS
# ============================
PERFORMANCE = {
    "use_gpu": True,               # Use CUDA if available
    "max_image_size": (1024, 1024), # Maximum image dimensions
    "enable_fallback": True,       # Use classical CV if models fail
}
