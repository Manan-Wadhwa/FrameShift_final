# ============================================================================
# F1 DRIVER ONBOARD MOTION TRACKER V3.0
# MOG2 + PatchCore + PatchCore+SAM Hybrid Approach on Grayscale
# ============================================================================

import cv2
import numpy as np
import sys
import os
from collections import deque
import tkinter as tk
from tkinter import filedialog
import gc
from tqdm import tqdm
import logging
from datetime import datetime

print("🏎️ F1 DRIVER ONBOARD MOTION TRACKER V3.0")
print("="*70)
print("✅ Imports loaded")

# Setup detailed logging for PatchCore debugging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"f1_tracker_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("F1 Motion Tracker V3.0 initialized")
logger.info(f"Log file: {log_file}")

# Check and fix PyTorch CUDA if needed
try:
    import torch
    if not torch.cuda.is_available():
        print("\n⚠️ PyTorch CUDA not available!")
        print("   Reinstalling PyTorch with CUDA support...")
        import subprocess
        import sys
        
        # Try to reinstall with CUDA
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-q', '--upgrade',
                'torch', 'torchvision', 'torchaudio',
                '--index-url', 'https://download.pytorch.org/whl/cu118'
            ], timeout=300)
            
            # Reimport torch
            import importlib
            importlib.reload(torch)
            
            if torch.cuda.is_available():
                print("   ✅ PyTorch CUDA enabled after reinstall!")
            else:
                print("   ⚠️ CUDA still not available, continuing with CPU")
        except Exception as e:
            print(f"   ⚠️ Could not reinstall: {e}")
            print("   Continuing with current PyTorch installation...")
except Exception as e:
    print(f"⚠️ PyTorch check failed: {e}")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Video input
    'use_grayscale': False,              # Convert to grayscale
    'detection_approach': 'patchcore',     # 'mog2', 'patchcore', 'patchcore_sam', 'hybrid'
    
    # MOG2 Settings (for grayscale)
    'mog2_history': 500,
    'mog2_var_threshold': 25,
    'mog2_learning_rate': 0.001,
    
    # PatchCore Settings
    'patchcore_model': 'resnet18',      # Feature extractor
    'patchcore_k': 3,                   # KNN neighbors
    
    # PatchCore+SAM Settings
    'patchcore_sam_enabled': True,
    
    # Preprocessing
    'denoise_strength': 5,
    'apply_clahe': True,
    'clahe_clip_limit': 2.0,
    'clahe_grid_size': (8, 8),
    
    # Mask refinement
    'morphology_iterations': 2,
    'open_kernel_size': (3, 3),
    'close_kernel_size': (9, 9),
    'dilate_kernel_size': (5, 5),
    
    # Motion filtering
    'min_motion_area': 200,
    'max_motion_area': 50000,
    'temporal_smoothing': True,
    'smooth_window': 5,
    
    # ROI
    'use_roi': True,
    'roi_coords': None,
    'roi_padding': 0.1,
    
    # Visualization
    'output_mode': 'heatmap',           # 'mask', 'overlay', 'side_by_side', 'heatmap', 'comparison', 'all'
    'mask_color': (0, 255, 0),
    'overlay_alpha': 0.6,
    'show_contours': True,
    'contour_color': (0, 255, 255),
    'contour_thickness': 2,
    'show_trails': True,
    'trail_length': 15,
    'trail_color': (255, 0, 255),
    
    # Output
    'output_codec': 'mp4v',
    'show_preview': True,
    'preview_scale': 0.6,
    'save_debug_frames': False,
    'debug_interval': 30,
    
    # Performance
    'frame_skip': 1,                    # Process every Nth frame
    'max_frames': None,                 # Process only first N frames (None = all)
}

print("📋 Configuration loaded")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def select_video_gui():
    """Open file dialog to select video"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    # Start in video folder if it exists
    initial_dir = "video" if os.path.isdir("video") else os.getcwd()
    
    print("📂 Select F1 onboard video...")
    video_path = filedialog.askopenfilename(
        title="Select F1 Onboard Video",
        initialdir=initial_dir,
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return video_path

def cut_video_to_duration(input_path, output_path, duration_seconds=5):
    """Cut video to specified duration using ffmpeg"""
    try:
        import subprocess
        
        print(f"\n✂️ Cutting video to {duration_seconds} seconds...")
        
        # ffmpeg command
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-t', str(duration_seconds),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-c:a', 'aac',
            '-y',  # Overwrite output
            output_path
        ]
        
        # Run ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Video cut successfully: {output_path}")
            return True
        else:
            print(f"⚠️ ffmpeg warning: {result.stderr}")
            return True  # Still return True as ffmpeg may complete with warnings
    
    except FileNotFoundError:
        print("⚠️ ffmpeg not found. Attempting alternative method...")
        return cut_video_opencv(input_path, output_path, duration_seconds)
    except Exception as e:
        print(f"⚠️ Warning: Could not cut video with ffmpeg: {e}")
        return cut_video_opencv(input_path, output_path, duration_seconds)

def cut_video_opencv(input_path, output_path, duration_seconds=5):
    """Cut video using OpenCV as fallback"""
    try:
        print(f"✂️ Cutting video to {duration_seconds} seconds using OpenCV...")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {input_path}")
            return False
        
        # Get properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frames to cut
        frames_to_cut = int(duration_seconds * fps)
        frames_to_cut = min(frames_to_cut, total_frames)
        
        # Setup output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Write frames
        for i in range(frames_to_cut):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()
        
        print(f"✅ Video cut successfully: {output_path}")
        return True
    
    except Exception as e:
        print(f"❌ Could not cut video: {e}")
        return False

def convert_to_grayscale(frame):
    """Convert frame to grayscale"""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def preprocess_frame_grayscale(frame):
    """Preprocess grayscale frame"""
    processed = frame.copy()
    
    # 1. Denoise
    if CONFIG['denoise_strength'] > 0:
        processed = cv2.fastNlMeansDenoising(
            processed, None,
            CONFIG['denoise_strength'],
            7, 21
        )
    
    # 2. Enhance contrast with CLAHE
    if CONFIG['apply_clahe']:
        clahe = cv2.createCLAHE(
            clipLimit=CONFIG['clahe_clip_limit'],
            tileGridSize=CONFIG['clahe_grid_size']
        )
        processed = clahe.apply(processed)
    
    return processed

def refine_mask(mask):
    """Clean up and refine the motion mask"""
    refined = mask.copy()
    
    # 1. Morphological operations
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CONFIG['open_kernel_size'])
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CONFIG['close_kernel_size'])
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CONFIG['dilate_kernel_size'])
    
    for _ in range(CONFIG['morphology_iterations']):
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel_open)
    
    for _ in range(CONFIG['morphology_iterations']):
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel_close)
    
    refined = cv2.dilate(refined, kernel_dilate, iterations=1)
    
    # 2. Filter by area
    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(refined)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if CONFIG['min_motion_area'] < area < CONFIG['max_motion_area']:
            cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
    
    return filtered_mask, contours

def apply_temporal_smoothing(mask, mask_history):
    """Smooth mask over time"""
    mask_history.append(mask.astype(float) / 255.0)
    avg_mask = np.mean(mask_history, axis=0)
    smoothed = (avg_mask > 0.3).astype(np.uint8) * 255
    return smoothed

# ============================================================================
# MOG2 MOTION DETECTION
# ============================================================================

class MOG2Detector:
    """MOG2 background subtraction with optional PatchCore enhancement"""
    
    def __init__(self, use_patchcore=False, frame_height=None, frame_width=None, alignment_method='ecc'):
        logger.info("Initializing MOG2Detector...")
        
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=CONFIG['mog2_history'],
            varThreshold=CONFIG['mog2_var_threshold'],
            detectShadows=False
        )
        logger.info("   ✓ MOG2 background subtractor initialized")
        
        # Optional PatchCore enhancement
        self.use_patchcore = use_patchcore
        self.patchcore = None
        
        if use_patchcore and frame_height and frame_width:
            logger.info("   Initializing PatchCore enhancement for MOG2...")
            self.patchcore = PatchCoreDetector(frame_height, frame_width, alignment_method=alignment_method)
            print(f"   ✓ MOG2 enhanced with PatchCore (alignment: {alignment_method})")
        
        print(f"   ✓ MOG2 detector initialized (PatchCore: {use_patchcore})")
    
    def detect(self, frame_gray):
        """Detect motion using MOG2 with optional PatchCore enhancement"""
        # Base MOG2 detection
        fg_mask = self.bg_subtractor.apply(
            frame_gray,
            learningRate=CONFIG['mog2_learning_rate']
        )
        
        # Enhance with PatchCore if enabled
        if self.use_patchcore and self.patchcore:
            logger.debug("Enhancing MOG2 mask with PatchCore...")
            patchcore_mask = self.patchcore.detect(frame_gray)
            
            # Combine: Take union of both detections
            # MOG2 catches fast motion, PatchCore catches subtle changes
            combined = cv2.bitwise_or(fg_mask, patchcore_mask)
            logger.debug(f"Combined mask - MOG2: {np.count_nonzero(fg_mask)}, PatchCore: {np.count_nonzero(patchcore_mask)}, Combined: {np.count_nonzero(combined)}")
            
            return combined
        
        return fg_mask
    
    def get_name(self):
        return "MOG2+PatchCore" if self.use_patchcore else "MOG2"

# ============================================================================
# PATCHCORE MOTION DETECTION (Grayscale)
# ============================================================================

class PatchCoreDetector:
    """PatchCore anomaly detection on grayscale"""
    
    def __init__(self, frame_height, frame_width, alignment_method='ecc'):
        self.reference_frame = None
        self.frame_height = frame_height
        self.frame_width = frame_width
        
        # Patch configuration - larger patches for faster compute
        self.patch_size = 128  # Size of each patch (128x128) - larger for speed
        self.stride = 64       # Stride for overlapping patches
        self.memory_bank = []  # Store reference patch features
        self.patch_positions = []  # Store patch positions for mapping
        
        # Alignment configuration
        self.alignment_method = alignment_method  # 'ecc', 'orb', 'phase', 'optical_flow', 'none'
        self.use_alignment = alignment_method != 'none'
        
        try:
            from torchvision import models
            import torch
            
            self.torch = torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load ResNet18 for feature extraction
            self.model = models.resnet18(pretrained=True)
            self.model.eval()
            self.model.to(self.device)
            
            # Remove classification layer, keep features
            self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
            
            print(f"   ✓ PatchCore initialized on {self.device}")
            print(f"   ✓ Patch size: {self.patch_size}x{self.patch_size}, stride: {self.stride}")
            print(f"   ✓ Alignment method: {self.alignment_method}")
            self.initialized = True
        except Exception as e:
            print(f"   ✗ PatchCore initialization failed: {e}")
            self.initialized = False
    
    def extract_patches(self, frame_gray):
        """Extract overlapping patches from frame"""
        patches = []
        positions = []
        
        h, w = frame_gray.shape
        
        # Slide window across image
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = frame_gray[y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)
                positions.append((y, x))
        
        logger.debug(f"Extracted {len(patches)} patches from {h}x{w} frame")
        return patches, positions
    
    def extract_patch_features(self, patch):
        """Extract features from a single patch"""
        try:
            # Convert patch to RGB
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
            
            # Resize to model input size
            patch_resized = cv2.resize(patch_rgb, (224, 224))
            
            # Normalize
            patch_normalized = patch_resized.astype(np.float32) / 255.0
            patch_normalized = (patch_normalized - 0.5) / 0.5
            
            # Convert to tensor
            patch_tensor = self.torch.from_numpy(patch_normalized).permute(2, 0, 1).unsqueeze(0)
            patch_tensor = patch_tensor.to(self.device)
            
            # Extract features
            with self.torch.no_grad():
                features = self.feature_extractor(patch_tensor)
            
            # Flatten to 1D feature vector
            features_flat = features.cpu().numpy().flatten()
            
            return features_flat
        except Exception as e:
            logger.error(f"Patch feature extraction failed: {e}")
            return None
    
    def extract_features(self, frame_gray):
        """Extract features from grayscale frame"""
        if not self.initialized:
            logger.error("PatchCore not initialized")
            return None
        
        try:
            logger.debug("="*60)
            logger.debug("EXTRACT_FEATURES - Starting")
            logger.debug(f"Input frame_gray shape: {frame_gray.shape}, dtype: {frame_gray.dtype}")
            
            # Prepare frame for model
            frame_rgb = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
            logger.debug(f"After GRAY2BGR: {frame_rgb.shape}")
            
            frame_resized = cv2.resize(frame_rgb, (224, 224))
            logger.debug(f"After resize to (224,224): {frame_resized.shape}")
            
            # Normalize
            frame_normalized = frame_resized.astype(np.float32) / 255.0
            logger.debug(f"After normalization [0,1]: dtype={frame_normalized.dtype}, min={frame_normalized.min():.4f}, max={frame_normalized.max():.4f}")
            
            frame_normalized = (frame_normalized - 0.5) / 0.5
            logger.debug(f"After normalization [-1,1]: min={frame_normalized.min():.4f}, max={frame_normalized.max():.4f}")
            
            # Convert to tensor
            logger.debug(f"Pre-permute shape: {frame_normalized.shape}")
            frame_tensor = self.torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)
            logger.debug(f"After permute+unsqueeze: {frame_tensor.shape}")
            
            if frame_tensor.shape != (1, 3, 224, 224):
                error_msg = f"DIMENSION ERROR: Expected (1,3,224,224), got {frame_tensor.shape}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            frame_tensor = frame_tensor.to(self.device)
            logger.debug(f"Tensor moved to device: {self.device}")
            
            # Extract features
            with self.torch.no_grad():
                logger.debug("Running feature extractor...")
                features = self.feature_extractor(frame_tensor)
                logger.debug(f"Features extracted: {features.shape}")
            
            features_np = features.cpu().numpy()
            logger.debug(f"Features converted to numpy: {features_np.shape}")
            logger.debug("EXTRACT_FEATURES - Complete")
            logger.debug("="*60)
            
            return features_np
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}", exc_info=True)
            print(f"   ✗ Feature extraction failed: {e}")
            return None
    
    def align_frame(self, frame_gray):
        """Align current frame to reference using selected method"""
        if not self.use_alignment or self.reference_frame is None:
            return frame_gray
        
        try:
            from utils.alignment import ecc_alignment, orb_alignment, phase_correlation_alignment, optical_flow_alignment
            
            # Convert to BGR for alignment functions
            frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
            ref_bgr = cv2.cvtColor(self.reference_frame, cv2.COLOR_GRAY2BGR)
            
            if self.alignment_method == 'ecc':
                aligned_bgr, _, _ = ecc_alignment(ref_bgr, frame_bgr, warp_mode=cv2.MOTION_EUCLIDEAN)
            elif self.alignment_method == 'orb':
                aligned_bgr, _ = orb_alignment(ref_bgr, frame_bgr, transform_type='similarity')
            elif self.alignment_method == 'phase':
                aligned_bgr, _, _ = phase_correlation_alignment(ref_bgr, frame_bgr)
            elif self.alignment_method == 'optical_flow':
                aligned_bgr, _ = optical_flow_alignment(ref_bgr, frame_bgr)
            else:
                aligned_bgr = frame_bgr
            
            # Convert back to grayscale
            aligned = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
            logger.debug(f"Frame aligned using {self.alignment_method}")
            return aligned
        except Exception as e:
            logger.warning(f"Alignment failed: {e}, using original frame")
            return frame_gray
    
    def detect(self, frame_gray):
        """Detect anomalies using patch-based PatchCore"""
        if not self.initialized:
            logger.warning("PatchCore not initialized, returning zero mask")
            return np.zeros((frame_gray.shape[0], frame_gray.shape[1]), dtype=np.uint8)
        
        try:
            logger.info("="*60)
            logger.info("PATCHCORE DETECT - Starting (Patch-based)")
            logger.info(f"Input frame shape: {frame_gray.shape}, dtype: {frame_gray.dtype}")
            
            # Initialize memory bank with first frame
            if len(self.memory_bank) == 0:
                logger.info("Building memory bank from reference frame...")
                self.reference_frame = frame_gray.copy()
                
                # Extract patches from reference frame
                current_patches, current_positions = self.extract_patches(frame_gray)
                self.patch_positions = current_positions
                
                for i, patch in enumerate(current_patches):
                    if i % 100 == 0:
                        logger.debug(f"Processing patch {i}/{len(current_patches)}")
                    features = self.extract_patch_features(patch)
                    if features is not None:
                        self.memory_bank.append(features)
                
                logger.info(f"Memory bank built: {len(self.memory_bank)} patch features")
                return np.zeros((frame_gray.shape[0], frame_gray.shape[1]), dtype=np.uint8)
            
            # Align current frame to reference
            if self.use_alignment:
                logger.debug(f"Aligning frame using {self.alignment_method}...")
                aligned_frame = self.align_frame(frame_gray)
            else:
                aligned_frame = frame_gray
            
            # Extract patches from aligned frame
            current_patches, current_positions = self.extract_patches(aligned_frame)
            logger.debug(f"Extracted {len(current_patches)} patches from aligned frame")
            
            # Compute anomaly score for each patch
            logger.debug("Computing patch-level anomaly scores...")
            anomaly_scores = []
            
            for i, patch in enumerate(current_patches):
                # Extract features for current patch
                patch_features = self.extract_patch_features(patch)
                if patch_features is None:
                    anomaly_scores.append(0)
                    continue
                
                # Find distance to nearest neighbor in memory bank
                if len(self.memory_bank) > 0:
                    min_distance = float('inf')
                    for ref_features in self.memory_bank:
                        distance = np.linalg.norm(patch_features - ref_features)
                        min_distance = min(min_distance, distance)
                    anomaly_scores.append(min_distance)
                else:
                    anomaly_scores.append(0)
            
            logger.debug(f"Computed {len(anomaly_scores)} anomaly scores")
            logger.debug(f"Score stats - min: {min(anomaly_scores):.2f}, max: {max(anomaly_scores):.2f}, mean: {np.mean(anomaly_scores):.2f}")
            
            # Create spatial anomaly map
            anomaly_map = np.zeros(frame_gray.shape, dtype=np.float32)
            count_map = np.zeros(frame_gray.shape, dtype=np.int32)
            
            # Map patch scores back to image coordinates
            for (y, x), score in zip(current_positions, anomaly_scores):
                anomaly_map[y:y+self.patch_size, x:x+self.patch_size] += score
                count_map[y:y+self.patch_size, x:x+self.patch_size] += 1
            
            # Average overlapping regions
            mask = count_map > 0
            anomaly_map[mask] = anomaly_map[mask] / count_map[mask]
            
            # Normalize to 0-255
            if anomaly_map.max() > 0:
                anomaly_map = (anomaly_map / anomaly_map.max() * 255).astype(np.uint8)
            else:
                anomaly_map = anomaly_map.astype(np.uint8)
            
            logger.debug(f"Anomaly map - min: {anomaly_map.min()}, max: {anomaly_map.max()}, mean: {anomaly_map.mean():.2f}")
            
            # Apply adaptive threshold
            mean_score = anomaly_map[anomaly_map > 0].mean() if np.any(anomaly_map > 0) else 0
            std_score = anomaly_map[anomaly_map > 0].std() if np.any(anomaly_map > 0) else 0
            threshold_value = max(10, min(150, mean_score + 2.0 * std_score))
            logger.debug(f"Adaptive threshold: {threshold_value:.2f} (mean: {mean_score:.2f}, std: {std_score:.2f})")
            
            # Create binary mask
            _, binary_map = cv2.threshold(anomaly_map, threshold_value, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations to reduce noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel)
            binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)
            
            logger.debug(f"Binary map: non-zero pixels={np.count_nonzero(binary_map)}")
            logger.info("PATCHCORE DETECT - Complete")
            logger.info("="*60)
            
            return binary_map
        except Exception as e:
            logger.error(f"PatchCore detection failed: {e}", exc_info=True)
            print(f"   ✗ PatchCore detection failed: {e}")
            return np.zeros((frame_gray.shape[0], frame_gray.shape[1]), dtype=np.uint8)
    
    def get_name(self):
        return "PatchCore"

# ============================================================================
# PATCHCORE + SAM DETECTION (Grayscale)
# ============================================================================

class PatchCoreSAMDetector:
    """PatchCore + SAM anomaly detection on grayscale"""
    
    def __init__(self, frame_height, frame_width, alignment_method='ecc'):
        self.reference_frame = None
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.initialized = False
        self.device = None
        
        # Alignment configuration
        self.alignment_method = alignment_method
        self.use_alignment = alignment_method != 'none'
        
        try:
            import torch
            from pipelines.anomaly_patchcore_sam import run_patchcore_sam_pipeline
            from utils.sam_refine import sam_refine
            from utils.rough_mask import generate_rough_mask
            
            self.run_patchcore_sam_pipeline = run_patchcore_sam_pipeline
            self.sam_refine = sam_refine
            self.generate_rough_mask = generate_rough_mask
            
            # Set device to GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"   ✓ PatchCore+SAM initialized on {self.device}")
            print(f"   ✓ Alignment method: {self.alignment_method}")
            
            # Force SAM to use GPU by setting torch default device
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                print(f"   ✓ GPU allocated for SAM - CUDA enabled")
                # Verify GPU is being used
                if torch.cuda.is_available():
                    print(f"   ✓ GPU Device: {torch.cuda.get_device_name(0)}")
                    print(f"   ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                print(f"   ⚠️ CUDA not available, using CPU")
            
            self.initialized = True
        except Exception as e:
            print(f"   ✗ PatchCore+SAM initialization failed: {e}")
            self.initialized = False
    
    def align_frame(self, frame_bgr):
        """Align current frame to reference using selected method"""
        if not self.use_alignment or self.reference_frame is None:
            return frame_bgr
        
        try:
            from utils.alignment import ecc_alignment, orb_alignment, phase_correlation_alignment, optical_flow_alignment
            
            if self.alignment_method == 'ecc':
                aligned_bgr, _, _ = ecc_alignment(self.reference_frame, frame_bgr, warp_mode=cv2.MOTION_EUCLIDEAN)
            elif self.alignment_method == 'orb':
                aligned_bgr, _ = orb_alignment(self.reference_frame, frame_bgr, transform_type='similarity')
            elif self.alignment_method == 'phase':
                aligned_bgr, _, _ = phase_correlation_alignment(self.reference_frame, frame_bgr)
            elif self.alignment_method == 'optical_flow':
                aligned_bgr, _ = optical_flow_alignment(self.reference_frame, frame_bgr)
            else:
                aligned_bgr = frame_bgr
            
            logger.debug(f"Frame aligned using {self.alignment_method}")
            return aligned_bgr
        except Exception as e:
            logger.warning(f"Alignment failed: {e}, using original frame")
            return frame_bgr
    
    def detect(self, frame_gray):
        """Detect anomalies using PatchCore + SAM"""
        if not self.initialized:
            logger.warning("PatchCore+SAM not initialized, returning zero mask")
            return np.zeros((frame_gray.shape[0], frame_gray.shape[1]), dtype=np.uint8)
        
        try:
            logger.info("="*60)
            logger.info("PATCHCORE+SAM DETECT - Starting")
            logger.info(f"Input frame_gray shape: {frame_gray.shape}, dtype: {frame_gray.dtype}")
            
            # Convert grayscale to BGR for pipeline
            frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
            logger.debug(f"After GRAY2BGR: {frame_bgr.shape}")
            
            # Initialize reference on first frame
            if self.reference_frame is None:
                logger.info("Initializing reference frame (first frame)")
                self.reference_frame = frame_bgr.copy()
                logger.debug(f"Reference frame stored: {self.reference_frame.shape}")
                return np.zeros((frame_gray.shape[0], frame_gray.shape[1]), dtype=np.uint8)
            
            # Align current frame to reference
            if self.use_alignment:
                logger.debug(f"Aligning frame using {self.alignment_method}...")
                frame_bgr = self.align_frame(frame_bgr)
                logger.debug(f"Frame aligned: {frame_bgr.shape}")
            
            logger.debug(f"Reference frame shape: {self.reference_frame.shape}")
            
            # Generate rough mask from difference
            logger.debug("Generating rough mask...")
            rough_mask, _ = self.generate_rough_mask(self.reference_frame, frame_bgr)
            logger.debug(f"Rough mask: shape={rough_mask.shape}, non-zero={np.count_nonzero(rough_mask)}")
            
            # Refine with SAM
            logger.debug("Refining with SAM...")
            refined_mask = self.sam_refine(frame_bgr, rough_mask)
            logger.debug(f"Refined mask: shape={refined_mask.shape}, non-zero={np.count_nonzero(refined_mask)}")
            
            # Run PatchCore + SAM
            logger.info("Running PatchCore+SAM pipeline...")
            result = self.run_patchcore_sam_pipeline(
                self.reference_frame, refined_mask, ref_img=self.reference_frame
            )
            
            if result:
                logger.debug(f"Pipeline result keys: {result.keys()}")
                heatmap = result.get("heatmap")
                if heatmap is not None:
                    logger.debug(f"Heatmap shape: {heatmap.shape}")
                    # Convert to grayscale if needed
                    if len(heatmap.shape) == 3:
                        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
                        logger.debug(f"Heatmap converted to grayscale: {heatmap.shape}")
                    
                    logger.info(f"PATCHCORE+SAM DETECT - Complete (non-zero pixels: {np.count_nonzero(heatmap)})")
                    logger.info("="*60)
                    return heatmap
                else:
                    logger.warning("Pipeline result has no heatmap")
            else:
                logger.warning("Pipeline returned None result")
            
            logger.info("PATCHCORE+SAM DETECT - Returning zero mask")
            logger.info("="*60)
            return np.zeros((frame_gray.shape[0], frame_gray.shape[1]), dtype=np.uint8)
        except Exception as e:
            logger.error(f"PatchCore+SAM detection failed: {e}", exc_info=True)
            print(f"   ✗ PatchCore+SAM detection failed: {e}")
            return np.zeros((frame_gray.shape[0], frame_gray.shape[1]), dtype=np.uint8)
    
    def get_name(self):
        return "PatchCore+SAM"

# ============================================================================
# HYBRID DETECTOR
# ============================================================================

class HybridDetector:
    """Combine MOG2+PatchCore, PatchCore, and PatchCore+SAM"""
    
    def __init__(self, frame_height, frame_width, alignment_method='ecc'):
        logger.info("Initializing HybridDetector...")
        
        # MOG2 with PatchCore enhancement
        self.mog2 = MOG2Detector(use_patchcore=True, frame_height=frame_height, frame_width=frame_width, alignment_method=alignment_method)
        logger.info("   ✓ MOG2+PatchCore initialized")
        
        self.patchcore = PatchCoreDetector(frame_height, frame_width, alignment_method=alignment_method)
        logger.info("   ✓ PatchCore initialized")
        
        self.patchcore_sam = PatchCoreSAMDetector(frame_height, frame_width, alignment_method=alignment_method)
        logger.info("   ✓ PatchCore+SAM initialized")
        
        print(f"   ✓ Hybrid detector initialized with alignment: {alignment_method}")
        print(f"   ✓ Combines: MOG2+PatchCore, PatchCore, PatchCore+SAM")
    
    def detect(self, frame_gray):
        """Detect using all three methods"""
        mog2_mask = self.mog2.detect(frame_gray)
        patchcore_mask = self.patchcore.detect(frame_gray)
        patchcore_sam_mask = self.patchcore_sam.detect(frame_gray)
        
        return {
            'mog2': mog2_mask,
            'patchcore': patchcore_mask,
            'patchcore_sam': patchcore_sam_mask
        }
    
    def combine_masks(self, masks):
        """Combine all masks intelligently"""
        # Average the three masks
        combined = (
            masks['mog2'].astype(float) / 3.0 +
            masks['patchcore'].astype(float) / 3.0 +
            masks['patchcore_sam'].astype(float) / 3.0
        ).astype(np.uint8)
        
        return combined

# ============================================================================
# VISUALIZATION CLASS
# ============================================================================

class DriverMotionVisualizer:
    """Elegant visualization of driver motion"""
    
    def __init__(self):
        self.motion_trails = deque(maxlen=CONFIG['trail_length'])
        self.contour_history = deque(maxlen=5)
    
    def create_overlay_gray(self, frame_gray, mask):
        """Create overlay on grayscale"""
        # Convert grayscale to BGR
        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        
        # Apply colored mask
        colored_mask = np.zeros_like(frame_bgr)
        colored_mask[mask > 0] = CONFIG['mask_color']
        
        # Blend
        result = cv2.addWeighted(frame_bgr, 1.0, colored_mask, CONFIG['overlay_alpha'], 0)
        
        return result
    
    def create_heatmap_gray(self, frame_gray, mask):
        """Create heatmap from mask"""
        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        
        # Convert mask to colormap
        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        
        # Blend
        result = cv2.addWeighted(frame_bgr, 0.6, heatmap, 0.4, 0)
        
        return result
    
    def create_side_by_side_gray(self, frame_gray, mask):
        """Side-by-side comparison"""
        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        result = np.hstack([frame_bgr, mask_3ch])
        
        return result
    
    def create_comparison(self, frame_gray, masks_dict):
        """Create 4-panel comparison (MOG2, PatchCore, PatchCore+SAM, Hybrid)"""
        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        
        # Get masks
        mog2_mask = masks_dict.get('mog2', np.zeros_like(frame_gray))
        patchcore_mask = masks_dict.get('patchcore', np.zeros_like(frame_gray))
        patchcore_sam_mask = masks_dict.get('patchcore_sam', np.zeros_like(frame_gray))
        
        # Combine
        hybrid_mask = (
            mog2_mask.astype(float) / 3.0 +
            patchcore_mask.astype(float) / 3.0 +
            patchcore_sam_mask.astype(float) / 3.0
        ).astype(np.uint8)
        
        # Create 4 panels with labels
        h, w = frame_gray.shape
        panel_h, panel_w = h // 2, w // 2
        
        # Resize masks
        mog2_resized = cv2.resize(mog2_mask, (panel_w, panel_h))
        patchcore_resized = cv2.resize(patchcore_mask, (panel_w, panel_h))
        patchcore_sam_resized = cv2.resize(patchcore_sam_mask, (panel_w, panel_h))
        hybrid_resized = cv2.resize(hybrid_mask, (panel_w, panel_h))
        
        # Convert to BGR
        mog2_bgr = cv2.cvtColor(mog2_resized, cv2.COLOR_GRAY2BGR)
        patchcore_bgr = cv2.cvtColor(patchcore_resized, cv2.COLOR_GRAY2BGR)
        patchcore_sam_bgr = cv2.cvtColor(patchcore_sam_resized, cv2.COLOR_GRAY2BGR)
        hybrid_bgr = cv2.cvtColor(hybrid_resized, cv2.COLOR_GRAY2BGR)
        
        # Add labels
        cv2.putText(mog2_bgr, "MOG2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(patchcore_bgr, "PatchCore", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(patchcore_sam_bgr, "PatchCore+SAM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(hybrid_bgr, "Hybrid (All 3)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Stack
        top_half = np.hstack([mog2_bgr, patchcore_bgr])
        bottom_half = np.hstack([patchcore_sam_bgr, hybrid_bgr])
        result = np.vstack([top_half, bottom_half])
        
        return result
    
    def draw_contours(self, frame, contours):
        """Draw motion contours"""
        if CONFIG['show_contours'] and contours:
            valid_contours = [c for c in contours if cv2.contourArea(c) > CONFIG['min_motion_area']]
            cv2.drawContours(frame, valid_contours, -1,
                           CONFIG['contour_color'], CONFIG['contour_thickness'])
    
    def add_info_panel(self, frame, frame_num, total_frames, motion_percentage, method):
        """Add information panel"""
        h, w = frame.shape[:2]
        
        # Semi-transparent panel
        panel = frame.copy()
        cv2.rectangle(panel, (10, h - 150), (450, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(panel, 0.5, frame, 0.5, 0, frame)
        
        # Text info
        info_lines = [
            f"Method: {method}",
            f"Frame: {frame_num}/{total_frames}",
            f"Progress: {(frame_num/total_frames)*100:.1f}%",
            f"Motion: {motion_percentage:.1f}%",
        ]
        
        y_offset = h - 130
        for line in info_lines:
            cv2.putText(frame, line, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
    
    def visualize(self, frame_gray, mask, contours, frame_num, total_frames, method, output_mode=None):
        """Create final visualization"""
        # Calculate motion percentage
        motion_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        motion_percentage = (motion_pixels / total_pixels) * 100
        
        # Use specified mode or default from CONFIG
        mode = output_mode if output_mode else CONFIG['output_mode']
        
        # Create visualization
        if mode == 'mask':
            result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        elif mode == 'overlay':
            result = self.create_overlay_gray(frame_gray, mask)
            self.draw_contours(result, contours)
        elif mode == 'heatmap':
            result = self.create_heatmap_gray(frame_gray, mask)
        elif mode == 'side_by_side':
            result = self.create_side_by_side_gray(frame_gray, mask)
        else:
            result = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        
        # Add info
        self.add_info_panel(result, frame_num, total_frames, motion_percentage, method)
        
        return result

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_video_hybrid(input_video_path, output_video_path, detection_method):
    """Process video with selected detection method"""
    
    # Open video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video: {input_video_path}")
        return False
    
    # Get properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_width == 0 or frame_height == 0:
        print("❌ Error: Invalid video dimensions")
        cap.release()
        return False
    
    print(f"\n📹 Video Properties:")
    print(f"   Resolution: {frame_width}x{frame_height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.1f}s")
    
    # Limit frames if specified
    frames_to_process = total_frames
    if CONFIG['max_frames']:
        frames_to_process = min(total_frames, CONFIG['max_frames'])
    
    # Initialize detectors
    print(f"\n🔧 Initializing {detection_method} detector...")
    
    # Alignment method for PatchCore-based detectors
    alignment_method = 'ecc'  # ECC is best for video frames
    
    if detection_method == 'mog2':
        # Option to enable PatchCore enhancement for MOG2
        use_patchcore_enhancement = False  # Set to True for MOG2+PatchCore
        detector = MOG2Detector(use_patchcore=use_patchcore_enhancement, 
                               frame_height=frame_height, 
                               frame_width=frame_width,
                               alignment_method=alignment_method)
        method_name = "MOG2+PatchCore" if use_patchcore_enhancement else "MOG2"
    elif detection_method == 'patchcore':
        detector = PatchCoreDetector(frame_height, frame_width, alignment_method=alignment_method)
        method_name = "PatchCore (Patch-based with ECC alignment)"
    elif detection_method == 'patchcore_sam':
        detector = PatchCoreSAMDetector(frame_height, frame_width, alignment_method=alignment_method)
        method_name = "PatchCore+SAM (with ECC alignment)"
    elif detection_method == 'hybrid':
        detector = HybridDetector(frame_height, frame_width, alignment_method=alignment_method)
        method_name = "Hybrid (MOG2+PatchCore, PatchCore, PatchCore+SAM with ECC alignment)"
    else:
        detector = MOG2Detector()
        method_name = "MOG2"
    
    # Setup output video(s)
    fourcc = cv2.VideoWriter_fourcc(*CONFIG['output_codec'])
    
    # Check if multi-output mode
    multi_output = CONFIG['output_mode'] == 'all'
    
    if multi_output:
        # Create 4 separate output files
        base_path = output_video_path.rsplit('.', 1)[0]
        ext = output_video_path.rsplit('.', 1)[1] if '.' in output_video_path else 'mp4'
        
        outputs = {
            'mask': cv2.VideoWriter(f"{base_path}_mask.{ext}", fourcc, fps, (frame_width, frame_height)),
            'overlay': cv2.VideoWriter(f"{base_path}_overlay.{ext}", fourcc, fps, (frame_width, frame_height)),
            'heatmap': cv2.VideoWriter(f"{base_path}_heatmap.{ext}", fourcc, fps, (frame_width, frame_height)),
            'side_by_side': cv2.VideoWriter(f"{base_path}_sidebyside.{ext}", fourcc, fps, (frame_width * 2, frame_height))
        }
        
        print(f"\n✅ Multi-Output Mode - Generating 4 videos:")
        print(f"   1. {base_path}_mask.{ext}")
        print(f"   2. {base_path}_overlay.{ext}")
        print(f"   3. {base_path}_heatmap.{ext}")
        print(f"   4. {base_path}_sidebyside.{ext}")
        print(f"   Method: {method_name}")
    else:
        # Single output mode
        output_width = frame_width
        output_height = frame_height
        
        if CONFIG['output_mode'] == 'side_by_side':
            output_width = frame_width * 2
        elif CONFIG['output_mode'] == 'comparison':
            output_width = frame_width
            output_height = frame_height
        
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))
        
        print(f"\n✅ Output: {output_video_path}")
        print(f"   Mode: {CONFIG['output_mode']}")
        print(f"   Method: {method_name}")
    
    # Initialize visualization
    visualizer = DriverMotionVisualizer()
    mask_history = deque(maxlen=CONFIG['smooth_window'])
    
    print(f"\n🚀 Processing video...")
    print("="*70)
    
    frame_count = 0
    processed_count = 0
    
    # Create progress bar
    pbar = tqdm(total=frames_to_process, desc="Processing frames", unit="frame", ncols=80)
    
    try:
        while cap.isOpened() and processed_count < frames_to_process:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Skip frames if specified
            if frame_count % CONFIG['frame_skip'] != 0:
                frame_count += 1
                continue
            
            # Convert to grayscale
            frame_gray = convert_to_grayscale(frame)
            
            # Preprocess
            processed_gray = preprocess_frame_grayscale(frame_gray)
            
            # Detect based on method
            if detection_method == 'hybrid':
                masks_dict = detector.detect(processed_gray)
                combined_mask = detector.combine_masks(masks_dict)
                
                # Refine
                refined_mask, contours = refine_mask(combined_mask)
                
                # Visualize with all masks
                if CONFIG['output_mode'] == 'comparison' and not multi_output:
                    output_frame = visualizer.create_comparison(frame_gray, masks_dict)
                    out.write(output_frame)
                elif multi_output:
                    # Generate all 4 visualization types
                    for mode, writer in outputs.items():
                        vis_frame = visualizer.visualize(
                            frame_gray, refined_mask, contours,
                            processed_count + 1, frames_to_process, method_name,
                            output_mode=mode
                        )
                        writer.write(vis_frame)
                else:
                    output_frame = visualizer.visualize(
                        frame_gray, refined_mask, contours,
                        processed_count + 1, frames_to_process, method_name
                    )
                    out.write(output_frame)
            else:
                fg_mask = detector.detect(processed_gray)
                
                # Refine
                refined_mask, contours = refine_mask(fg_mask)
                
                # Temporal smoothing
                if CONFIG['temporal_smoothing']:
                    refined_mask = apply_temporal_smoothing(refined_mask, mask_history)
                
                # Visualize
                if multi_output:
                    # Generate all 4 visualization types
                    for mode, writer in outputs.items():
                        vis_frame = visualizer.visualize(
                            frame_gray, refined_mask, contours,
                            processed_count + 1, frames_to_process, method_name,
                            output_mode=mode
                        )
                        writer.write(vis_frame)
                else:
                    output_frame = visualizer.visualize(
                        frame_gray, refined_mask, contours,
                        processed_count + 1, frames_to_process, method_name
                    )
                    out.write(output_frame)
            
            # Show preview
            if CONFIG['show_preview']:
                try:
                    preview = cv2.resize(output_frame, None,
                                       fx=CONFIG['preview_scale'],
                                       fy=CONFIG['preview_scale'])
                    cv2.imshow('F1 Driver Motion Tracker V3', preview)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n⏹️ Stopped by user")
                        break
                except cv2.error as e:
                    # GUI not available, just skip preview
                    print(f"\n⚠️ Preview not available (headless environment)")
                    CONFIG['show_preview'] = False
            
            # Update progress bar
            processed_count += 1
            pbar.update(1)
            pbar.set_postfix({"method": method_name})
            
            frame_count += 1
            
            # Garbage collection and GPU cache clearing
            if processed_count % 100 == 0:
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
    
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    
    finally:
        pbar.close()
        cap.release()
        
        # Release video writer(s)
        if multi_output:
            for writer in outputs.values():
                writer.release()
        else:
            out.release()
        
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # GUI not available, skip cleanup
            pass
        
        # Clear GPU cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("\n   ✓ GPU memory cleared")
        except:
            pass
    
    print("\n" + "="*70)
    print(f"✅ Processing complete!")
    print(f"   Processed {processed_count} frames")
    print(f"   Output saved: {output_video_path}")
    print("="*70)
    
    return True

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🏁 F1 DRIVER ONBOARD MOTION TRACKER V3.0")
    print("   Hybrid Motion Detection System")
    print("="*70 + "\n")
    
    # Grayscale option
    grayscale_opt = input("📺 Convert video to grayscale? (y/n) [y]: ").strip().lower()
    CONFIG['use_grayscale'] = grayscale_opt != 'n'
    
    # Detection method
    print("\n🔍 DETECTION METHODS:")
    print("   1. MOG2 (Fast, motion-focused)")
    print("   2. PatchCore (Moderate, anomaly-focused)")
    print("   3. PatchCore+SAM (Slow, detailed anomaly detection)")
    print("   4. Hybrid (Combine all 3)")
    
    method_choice = input("\nSelect method (1-4) or press Enter for default [4]: ").strip()
    
    if method_choice == "1":
        detection_method = 'mog2'
    elif method_choice == "2":
        detection_method = 'patchcore'
    elif method_choice == "3":
        detection_method = 'patchcore_sam'
    else:
        detection_method = 'hybrid'
    
    CONFIG['detection_approach'] = detection_method
    
    # Output mode
    print("\n🎨 OUTPUT MODES:")
    print("   1. Overlay (colored motion on video)")
    print("   2. Heatmap (thermal-style)")
    print("   3. Side-by-Side (original + mask)")
    print("   4. Mask Only (binary)")
    if detection_method == 'hybrid':
        print("   5. Comparison (4-panel all methods)")
    print("   6. ALL (generates mask, overlay, heatmap, side-by-side in one pass)")
    
    viz_choice = input("\nSelect mode (1-6) or press Enter for default [1]: ").strip()
    
    if viz_choice == "2":
        CONFIG['output_mode'] = 'heatmap'
    elif viz_choice == "3":
        CONFIG['output_mode'] = 'side_by_side'
    elif viz_choice == "4":
        CONFIG['output_mode'] = 'mask'
    elif viz_choice == "5" and detection_method == 'hybrid':
        CONFIG['output_mode'] = 'comparison'
    elif viz_choice == "6":
        CONFIG['output_mode'] = 'all'
    else:
        CONFIG['output_mode'] = 'overlay'
    
    # Video selection
    use_gui = input("\n📂 Use file dialog to select video? (y/n) [y]: ").strip().lower()
    
    if use_gui != 'n':
        INPUT_VIDEO = select_video_gui()
        if not INPUT_VIDEO:
            print("❌ No video selected. Exiting.")
            sys.exit(1)
    else:
        video_input = input("Enter video name or path (press Enter to list video folder): ").strip().strip('"')
        
        # If empty, list videos in video folder
        if not video_input:
            if os.path.isdir("video"):
                print("\n📂 Videos in ./video/:")
                videos = [f for f in os.listdir("video") if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                for i, v in enumerate(videos, 1):
                    print(f"   {i}. {v}")
                if videos:
                    choice = input("\nSelect video number: ").strip()
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(videos):
                            INPUT_VIDEO = os.path.join("video", videos[idx])
                        else:
                            print("❌ Invalid selection.")
                            sys.exit(1)
                    except ValueError:
                        print("❌ Invalid input.")
                        sys.exit(1)
                else:
                    print("❌ No videos found in ./video/")
                    sys.exit(1)
            else:
                print("❌ video/ folder not found.")
                sys.exit(1)
        else:
            # Check if it's in video folder
            if not os.path.exists(video_input):
                video_path_in_folder = os.path.join("video", video_input)
                if os.path.exists(video_path_in_folder):
                    INPUT_VIDEO = video_path_in_folder
                else:
                    INPUT_VIDEO = video_input
            else:
                INPUT_VIDEO = video_input
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]
    method_suffix = {
        'mog2': 'mog2',
        'patchcore': 'patchcore',
        'patchcore_sam': 'patchcore_sam',
        'hybrid': 'hybrid'
    }.get(detection_method, 'tracked')
    
    OUTPUT_VIDEO = f"{base_name}_{method_suffix}_v3.mp4"
    
    # Cut video to 5 seconds
    CUT_VIDEO = f"{base_name}_5sec.mp4"
    print(f"\n✂️ Creating 5-second cut of video...")
    cut_success = cut_video_to_duration(INPUT_VIDEO, CUT_VIDEO, duration_seconds=5)
    
    if cut_success:
        INPUT_VIDEO = CUT_VIDEO
        print(f"   Using cut video: {INPUT_VIDEO}")
    else:
        print(f"⚠️ Continuing with original video: {INPUT_VIDEO}")
    
    print(f"\n📹 Input: {INPUT_VIDEO}")
    if CONFIG['output_mode'] == 'all':
        base_path = OUTPUT_VIDEO.rsplit('.', 1)[0]
        ext = OUTPUT_VIDEO.rsplit('.', 1)[1] if '.' in OUTPUT_VIDEO else 'mp4'
        print(f"💾 Outputs (4 files):")
        print(f"   1. {base_path}_mask.{ext}")
        print(f"   2. {base_path}_overlay.{ext}")
        print(f"   3. {base_path}_heatmap.{ext}")
        print(f"   4. {base_path}_sidebyside.{ext}")
    else:
        print(f"💾 Output: {OUTPUT_VIDEO}")
    print(f"🎯 Method: {detection_method.upper()}")
    print(f"🎨 Mode: {CONFIG['output_mode'].upper()}")
    
    # Confirm
    proceed = input("\n▶️ Start processing? (y/n) [y]: ").strip().lower()
    
    if proceed == 'n':
        print("❌ Processing cancelled.")
        sys.exit(0)
    
    # Process
    try:
        success = process_video_hybrid(INPUT_VIDEO, OUTPUT_VIDEO, detection_method)
        
        if success:
            print("\n🎉 SUCCESS! Your motion tracking video is ready!")
            if CONFIG['output_mode'] == 'all':
                base_path = OUTPUT_VIDEO.rsplit('.', 1)[0]
                ext = OUTPUT_VIDEO.rsplit('.', 1)[1] if '.' in OUTPUT_VIDEO else 'mp4'
                print(f"   📁 Generated 4 videos:")
                print(f"      1. {base_path}_mask.{ext}")
                print(f"      2. {base_path}_overlay.{ext}")
                print(f"      3. {base_path}_heatmap.{ext}")
                print(f"      4. {base_path}_sidebyside.{ext}")
            else:
                print(f"   Location: {OUTPUT_VIDEO}")
        else:
            print("\n❌ Processing failed. Check error messages above.")
    
    except FileNotFoundError:
        print(f"\n❌ ERROR: Video file not found: {INPUT_VIDEO}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("🏁 F1 DRIVER ONBOARD MOTION TRACKER V3.0 - Session Complete")
    print("="*70)
