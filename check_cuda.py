#!/usr/bin/env python
"""Check CUDA and PyTorch configuration"""

import sys
import subprocess

print("="*70)
print("🔍 CUDA & PyTorch Diagnostic Check")
print("="*70)

# Check Python
print(f"\n📌 Python: {sys.version}")
print(f"📌 Executable: {sys.executable}")

# Check PyTorch
try:
    import torch
    print(f"\n✓ PyTorch: {torch.__version__}")
    print(f"  Location: {torch.__file__}")
    
    # CUDA checks
    print(f"\n🔧 CUDA Status:")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"  Current Device: {torch.cuda.current_device()}")
        print(f"  Device Name: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
    else:
        print(f"  ⚠️ CUDA not available in PyTorch")
        print(f"  Check PyTorch installation: pip show torch")
        
except ImportError as e:
    print(f"✗ PyTorch not installed: {e}")

# Check NVIDIA GPU
print(f"\n🖥️ GPU Check:")
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print(f"  nvidia-smi output:")
        for line in result.stdout.strip().split('\n'):
            print(f"    {line}")
    else:
        print(f"  ⚠️ nvidia-smi failed: {result.stderr}")
except FileNotFoundError:
    print(f"  ⚠️ nvidia-smi not found in PATH")
except Exception as e:
    print(f"  ⚠️ Error running nvidia-smi: {e}")

# Check CUDA environment variables
print(f"\n📋 Environment Variables:")
import os
cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES']
for var in cuda_vars:
    value = os.environ.get(var, "Not set")
    print(f"  {var}: {value}")

print("\n" + "="*70)
print("💡 If CUDA not available:")
print("  1. Reinstall PyTorch with CUDA support:")
print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
print("  2. Or use torch nightly with CUDA 12.1:")
print("     pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121")
print("="*70)
