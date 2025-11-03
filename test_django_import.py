#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test PyTorch import in Django environment
在 Django 环境中测试 PyTorch 导入
"""

import os
import sys

print("=" * 70)
print("Testing PyTorch in Django Environment")
print("=" * 70)

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'map.settings')

import django
django.setup()

print(f"\n[1] Django setup complete")
print(f"   Django version: {django.__version__}")

# Now try to import from image_detection_package
print(f"\n[2] Attempting to import recognition_dispatcher...")
try:
    from image_detection_package import recognition_dispatcher
    print(f"   SUCCESS: recognition_dispatcher imported")
except Exception as e:
    print(f"   FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Try to import LandmarkPredictor directly
print(f"\n[3] Attempting to import LandmarkPredictor...")
try:
    from image_detection_package.landmark_predictor import LandmarkPredictor
    print(f"   SUCCESS: LandmarkPredictor imported")
    
    # Try to create instance
    print(f"\n[4] Attempting to create LandmarkPredictor instance...")
    predictor = LandmarkPredictor()
    print(f"   SUCCESS: LandmarkPredictor instance created")
    print(f"   Model path: {predictor.model_path}")
    print(f"   Model loaded: {predictor.model is not None}")
except Exception as e:
    print(f"   FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Try torch directly
print(f"\n[5] Attempting to import torch directly...")
try:
    import torch
    print(f"   SUCCESS: torch imported")
    print(f"   Version: {torch.__version__}")
except Exception as e:
    print(f"   FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'=' * 70}")
print(f"Test complete")
print(f"{'=' * 70}")

