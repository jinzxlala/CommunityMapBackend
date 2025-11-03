#!/usr/bin/env python
"""
Django 环境诊断脚本
用于检查 Django 服务器是否在正确的虚拟环境中运行
"""

import sys
import os

print("=" * 60)
print("🔍 Django 环境诊断")
print("=" * 60)

# 1. 检查 Python 版本和路径
print(f"\n1️⃣ Python 信息:")
print(f"   版本: {sys.version}")
print(f"   可执行文件路径: {sys.executable}")
print(f"   虚拟环境: {sys.prefix}")

# 2. 检查是否在虚拟环境中
in_venv = hasattr(sys, 'real_prefix') or (
    hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
)
print(f"\n2️⃣ 虚拟环境状态:")
if in_venv:
    print(f"   ✅ 当前运行在虚拟环境中")
else:
    print(f"   ⚠️ 当前未在虚拟环境中运行")

# 3. 检查关键包是否安装
print(f"\n3️⃣ 关键依赖检查:")

packages_to_check = [
    ('django', 'Django'),
    ('torch', 'PyTorch'),
    ('torchvision', 'TorchVision'),
    ('PIL', 'Pillow'),
    ('rest_framework', 'Django REST Framework'),
]

for module_name, display_name in packages_to_check:
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"   ✅ {display_name}: {version}")
    except ImportError:
        print(f"   ❌ {display_name}: 未安装")

# 4. 检查 image_detection_package 模块
print(f"\n4️⃣ 检查 image_detection_package 模块:")
try:
    from image_detection_package.landmark_predictor import LandmarkPredictor
    print(f"   ✅ LandmarkPredictor 导入成功")
    
    # 尝试初始化
    try:
        predictor = LandmarkPredictor()
        print(f"   ✅ LandmarkPredictor 初始化成功")
        print(f"   模型路径: {predictor.model_path}")
        print(f"   模型文件存在: {os.path.exists(predictor.model_path)}")
    except Exception as e:
        print(f"   ⚠️ LandmarkPredictor 初始化失败: {e}")
        
except ImportError as e:
    print(f"   ❌ LandmarkPredictor 导入失败: {e}")

# 5. 检查模型文件
print(f"\n5️⃣ 检查模型文件:")
model_path = os.path.join('image_detection_package', 'best_landmark_model_finetuned.pt')
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   ✅ 模型文件存在: {model_path}")
    print(f"   大小: {size_mb:.2f} MB")
else:
    print(f"   ❌ 模型文件不存在: {model_path}")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)

# 6. 给出建议
print(f"\n💡 建议:")
if not in_venv:
    print("   ⚠️ 请确保在虚拟环境中启动 Django 服务器")
    print("   激活虚拟环境后再运行: python manage.py runserver")

