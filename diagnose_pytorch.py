#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Detailed PyTorch DLL dependency diagnosis
"""

import sys
import os
import ctypes

print("=" * 70)
print("PyTorch DLL Dependency Diagnosis")
print("=" * 70)

# 1. Check Python environment
print(f"\n[1] Python Environment:")
print(f"   Python version: {sys.version}")
print(f"   Python bits: {ctypes.sizeof(ctypes.c_voidp) * 8}-bit")
print(f"   Executable: {sys.executable}")

# 2. Check system PATH
print(f"\n[2] System PATH (first 5):")
paths = os.environ.get('PATH', '').split(os.pathsep)
for i, p in enumerate(paths[:5], 1):
    print(f"   {i}. {p}")

# 3. Check critical system DLLs
print(f"\n[3] Check critical system DLLs:")
system_dlls = [
    'vcruntime140.dll',      # VC++ 2015-2022 runtime
    'vcruntime140_1.dll',    # VC++ 2019+ runtime
    'msvcp140.dll',          # C++ standard library
    'concrt140.dll',         # Concurrency runtime
]

system32 = os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'System32')
print(f"   System directory: {system32}")

missing_dlls = []
for dll in system_dlls:
    dll_path = os.path.join(system32, dll)
    exists = os.path.exists(dll_path)
    status = "OK" if exists else "MISSING"
    print(f"   [{status}] {dll}")
    if not exists:
        missing_dlls.append(dll)

# 4. Try to import torch
print(f"\n[4] Try importing PyTorch:")
torch_imported = False
try:
    import torch
    torch_imported = True
    print(f"   SUCCESS: PyTorch imported")
    print(f"   Version: {torch.__version__}")
    print(f"   Install path: {torch.__path__[0]}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"   FAILED: Cannot import PyTorch")
    print(f"   Error type: {type(e).__name__}")
    print(f"   Error message: {e}")
    
    error_msg = str(e)
    if 'shm.dll' in error_msg:
        print(f"\n   Problem: shm.dll dependencies missing")
        
        # Check torch lib directory
        try:
            import site
            torch_lib = None
            for sp in site.getsitepackages():
                potential_path = os.path.join(sp, 'torch', 'lib')
                if os.path.exists(potential_path):
                    torch_lib = potential_path
                    break
            
            if torch_lib:
                print(f"   Torch lib directory: {torch_lib}")
                print(f"   DLL files in directory:")
                for f in os.listdir(torch_lib):
                    if f.endswith('.dll'):
                        file_path = os.path.join(torch_lib, f)
                        size_mb = os.path.getsize(file_path) / (1024*1024)
                        print(f"      - {f} ({size_mb:.1f} MB)")
        except Exception as e2:
            print(f"   Cannot check torch lib directory: {e2}")

# 5. Check installed PyTorch package info
print(f"\n[5] PyTorch package info:")
try:
    import subprocess
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'show', 'torch'],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        for line in result.stdout.split('\n'):
            if line.strip() and any(key in line for key in ['Name:', 'Version:', 'Location:']):
                print(f"   {line}")
except Exception as e:
    print(f"   Cannot get package info: {e}")

# 6. Recommendations
print(f"\n{'=' * 70}")
print(f"RECOMMENDATIONS:")
print(f"{'=' * 70}")

if missing_dlls:
    print(f"\nMISSING DLLs: {', '.join(missing_dlls)}")
    print(f"""
Solution A: Install complete VC++ Redistributable:
  1. Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
  2. Run and install
  3. RESTART your computer
  4. Try again
""")
elif not torch_imported:
    print(f"""
DLLs seem OK, but PyTorch still fails. Try:

Solution B: Reinstall PyTorch (CPU version):
  Run in PowerShell:
  
  .\.venv\Scripts\Activate.ps1
  pip uninstall torch torchvision torchaudio -y
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  
This will install a CPU-only version with better Windows compatibility.
""")
else:
    print(f"\nSUCCESS! PyTorch is working correctly.")

print(f"\n{'=' * 70}")
print(f"Diagnosis complete")
print(f"{'=' * 70}")

