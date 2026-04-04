#!/usr/bin/env python
"""Test script for core_rebuild"""

import sys
sys.path.insert(0, '.')

print("Checking dependencies...")
try:
    from fitparse import FitFile
    print("  fitparse: OK")
except Exception as e:
    print(f"  fitparse: {e}")

try:
    import lightgbm as lgb
    print("  lightgbm: OK")
except Exception as e:
    print(f"  lightgbm: {e}")

try:
    import numpy as np
    print("  numpy: OK")
except Exception as e:
    print(f"  numpy: {e}")

from pathlib import Path

print("\nChecking files...")
fit_path = Path("temp/records/玉环100KM_431586142.fit")
print(f"  FIT exists: {fit_path.exists()}, size: {fit_path.stat().st_size if fit_path.exists() else 0}")

gpx_path = Path("example/2025黄岩九峰大师赛最终版.gpx")
print(f"  GPX exists: {gpx_path.exists()}, size: {gpx_path.stat().st_size if gpx_path.exists() else 0}")

print("\nLoading MLRacePredictor...")
from core_rebuild.predictor import MLRacePredictor
print("  MLRacePredictor: OK")

print("\nTest completed successfully!")