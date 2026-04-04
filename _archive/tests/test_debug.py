#!/usr/bin/env python
"""Step-by-step debug for core_rebuild"""

import sys
sys.path.insert(0, '.')

print("Step 1: Imports...", flush=True)
from core_rebuild.predictor import MLRacePredictor
print("Step 1: Done", flush=True)

from pathlib import Path
records_dir = Path("temp/records")
example_dir = Path("example")

print("Step 2: File check...", flush=True)
fit_files = [
    records_dir / "玉环100KM_431586142.fit",
]
existing = [f for f in fit_files if f.exists()]
print(f"Found {len(existing)} files", flush=True)

if not existing:
    print("ERROR: No files")
    sys.exit(1)

print("Step 3: Create predictor...", flush=True)
predictor = MLRacePredictor()
print("Step 3: Done", flush=True)

print("Step 4: Train...", flush=True)
result = predictor.train_from_files(existing)
print(f"Step 4: Result = {result}", flush=True)

if not result:
    print("ERROR: Training failed")
    sys.exit(1)

print("\nTraining stats:", predictor.training_stats)
print("\nTest PASSED!")