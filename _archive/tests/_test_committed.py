"""Test: Run the COMMITTED version of core (no seed) vs core_rebuild (has random_state)."""
import sys
import os
import subprocess
from pathlib import Path

project_root = Path('.').resolve()
sys.path.insert(0, str(project_root))

# Extract committed core/predictor.py to a temp file
committed_predictor = subprocess.check_output(
    ['git', 'show', 'd0a60be:core/predictor.py']
).decode('utf-8', errors='replace')

# Write to temp module
import tempfile
temp_dir = Path(tempfile.mkdtemp())
committed_path = temp_dir / 'committed_core.py'
# Fix imports for standalone execution
committed_predictor_fixed = committed_predictor.replace('from .utils', 'from core.utils')
committed_predictor_fixed = committed_predictor_fixed.replace('from .gpx_filter', 'from core.gpx_filter')
committed_predictor_fixed = committed_predictor_fixed.replace('from .types', 'from core.types')
with open(committed_path, 'w', encoding='utf-8') as f:
    f.write(committed_predictor_fixed)

# Add temp_dir to path before core.utils
sys.path.insert(0, str(temp_dir))

# Now test
records_dir = project_root / 'temp' / 'records'
gpx_file = project_root / 'temp' / 'routes' / '2025黄岩九峰大师赛最终版.gpx'

fit_files = sorted(records_dir.glob('*.fit'), key=lambda p: p.stat().st_size, reverse=True)
train_files = [str(f) for f in fit_files]

print(f"Using {len(train_files)} FIT files", flush=True)
print(f"GPX: {gpx_file.name}", flush=True)

# Import committed version
sys.path.insert(0, str(temp_dir))
import importlib
import core.utils
importlib.reload(core.utils)
from core import utils as committed_utils

# Force reload
if 'core.predictor' in sys.modules:
    del sys.modules['core.predictor']
if 'core' in sys.modules:
    del sys.modules['core']

# Read and exec committed code
exec_globals = {'__name__': 'committed_core', '__file__': str(committed_path)}
exec(compile(committed_predictor_fixed, str(committed_path), 'exec'), exec_globals)

MLRacePredictorCommitted = exec_globals['MLRacePredictor']

print("\n=== COMMITTED core (NO seed) ===", flush=True)
pred_committed = MLRacePredictorCommitted()
pred_committed.train_from_files(train_files)
r_committed = pred_committed.predict_race(str(gpx_file), effort_factor=1.0)
print(f"  predicted_time_hm: {r_committed['predicted_time_hm']}", flush=True)
print(f"  predicted_time_min: {r_committed['predicted_time_min']}", flush=True)

print("\n=== core_rebuild (HAS random_state=42) ===", flush=True)
from core_rebuild.predictor import MLRacePredictor as NEW
pred_new = NEW()
pred_new.train_from_files(train_files)
r_new = pred_new.predict_race(str(gpx_file), effort_factor=1.0)
print(f"  predicted_time_hm: {r_new['predicted_time_hm']}", flush=True)
print(f"  predicted_time_min: {r_new['predicted_time_min']}", flush=True)

print("\n=== HISTORICAL ===", flush=True)
print(f"  core (historical from file): 6:44:54 (405 min)", flush=True)
print(f"  core_rebuild (from file):    6:34:32 (395 min)", flush=True)

print("\n=== ROOT CAUSE ===", flush=True)
print(f"  COMMITTED core result: {r_committed['predicted_time_hm']} ({r_committed['predicted_time_min']} min)", flush=True)
print(f"  core_rebuild result:   {r_new['predicted_time_hm']} ({r_new['predicted_time_min']} min)", flush=True)

# Cleanup
import shutil
shutil.rmtree(temp_dir, ignore_errors=True)
