#!/usr/bin/env python3
"""
Quick retraining with SMOTE disabled for BERT MLP - saves results to file.
"""
import sys
from pathlib import Path
import subprocess
import json

repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

# Simply run main.py and capture exit code
result = subprocess.run(
    [sys.executable, str(repo_root / "main.py"), "--skip-eda"],
    cwd=repo_root,
    capture_output=False  # Let output go to console
)

print(f"\n\n=== MAIN.PY COMPLETED WITH EXIT CODE: {result.returncode} ===")

# Now read results
try:
    import pandas as pd
    results = pd.read_csv(repo_root / "outputs" / "model_comparison.csv")
    print("\n=== MODEL RESULTS ===")
    print(results.to_string())
    
    mlp_acc = results[results['model'] == 'MLP (BERT)']['accuracy'].values[0]
    print(f"\n✓ MLP (BERT) accuracy: {mlp_acc:.4f}")
    
    if mlp_acc > 0.40:
        print(f"✓ SUCCESS: MLP improved from 0.214 → {mlp_acc:.4f}")
    else:
        print(f"✗ REGRESS: MLP still low at {mlp_acc:.4f}")
except Exception as e:
    print(f"Could not read results: {e}")

sys.exit(result.returncode)
