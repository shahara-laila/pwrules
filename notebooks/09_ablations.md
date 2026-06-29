# Phase 9 — Ablations + Statistics Notebook

**Requires:** multiple-seed result CSVs from Phase 8 runs.

Each seed is a separate Kaggle run.  Save each run's `results.csv` as
`results_seed1.csv`, `results_seed2.csv`, `results_seed3.csv` and attach
them all here.

**Accelerator: None (CPU).** Pure stats — do not install `.[train]` or hashcat here.

Attach datasets: `pwrules-results-seed1`, `pwrules-results-seed2`,
`pwrules-results-seed3` (Phase 8, different seeds).

### Cell 1 — clone repo

```python
import os, subprocess
REPO_DIR = "/kaggle/working/pwrules"
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone",
                    "https://github.com/shahara-laila/pwrules.git", REPO_DIR], check=True)
os.chdir(REPO_DIR)
print("repo ready:", REPO_DIR)
```

### Cell 2 — core install only (fast)

```python
import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
print("core install OK")
```

### Cell 3 — make package importable + list inputs

```python
import sys, os, subprocess
REPO_DIR = "/kaggle/working/pwrules"
if not os.path.isdir(REPO_DIR):  # self-heal: clone if Cell 1 was skipped
    subprocess.run(["git", "clone",
                    "https://github.com/shahara-laila/pwrules.git", REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
import pwrules
print("pwrules importable at:", pwrules.__file__)
print("inputs:", os.listdir("/kaggle/input"))

# Preview the paths auto-discovered for this phase (slug-agnostic).
from pwrules import paths
paths.show()
```

```python
# Run ablation analysis.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pwrules.eval", "ablate", "--results-dir", "/kaggle/working/all_results", "--out", "/kaggle/working/ablations", "--baseline", "best64", "--k-pivot", "1000000", "--n-bootstrap", "10000", "--min-seeds", "3"],
               cwd="/kaggle/working/pwrules", check=True)
```

```python
# Inspect aggregated results.
import pandas as pd
agg = pd.read_csv("/kaggle/working/ablations/aggregated_results.csv")
pivot = agg.pivot(index="method", columns="k", values="mean_hit_rate")
print(pivot.to_string())
```

```python
# Ablation table (pairwise comparisons).
abl = pd.read_csv("/kaggle/working/ablations/ablations.csv")
print(abl[["label", "mean_a", "std_a", "mean_b", "std_b", "delta"]].to_string())
```

```python
# Significance report.
import json
sig = json.load(open("/kaggle/working/ablations/significance_report.json"))
for row in sig[:10]:
    print(
        f"{row['method']} vs {row['baseline']} @ k={row['k']:>9,}: "
        f"Δ={row['observed_delta']:+.4f}  "
        f"95%CI=[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]  "
        f"McNemar p={row.get('mcnemar_p', 'N/A')}  "
        f"sig={row['significant_005']}"
    )
```

Save `/kaggle/working/ablations` as `yourname/pwrules-ablations`.
