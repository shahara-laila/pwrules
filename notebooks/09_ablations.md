# Phase 9 — Ablations + Statistics Notebook

**Requires:** multiple-seed result CSVs from Phase 8 runs.

Each seed is a separate Kaggle run.  Save each run's `results.csv` as
`results_seed1.csv`, `results_seed2.csv`, `results_seed3.csv` and attach
them all here.

Attach datasets: `pwrules-results-seed1`, `pwrules-results-seed2`,
`pwrules-results-seed3` (Phase 8, different seeds).

```python
!git clone https://github.com/shahara-laila/pwrules.git
%cd pwrules
!pip install -q -e .
!pip install -q scipy  # for McNemar's test
import os, shutil

# Collect all seed CSVs into a single directory.
os.makedirs("/kaggle/working/all_results", exist_ok=True)
for i in [1, 2, 3]:
    src = f"/kaggle/input/pwrules-results-seed{i}/results.csv"
    dst = f"/kaggle/working/all_results/results_seed{i}.csv"
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Copied seed {i}")
    else:
        print(f"MISSING: seed {i} results")
```

```python
# Run ablation analysis.
!python -m pwrules.eval ablate \
    --results-dir /kaggle/working/all_results \
    --out         /kaggle/working/ablations   \
    --baseline    best64                      \
    --k-pivot     1000000                     \
    --n-bootstrap 10000                       \
    --min-seeds   3
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
