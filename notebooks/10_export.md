# Phase 10 — Results Export Notebook

**Produces all paper-ready tables and figures.**

**Accelerator: None (CPU).** Pure reporting — do not install `.[train]` or hashcat here.

Attach datasets: `pwrules-results` (Phase 8), `pwrules-ablations` (Phase 9),
`pwrules-filtered` (Phase 7, for filter funnel).

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
# Export all paper artifacts.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pwrules.eval", "report", "--out", "/kaggle/working/paper", "--k-values", "10,100,1000,10000,100000,1000000,10000000", "--dataset", "rockyou"],
               cwd="/kaggle/working/pwrules", check=True)
```

```python
# Check for missing source files.
missing = open("/kaggle/working/paper/MISSING.txt").read()
print(missing)
```

```python
# Main Hit@k comparison table.
import pandas as pd
df = pd.read_csv("/kaggle/working/paper/table_hit_at_k.csv", index_col=0)
print(df.to_string())
```

```python
# LaTeX source for Table 3 (main comparison).
print(open("/kaggle/working/paper/table_hit_at_k.tex").read())
```

```python
# Ablation table.
abl = pd.read_csv("/kaggle/working/paper/table_ablations.csv", index_col=0)
print(abl.to_string())
```

```python
# LaTeX source for ablation table.
print(open("/kaggle/working/paper/table_ablations.tex").read())
```

```python
# Guessing-number curve.
from IPython.display import Image
Image("/kaggle/working/paper/guessing_curve.png")
```

```python
# Targeted table.
tgt = pd.read_csv("/kaggle/working/paper/table_targeted.csv", index_col=0)
print(tgt.to_string())
print(open("/kaggle/working/paper/table_targeted.tex").read())
```

```python
# Filter funnel table.
funnel = pd.read_csv("/kaggle/working/paper/table_filter_funnel.csv", index_col=0)
print(funnel.to_string())
print(open("/kaggle/working/paper/table_filter_funnel.tex").read())
```

```python
# List all generated artifacts.
import os
for f in sorted(os.listdir("/kaggle/working/paper")):
    size = os.path.getsize(f"/kaggle/working/paper/{f}")
    print(f"  {f:45s}  {size:>8,} bytes")
```

Save `/kaggle/working/paper` as `yourname/pwrules-paper`.
