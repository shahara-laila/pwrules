# Phase 7 — Rule Filtering Notebook

**Requires:** the generated rules from Phase 6 (`pwrules-generated-rules` dataset).

**Accelerator: None (CPU).** Needs hashcat (syntax probe) but NOT the GPU/torch
stack — do not install `.[train]` here.

Attach datasets: `pwrules-generated-rules` (Phase 6), `pwrules-clean` (Phase 1).

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

### Cell 2 — core install + hashcat (no torch/unsloth)

```python
import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
subprocess.run(["apt-get", "-qq", "update"], check=False)
subprocess.run(["apt-get", "-qq", "install", "-y", "hashcat"], check=False)
print("core + hashcat install OK")
```

### Cell 3 — make package importable + list inputs

```python
import sys, os
REPO_DIR = "/kaggle/working/pwrules"
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
# Stage 1+2: syntax check and semantic dedup (no hashcat probe on val set).
import subprocess, sys
subprocess.run([sys.executable, "-m", "pwrules.filter", "--out", "/kaggle/working/filtered", "--hashcat-sample", "200", "--log-level", "INFO"],
               cwd="/kaggle/working/pwrules", check=True)
```

```python
# Stage 3: effectiveness ranking on validation set.
# (Only run if you have the validation corpus attached.)
import subprocess, sys
subprocess.run([sys.executable, "-m", "pwrules.filter", "--out", "/kaggle/working/filtered_ranked", "--effectiveness-ranking", "--top-k", "5000", "--log-level", "INFO"],
               cwd="/kaggle/working/pwrules", check=True)
```

```python
# Inspect funnel summary.
import pandas as pd
df = pd.read_csv("/kaggle/working/filtered/filter_funnel.csv")
print(df.to_string())
```

```python
from IPython.display import Image
Image("/kaggle/working/filtered/filter_funnel.png")
```

```python
# Count remaining rules.
with open("/kaggle/working/filtered/llm_untargeted_filtered.rule") as f:
    rules = [l.strip() for l in f if l.strip()]
print(f"Unique valid rules: {len(rules)}")
print("Sample rules:")
for r in rules[:15]:
    print(" ", r)
```

Save `/kaggle/working/filtered` as `yourname/pwrules-filtered`.
