# Phase 8 — Evaluation + Baselines Notebook

**Accelerator: None (CPU) — hashcat runs on CPU.** Needs hashcat but NOT the
GPU/torch stack — do not install `.[train]` here.

Attach datasets: `pwrules-filtered` (Phase 7), `pwrules-clean` (Phase 1),
`pwrules-targeted` (Phase 4).

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
subprocess.run(["apt-get", "-qq", "install", "-y", "--no-install-recommends", "hashcat"], check=False)
print("core + hashcat install OK")
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
# Verify hashcat is present.
!hashcat --version
!ls /usr/share/hashcat/rules/best64.rule 2>/dev/null || echo "best64 not in default location"
```

```python
# Run full evaluation — LLM rules + baselines + complementarity.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pwrules.eval", "run", "--out", "/kaggle/working/results", "--run-ruleforge", "--dataset-name", "rockyou", "--log-level", "INFO"],
               cwd="/kaggle/working/pwrules", check=True)
```

```python
# Inspect main results.
import pandas as pd
df = pd.read_csv("/kaggle/working/results/results.csv")
print(df.pivot(index="method", columns="k", values="hit_rate").to_string())
```

```python
from IPython.display import Image
Image("/kaggle/working/results/guessing_curve.png")
```

```python
# Cross-dataset evaluation (if a second test corpus is attached).
# Uncomment and adjust paths if running cross-dataset.
# import subprocess, sys
# subprocess.run([sys.executable, "-m", "pwrules.eval", "run", "--wordlist", "/kaggle/input/pwrules-clean/clean/train.txt", "--test", "/kaggle/input/pwrules-xdataset/test.txt", "--out", "/kaggle/working/results_xdataset", "--llm-untargeted", "/kaggle/input/pwrules-filtered/llm_untargeted_filtered.rule", "--dataset-name", "linkedin", "--log-level", "INFO"],
#                cwd="/kaggle/working/pwrules", check=True)
```

```python
# Inspect targeted results.
tgt = pd.read_csv("/kaggle/working/results/targeted_results.csv")
agg = tgt.groupby("k")["hit"].mean().reset_index()
agg.columns = ["k", "targeted_hit_rate"]
print(agg.to_string())
```

Save `/kaggle/working/results` as `yourname/pwrules-results`.
