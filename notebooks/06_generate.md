# Phase 6 — Rule Generation Notebook

**Requires: Accelerator = GPU**, plus the saved adapter from Phase 5. No hashcat
needed here (generation only emits rule strings; hashcat is used later in eval).

Attach datasets: `pwrules-adapter` (Phase 5), `pwrules-targeted` (Phase 4), `rockyou` (for probe words).

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

### Cell 2 — install GPU stack (`.[train]`; several GB, ~5–15 min first run)

`unsloth` is already included in `.[train]`. RAPIDS/cudf conflict warnings are harmless.

```python
import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[train]"], check=True)
print("train stack install OK")
```

### Cell 3 — make package importable + GPU sanity

```python
import sys, os, shutil, subprocess
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

if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                          "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip())
else:
    print("No GPU attached (Accelerator = None). Enable GPU for this phase.")
```

```python
# Untargeted + targeted generation.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pwrules.generate", "--out", "/kaggle/working/rules", "--budget", "10000", "--log-level", "INFO"],
               cwd="/kaggle/working/pwrules", check=True)
```

```python
# Inspect the generated rule file.
with open("/kaggle/working/rules/rules/llm_untargeted.rule") as f:
    rules = [l.strip() for l in f if l.strip()]

print(f"Total rules: {len(rules)}")
print("First 20 rules:")
for r in rules[:20]:
    print(" ", r)
```

```python
# Diversity stats.
import json
stats = json.load(open("/kaggle/working/rules/generation_stats.json"))
print(json.dumps(stats["combined"], indent=2))
```

```python
from IPython.display import Image
Image("/kaggle/working/rules/rule_length_dist.png")
```

Save `/kaggle/working/rules` as `yourname/pwrules-generated-rules`.
