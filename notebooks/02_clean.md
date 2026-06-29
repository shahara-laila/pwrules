# Phase 2 — Clean Corpus Notebook

**Accelerator: None (CPU).** This phase is pure Python — do NOT install the
`.[train]` GPU stack here; that heavy install is what crashes the kernel.

### Cell 1 — clone repo (safe to re-run; skips if already present)

```python
import os, subprocess
REPO_DIR = "/kaggle/working/pwrules"
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone",
                    "https://github.com/shahara-laila/pwrules.git", REPO_DIR], check=True)
os.chdir(REPO_DIR)
print("repo ready:", REPO_DIR)
```

### Cell 2 — core install only (fast, ~30s; no torch/unsloth, no hashcat)

```python
import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
print("core install OK")
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
import subprocess, sys
subprocess.run([sys.executable, "-m", "pwrules.clean", "--out", "/kaggle/working/clean", "--log-level", "INFO"],
               cwd="/kaggle/working/pwrules", check=True)
```

```python
# Verify splits are disjoint and test checksum is frozen.
from pwrules.clean import verify_test_checksum
assert verify_test_checksum("/kaggle/working/clean"), "Checksum mismatch!"
print("Test split checksum OK")

import json
manifest = json.load(open("/kaggle/working/clean/split_manifest.json"))
print(json.dumps(manifest, indent=2))
```

```python
# View stats figure inline.
from IPython.display import Image
Image("/kaggle/working/clean/stats/stats.png")
```

Save `/kaggle/working/clean` as a new Kaggle Dataset (`yourname/pwrules-clean`).
