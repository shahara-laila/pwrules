# Phase 4 — Target-Conditioning Dataset Notebook

**Accelerator: None (CPU).** Pure Python — do not install `.[train]` or hashcat here.

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
import subprocess, sys
subprocess.run([sys.executable, "-m", "pwrules.conditioning", "--mode", "synthetic", "--n-users", "500", "--out", "/kaggle/working/targeted", "--log-level", "INFO"],
               cwd="/kaggle/working/pwrules", check=True)
```

```python
import json
manifest = json.load(open("/kaggle/working/targeted/split_manifest.json"))
print(json.dumps(manifest, indent=2))

# Confirm held-out users are disjoint from training.
train_users = set()
with open("/kaggle/working/targeted/targeted_dataset.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        # user_id embedded in input — extract from the raw triple records instead.

# Direct disjoint check via the manifest.
assert manifest["disjoint"], "User split is NOT disjoint!"
print("Disjoint check PASSED")
```

```python
# Preview a few targeted instruction examples.
with open("/kaggle/working/targeted/targeted_dataset.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 3: break
        rec = json.loads(line)
        print(f"INPUT : {rec['input']}")
        print(f"OUTPUT: {rec['output']}")
        print()
```

Save `/kaggle/working/targeted` as `yourname/pwrules-targeted`.
