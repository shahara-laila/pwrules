# Phase 1 — Kaggle Smoke Test Notebook

Copy each cell block below into a **Kaggle Notebook** (Python, GPU enabled).
Attach your private RockYou dataset before running.

---

## Cell 1 — Clone repo and install

```python
import subprocess, sys

GH_USER = "YOUR_GH_USER"   # <-- replace
REPO = f"https://github.com/{GH_USER}/pwrules.git"

subprocess.run(["git", "clone", REPO], check=True)
%cd pwrules
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[train]"], check=True)
print("pwrules installed OK")
```

---

## Cell 2 — Install Hashcat and smoke-test --stdout mode

```python
import subprocess, tempfile, os

subprocess.run(["apt-get", "-qq", "update"], check=True)
subprocess.run(["apt-get", "-qq", "install", "-y", "hashcat"], check=True)

# Print device info (CPU backend is all we need for --stdout)
result = subprocess.run(["hashcat", "-I"], capture_output=True, text=True)
print(result.stdout or result.stderr)

# Verify --stdout works with a trivial no-op rule
with tempfile.NamedTemporaryFile(mode="w", suffix=".rule", delete=False) as rf:
    rf.write(":\n")          # no-op rule
    rf_path = rf.name

with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as wf:
    wf.write("password\n")   # probe word
    wf_path = wf.name

res = subprocess.run(
    ["hashcat", "--stdout", "-r", rf_path, wf_path, "--quiet"],
    capture_output=True, text=True,
)
assert res.returncode == 0 and "password" in res.stdout, \
    f"hashcat --stdout failed:\n{res.stderr}"
print("hashcat --stdout OK:", res.stdout.strip())

os.unlink(rf_path); os.unlink(wf_path)
```

---

## Cell 3 — GPU sanity + 1-step QLoRA probe

```python
import subprocess
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    capture_output=True, text=True,
)
print("GPU:", result.stdout.strip())

from pwrules.env import check_env
status = check_env(
    require_gpu=True,
    require_hashcat=True,
    probe_model=True,       # 1-step dummy forward to catch OOM early
)

assert status["cuda_available"], "No GPU!"
assert status["hashcat_available"], "Hashcat missing!"
assert status["vram_probe_ok"], "VRAM probe failed — model may OOM during training!"
print("All GPU checks passed.")
```

---

## Cell 4 — Confirm private dataset is attached and readable

```python
import os
from pathlib import Path

# List all attached inputs
inputs = list(Path("/kaggle/input").iterdir())
print("Attached datasets:", [d.name for d in inputs])

# Print first 5 lines of the corpus  (adjust slug as needed)
corpus_candidates = list(Path("/kaggle/input").rglob("rockyou.txt"))
assert corpus_candidates, (
    "rockyou.txt not found under /kaggle/input. "
    "Did you attach the private dataset?"
)
corpus_path = corpus_candidates[0]
print(f"\nCorpus: {corpus_path}")
with open(corpus_path, "rb") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        print(line.decode("utf-8", errors="replace").rstrip())
```

---

## Cell 5 — Full env report

```python
from pwrules.env import check_env
status = check_env(
    require_gpu=True,
    require_hashcat=True,
    kaggle_datasets=["rockyou"],   # adjust to your dataset slug
)
for k, v in status.items():
    print(f"  {k}: {v}")
```

---

**Expected output:** all assertions pass, GPU shown, hashcat prints candidates, corpus first lines visible.
Save this notebook run, then proceed to Phase 2.
