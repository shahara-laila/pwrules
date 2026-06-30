# Phase 5 — QLoRA Fine-Tuning Notebook

**Requires: Accelerator = GPU T4 x2 (or T4).** Set it in the right sidebar
BEFORE running, or the install/training will fail. No hashcat needed here.

> ⚠️ **Do NOT use P100.** The training stack (Unsloth/bitsandbytes) needs compute
> capability ≥ 7.0; the P100 is Pascal (6.0) and fails with
> `CUDA error: no kernel image is available for execution on the device`.
> T4 (compute 7.5) is supported and on the free tier.

Attach datasets: `pwrules-rules` (Phase 3) and optionally `pwrules-targeted` (Phase 4).

> 💡 **Free-tier note.** The Phase-3 instruction set can be millions of rows,
> which OOM-kills the tokenizer and won't finish in 12h. `configs/train.yaml`
> caps training to a deterministic seeded subsample (`max_train_samples`, default
> 150k) and limits tokenizer workers (`dataset_num_proc: 2`). Raise/`null` these
> on a bigger machine.

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

### Cell 2 — install GPU training stack (`.[train]` = torch/unsloth/peft/trl/bitsandbytes)

This is several GB and takes ~5–15 min on the first run of a fresh session; the
RAPIDS/cudf dependency-conflict warnings Kaggle prints are EXPECTED and harmless.
`unsloth` is already included in `.[train]`, so no separate install is needed.

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
# Untargeted training (Phase 3 instruction dataset).
# Unsloth is single-GPU: pin to ONE GPU even on a "T4 x2" session.
import os, subprocess, sys
env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
r = subprocess.run(
    [sys.executable, "-m", "pwrules.train",
     "--out", "/kaggle/working/adapter",
     "--config", "configs/train.yaml", "--log-level", "INFO"],
    cwd="/kaggle/working/pwrules", env=env,
)
if r.returncode != 0:
    raise SystemExit(f"training failed (exit {r.returncode}) — read the error above")
```

```python
# OR: Targeted training (Phase 4 instruction dataset).
# import os, subprocess, sys
# env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
# r = subprocess.run(
#     [sys.executable, "-m", "pwrules.train",
#      "--out", "/kaggle/working/adapter_targeted",
#      "--targeted", "--log-level", "INFO"],
#     cwd="/kaggle/working/pwrules", env=env,
# )
# if r.returncode != 0:
#     raise SystemExit(f"training failed (exit {r.returncode}) — read the error above")
```

```python
# View training curves.
from IPython.display import Image
Image("/kaggle/working/adapter/training_curves.png")
```

```python
# Memorisation report.
import json
report = json.load(open("/kaggle/working/adapter/memorisation_report.json"))
print(json.dumps(report, indent=2))
print(f"\nNovel-rule fraction: {report['novel_fraction']:.1%}  (want > 50%)")
```

Save `/kaggle/working/adapter` as `yourname/pwrules-adapter`.
If the session hits the 12-hour cap before training completes, re-run with:

```python
import os, subprocess, sys
env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
r = subprocess.run(
    [sys.executable, "-m", "pwrules.train",
     "--out", "/kaggle/working/adapter",
     "--resume", "/kaggle/input/pwrules-adapter/checkpoints"],
    cwd="/kaggle/working/pwrules", env=env,
)
if r.returncode != 0:
    raise SystemExit(f"training failed (exit {r.returncode}) — read the error above")
```
