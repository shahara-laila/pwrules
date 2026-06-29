# Phase 3 — Rule Extraction Notebook

**Accelerator: None (CPU).** Needs hashcat (for the `--parity` check) but NOT the
GPU/torch stack — do not install `.[train]` here.

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
import subprocess, sys
subprocess.run([sys.executable, "-m", "pwrules.ruleextract", "--out", "/kaggle/working/rules", "--parity", "200", "--log-level", "INFO"],
               cwd="/kaggle/working/pwrules", check=True)
```

```python
import json
report = json.load(open("/kaggle/working/rules/coverage_report.json"))
print("Coverage report:")
print(json.dumps(report, indent=2))
```

```python
# Spot-check 30 random triples.
import json, random
from pwrules.ruleextract.applier import apply_rule

triples = []
with open("/kaggle/working/rules/rules_dataset.jsonl") as f:
    for line in f:
        triples.append(json.loads(line))

sample = random.sample(triples, min(30, len(triples)))
for t in sample:
    result = apply_rule(t["base"], t["rule"])
    status = "OK" if result == t["password"] else f"FAIL ({result!r})"
    print(f"  {t['base']!r:15s} + {t['rule']!r:25s} → {t['password']!r:20s}  [{status}]")
```

```python
from IPython.display import Image
Image("/kaggle/working/rules/rule_frequency.png")
```

Save `/kaggle/working/rules` as `yourname/pwrules-rules`.
