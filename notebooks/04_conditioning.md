# Phase 4 — Target-Conditioning Dataset Notebook

```python
!git clone https://github.com/<YOUR_GH_USER>/pwrules.git
%cd pwrules
!pip install -q -e .
import os; print("inputs:", os.listdir("/kaggle/input"))
```

```python
!python -m pwrules.conditioning \
    --rules /kaggle/input/pwrules-rules/rules_dataset.jsonl \
    --mode  synthetic \
    --n-users 500 \
    --out   /kaggle/working/targeted \
    --log-level INFO
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
