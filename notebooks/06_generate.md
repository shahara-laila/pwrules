# Phase 6 — Rule Generation Notebook

**Requires: GPU, the saved adapter from Phase 5.**

Attach datasets: `pwrules-adapter` (Phase 5), `pwrules-targeted` (Phase 4), `rockyou` (for probe words).

```python
!git clone https://github.com/<YOUR_GH_USER>/pwrules.git
%cd pwrules
!pip install -q -e ".[train]"
!pip install -q unsloth
import os; print("inputs:", os.listdir("/kaggle/input"))
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

```python
# Untargeted + targeted generation.
!python -m pwrules.generate \
    --adapter      /kaggle/input/pwrules-adapter/adapter \
    --out          /kaggle/working/rules \
    --target-users /kaggle/input/pwrules-targeted/target_users_test.jsonl \
    --budget       10000 \
    --log-level    INFO
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
