# Phase 3 — Rule Extraction Notebook

```python
!git clone https://github.com/shahara-laila/pwrules.git
%cd pwrules
!pip install -q -e .
!apt-get -qq update && apt-get -qq install -y hashcat
import os; print("inputs:", os.listdir("/kaggle/input"))
```

```python
!python -m pwrules.ruleextract \
    --clean   /kaggle/input/pwrules-clean/clean \
    --out     /kaggle/working/rules \
    --parity  200 \
    --log-level INFO
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
