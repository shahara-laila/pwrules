# Phase 7 — Rule Filtering Notebook

**Requires:** the generated rules from Phase 6 (`pwrules-generated-rules` dataset).

Attach datasets: `pwrules-generated-rules` (Phase 6), `pwrules-clean` (Phase 1).

```python
!git clone https://github.com/shahara-laila/pwrules.git
%cd pwrules
!pip install -q -e .
import os; print("inputs:", os.listdir("/kaggle/input"))
```

```python
# Stage 1+2: syntax check and semantic dedup (no hashcat probe on val set).
!python -m pwrules.filter \
    --rules    /kaggle/input/pwrules-generated-rules/rules/llm_untargeted.rule \
               /kaggle/input/pwrules-generated-rules/rules/llm_unfiltered.rule \
    --out      /kaggle/working/filtered \
    --hashcat-sample 200 \
    --log-level INFO
```

```python
# Stage 3: effectiveness ranking on validation set.
# (Only run if you have the validation corpus attached.)
!python -m pwrules.filter \
    --rules    /kaggle/input/pwrules-generated-rules/rules/llm_untargeted.rule \
    --out      /kaggle/working/filtered_ranked \
    --val      /kaggle/input/pwrules-clean/clean/val.txt \
    --wordlist /kaggle/input/pwrules-clean/clean/train.txt \
    --effectiveness-ranking \
    --top-k    5000 \
    --log-level INFO
```

```python
# Inspect funnel summary.
import pandas as pd
df = pd.read_csv("/kaggle/working/filtered/filter_funnel.csv")
print(df.to_string())
```

```python
from IPython.display import Image
Image("/kaggle/working/filtered/filter_funnel.png")
```

```python
# Count remaining rules.
with open("/kaggle/working/filtered/llm_untargeted_filtered.rule") as f:
    rules = [l.strip() for l in f if l.strip()]
print(f"Unique valid rules: {len(rules)}")
print("Sample rules:")
for r in rules[:15]:
    print(" ", r)
```

Save `/kaggle/working/filtered` as `yourname/pwrules-filtered`.
