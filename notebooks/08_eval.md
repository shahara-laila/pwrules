# Phase 8 — Evaluation + Baselines Notebook

**Requires: GPU optional (hashcat runs on CPU), filtered rules from Phase 7.**

Attach datasets: `pwrules-filtered` (Phase 7), `pwrules-clean` (Phase 1),
`pwrules-targeted` (Phase 4).

```python
!git clone https://github.com/shahara-laila/pwrules.git
%cd pwrules
!pip install -q -e .
!pip install -q scikit-learn  # for RuleForge DBSCAN/HAC fallback
import os; print("inputs:", os.listdir("/kaggle/input"))
```

```python
# Verify hashcat is present.
!hashcat --version
!ls /usr/share/hashcat/rules/best64.rule 2>/dev/null || echo "best64 not in default location"
```

```python
# Run full evaluation — LLM rules + baselines + complementarity.
!python -m pwrules.eval run \
    --wordlist /kaggle/input/pwrules-clean/clean/train.txt \
    --test     /kaggle/input/pwrules-clean/clean/test.txt  \
    --out      /kaggle/working/results                     \
    --llm-untargeted /kaggle/input/pwrules-filtered/llm_untargeted_filtered.rule \
    --llm-filtered   /kaggle/input/pwrules-filtered/llm_untargeted_filtered.rule \
    --targeted-rules-dir /kaggle/input/pwrules-filtered    \
    --target-users   /kaggle/input/pwrules-targeted/target_users_test.jsonl \
    --run-ruleforge  \
    --dataset-name   rockyou                               \
    --log-level INFO
```

```python
# Inspect main results.
import pandas as pd
df = pd.read_csv("/kaggle/working/results/results.csv")
print(df.pivot(index="method", columns="k", values="hit_rate").to_string())
```

```python
from IPython.display import Image
Image("/kaggle/working/results/guessing_curve.png")
```

```python
# Cross-dataset evaluation (if a second test corpus is attached).
# Uncomment and adjust paths if running cross-dataset.
# !python -m pwrules.eval run \
#     --wordlist   /kaggle/input/pwrules-clean/clean/train.txt \
#     --test       /kaggle/input/pwrules-xdataset/test.txt    \
#     --out        /kaggle/working/results_xdataset           \
#     --llm-untargeted /kaggle/input/pwrules-filtered/llm_untargeted_filtered.rule \
#     --dataset-name   linkedin                               \
#     --log-level INFO
```

```python
# Inspect targeted results.
tgt = pd.read_csv("/kaggle/working/results/targeted_results.csv")
agg = tgt.groupby("k")["hit"].mean().reset_index()
agg.columns = ["k", "targeted_hit_rate"]
print(agg.to_string())
```

Save `/kaggle/working/results` as `yourname/pwrules-results`.
