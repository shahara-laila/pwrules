# Phase 10 — Results Export Notebook

**Produces all paper-ready tables and figures.**

Attach datasets: `pwrules-results` (Phase 8), `pwrules-ablations` (Phase 9),
`pwrules-filtered` (Phase 7, for filter funnel).

```python
!git clone https://github.com/shahara-laila/pwrules.git
%cd pwrules
!pip install -q -e .
import os; print("inputs:", os.listdir("/kaggle/input"))
```

```python
# Export all paper artifacts.
!python -m pwrules.eval report \
    --results-dir   /kaggle/input/pwrules-results                              \
    --ablations-dir /kaggle/input/pwrules-ablations                            \
    --filter-dir    /kaggle/input/pwrules-filtered                             \
    --out           /kaggle/working/paper                                       \
    --k-values      10,100,1000,10000,100000,1000000,10000000                  \
    --dataset       rockyou
```

```python
# Check for missing source files.
missing = open("/kaggle/working/paper/MISSING.txt").read()
print(missing)
```

```python
# Main Hit@k comparison table.
import pandas as pd
df = pd.read_csv("/kaggle/working/paper/table_hit_at_k.csv", index_col=0)
print(df.to_string())
```

```python
# LaTeX source for Table 3 (main comparison).
print(open("/kaggle/working/paper/table_hit_at_k.tex").read())
```

```python
# Ablation table.
abl = pd.read_csv("/kaggle/working/paper/table_ablations.csv", index_col=0)
print(abl.to_string())
```

```python
# LaTeX source for ablation table.
print(open("/kaggle/working/paper/table_ablations.tex").read())
```

```python
# Guessing-number curve.
from IPython.display import Image
Image("/kaggle/working/paper/guessing_curve.png")
```

```python
# Targeted table.
tgt = pd.read_csv("/kaggle/working/paper/table_targeted.csv", index_col=0)
print(tgt.to_string())
print(open("/kaggle/working/paper/table_targeted.tex").read())
```

```python
# Filter funnel table.
funnel = pd.read_csv("/kaggle/working/paper/table_filter_funnel.csv", index_col=0)
print(funnel.to_string())
print(open("/kaggle/working/paper/table_filter_funnel.tex").read())
```

```python
# List all generated artifacts.
import os
for f in sorted(os.listdir("/kaggle/working/paper")):
    size = os.path.getsize(f"/kaggle/working/paper/{f}")
    print(f"  {f:45s}  {size:>8,} bytes")
```

Save `/kaggle/working/paper` as `yourname/pwrules-paper`.
