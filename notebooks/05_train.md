# Phase 5 — QLoRA Fine-Tuning Notebook

**Requires: GPU (T4 or better), Unsloth.**

Attach datasets: `pwrules-rules` (Phase 3) and optionally `pwrules-targeted` (Phase 4).

```python
!git clone https://github.com/shahara-laila/pwrules.git
%cd pwrules
!pip install -q -e ".[train]"
!pip install -q unsloth
import os; print("inputs:", os.listdir("/kaggle/input"))
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

```python
# Untargeted training (Phase 3 instruction dataset).
!python -m pwrules.train \
    --data   /kaggle/input/pwrules-rules \
    --out    /kaggle/working/adapter \
    --config configs/train.yaml \
    --log-level INFO
```

```python
# OR: Targeted training (Phase 4 instruction dataset).
# !python -m pwrules.train \
#     --data     /kaggle/input/pwrules-targeted \
#     --out      /kaggle/working/adapter_targeted \
#     --targeted \
#     --log-level INFO
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
!python -m pwrules.train \
    --data   /kaggle/input/pwrules-rules \
    --out    /kaggle/working/adapter \
    --resume /kaggle/input/pwrules-adapter/checkpoints
```
