# Phase 2 — Clean Corpus Notebook

Standard starter cell then:

```python
# Standard starter (see PLAYBOOK.md Appendix)
!git clone https://github.com/shahara-laila/pwrules.git
%cd pwrules
!pip install -q -e .
!apt-get -qq update && apt-get -qq install -y hashcat

import os; print("inputs:", os.listdir("/kaggle/input"))
```

```python
!python -m pwrules.clean \
    --input /kaggle/input/rockyou/rockyou.txt \
    --out   /kaggle/working/clean \
    --log-level INFO
```

```python
# Verify splits are disjoint and test checksum is frozen.
from pwrules.clean import verify_test_checksum
assert verify_test_checksum("/kaggle/working/clean"), "Checksum mismatch!"
print("Test split checksum OK")

import json
manifest = json.load(open("/kaggle/working/clean/split_manifest.json"))
print(json.dumps(manifest, indent=2))
```

```python
# View stats figure inline.
from IPython.display import Image
Image("/kaggle/working/clean/stats/stats.png")
```

Save `/kaggle/working/clean` as a new Kaggle Dataset (`yourname/pwrules-clean`).
