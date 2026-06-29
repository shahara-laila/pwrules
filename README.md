# pwrules

LLM-generated, Hashcat-compatible password mangling rules — optionally conditioned on
target-user attributes — fine-tuned with QLoRA and evaluated by Hit@k against best64,
RuleForge, and MDBSCAN.

> **Code only.** This repository contains no passwords or leaked data. Corpora live in
> private Kaggle Datasets and are read at runtime from `/kaggle/input`. See
> [PLAYBOOK.md](PLAYBOOK.md) for the full workflow and [CLAUDE.md](CLAUDE.md) for project
> rules.

## Install

```bash
pip install -e .            # core (config, data, eval, tests)
pip install -e ".[train]"   # + Unsloth/TRL/PEFT/bitsandbytes (GPU, run on Kaggle)
```

## Layout

```
pwrules/{clean,ruleextract,conditioning,train,generate,filter,eval}/  # pipeline modules
pwrules/paths.py   slug-agnostic path discovery (single override point)
configs/   protocol.yaml (FROZEN) + train.yaml
tests/     pytest unit tests
notebooks/ thin Kaggle notebooks
PROGRESS.md  phase tracker (source of truth)
```

Each module is runnable as `python -m pwrules.<module>`. Track progress in
[PROGRESS.md](PROGRESS.md).

### Paths are auto-discovered

Kaggle mounts datasets under account-specific slugs (e.g.
`/kaggle/input/datasets/<user>/rockyou/rockyou.txt`), so paths are **not**
hard-coded. Every CLI finds its inputs by filename via `pwrules/paths.py`,
searching `/kaggle/working` then `/kaggle/input`. Just run, e.g.:

```bash
python -m pwrules.clean --out /kaggle/working/clean   # finds rockyou.txt itself
```

Override any path from one place — environment variables (set once at the top of
a notebook):

```python
import os
os.environ["PWRULES_ROCKYOU"] = "/kaggle/input/datasets/me/rockyou/rockyou.txt"
os.environ["PWRULES_INPUT"]   = "/kaggle/input"   # change the search root(s)
```

Call `pwrules.paths.show()` in a notebook to print everything it resolved.
