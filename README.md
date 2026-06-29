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
configs/   protocol.yaml (FROZEN) + train.yaml
tests/     pytest unit tests
notebooks/ thin Kaggle notebooks
PROGRESS.md  phase tracker (source of truth)
```

Each module is runnable as `python -m pwrules.<module>`. Track progress in
[PROGRESS.md](PROGRESS.md).
