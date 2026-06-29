# Project: pwrules — LLM-generated password mangling rules

## What this is
A research pipeline that fine-tunes a compact instruction-tuned LLM (QLoRA) to
generate Hashcat-compatible password mangling rules, optionally conditioned on
target-user attributes, and evaluates them by Hit@k against best64, RuleForge,
and MDBSCAN. Target output: a conference paper.

## Execution model
Code runs on Kaggle (single GPU, ~12h sessions). This repo is CODE ONLY and
public. Data lives in private Kaggle Datasets, read at runtime from /kaggle/input.
Kaggle notebooks clone this repo, `pip install -e .`, and call the package.

## Stack
- Python 3.10+, packaged as `pwrules` with a CLI (`python -m pwrules.<module>`).
- Fine-tuning: Unsloth + TRL (SFTTrainer) + PEFT + bitsandbytes, 4-bit QLoRA.
- Base model: configurable, compact instruction-tuned LLM (default a small Qwen3
  variant). MODEL-AGNOSTIC — never hard-code one model; read it from config.
- Cracking/eval: Hashcat in `--stdout` (candidate-generation) mode. No hash cracking.
- Config-driven: all paths, hyperparameters, budgets in YAML under configs/.

## Repo layout
pwrules/{clean,ruleextract,conditioning,train,generate,filter,eval}/  # modules
configs/   # protocol.yaml (frozen), train.yaml
tests/     # pytest unit tests
notebooks/ # thin Kaggle notebooks (markdown describing cells is fine)
PROGRESS.md  # phase tracker, updated after every phase
.gitignore   # MUST exclude data/, *.txt corpora, model weights, /kaggle outputs

## Conventions
- Small, composable functions; type hints; docstrings; no notebooks doing logic.
- Every module gets a CLI entrypoint and at least one pytest test.
- Deterministic: read SEED from configs and set it everywhere.
- After finishing a phase: update PROGRESS.md (status + the exact output files),
  run the tests, and stop for my review. Do not start the next phase unsolicited.

## Non-negotiables
- NEVER commit raw passwords or any leaked data. NEVER fabricate experimental
  results or hard-code metric values. NEVER weaken the evaluation to look better.
- The four protocol decisions in configs/protocol.yaml (base wordlist, guess
  budget, split protocol, Hit@k definition) are FROZEN; do not change them.
