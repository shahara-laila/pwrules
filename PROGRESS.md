# PROGRESS

Single source of truth for what's done and where each output lives.
Update after every phase: status, exact output files, and Kaggle Dataset slug.

## Phase tracker

| ✅ | Phase | Builds | Outcome (deliverable) |
|----|-------|--------|-----------------------|
| ☑ | 0 | Repo scaffold + CLAUDE.md + configs + CI | Clean repo, frozen `protocol.yaml`, `PROGRESS.md` |
| ☑ | 1 | Kaggle env + smoke notebook | `pwrules/env.py`, `notebooks/01_smoke.md`, `tests/test_env.py` |
| ☑ | 2 | Cleaning/preprocessing + splits | `pwrules/clean/`, `notebooks/02_clean.md`, `tests/test_clean.py` |
| ☑ | 3 | **Rule extraction** | `pwrules/ruleextract/` (applier + extractor), `notebooks/03_ruleextract.md`, `tests/test_ruleextract.py` |
| ☑ | 4 | Target-conditioning dataset | `pwrules/conditioning/`, `notebooks/04_conditioning.md`, `tests/test_conditioning.py` |
| ☑ | 5 | QLoRA fine-tuning | `pwrules/train/`, `notebooks/05_train.md`, `tests/test_train.py` |
| ☑ | 6 | Rule generation (inference) | `pwrules/generate/`, `notebooks/06_generate.md`, `tests/test_generate.py` |
| ☐ | 7 | Rule filtering | Filtered rules + filter-funnel counts |
| ☐ | 8 | **Evaluation + baselines** | Hit@k tables/curves vs best64/RuleForge/MDBSCAN + complementarity |
| ☐ | 9 | Ablations + statistics | Ablation table + variance/significance |
| ☐ | 10 | Results export | Paper-ready figures + CSVs |

---

## Phase notes

### Phase 0 — Repo scaffold ✅
- `CLAUDE.md`, `pyproject.toml`, `.gitignore`, `README.md`, `PLAYBOOK.md`
- `configs/protocol.yaml` (FROZEN), `configs/train.yaml`
- `pwrules/` package skeleton, `tests/test_import.py`

### Phase 1 — Environment ✅
- `pwrules/env.py` → `check_env(require_gpu, require_hashcat, kaggle_datasets, probe_model)`
- `notebooks/01_smoke.md` — 5-cell Kaggle smoke notebook
- `tests/test_env.py` — 9 tests; GPU tests skip when CUDA absent

### Phase 2 — Clean ✅
- `pwrules/clean/__init__.py` → `clean_password`, `iter_clean`, `compute_stats`, `save_stats`, `split_corpus`, `clean_corpus`, `verify_test_checksum`
- `pwrules/clean/__main__.py` → CLI `python -m pwrules.clean --input ... --out ...`
- `notebooks/02_clean.md`
- `tests/test_clean.py` — 20 tests
- **Kaggle run:** `!python -m pwrules.clean --input /kaggle/input/rockyou/rockyou.txt --out /kaggle/working/clean`
- **Save as:** `yourname/pwrules-clean`
- **Outputs:** `train.txt`, `val.txt`, `test.txt`, `test_checksum.txt`, `split_manifest.json`, `stats/stats.png`, `stats/*.csv`

### Phase 3 — Rule extraction ✅
- `pwrules/ruleextract/applier.py` → full Python Hashcat rule engine (25+ functions)
- `pwrules/ruleextract/extractor.py` → `select_base`, `infer_rule`, `parity_check`
- `pwrules/ruleextract/__init__.py` → `extract_rules` pipeline
- `pwrules/ruleextract/__main__.py` → CLI with `--parity N` flag
- `notebooks/03_ruleextract.md`
- `tests/test_ruleextract.py` — 40+ tests; hashcat parity test skips when hashcat absent
- **Kaggle run:** `!python -m pwrules.ruleextract --clean /kaggle/input/pwrules-clean/clean --out /kaggle/working/rules --parity 200`
- **Save as:** `yourname/pwrules-rules`
- **Outputs:** `rules_dataset.jsonl`, `coverage_report.json`, `rule_frequency.csv`, `rule_frequency.png`, `instructions_train.jsonl`, `instructions_val.jsonl`

### Phase 4 — Target-conditioning ✅
- `pwrules/conditioning/__init__.py` → `generate_synthetic_users`, `assign_synthetic_attributes`, `split_users`, `build_targeted_dataset`
- `pwrules/conditioning/__main__.py` → CLI with `--mode [synthetic|real]`
- `notebooks/04_conditioning.md`
- `tests/test_conditioning.py` — 15 tests; disjoint assertion verified
- **Kaggle run:** `!python -m pwrules.conditioning --rules /kaggle/input/pwrules-rules/rules_dataset.jsonl --mode synthetic --out /kaggle/working/targeted`
- **Save as:** `yourname/pwrules-targeted`
- **Outputs:** `targeted_dataset.jsonl`, `target_users_test.jsonl`, `split_manifest.json`

### Phase 5 — QLoRA fine-tuning ✅
- `pwrules/train/__init__.py` → `load_model_and_tokenizer`, `load_instruction_dataset`, `save_training_curves`, `check_memorisation`, `train`
- `pwrules/train/__main__.py` → CLI with `--targeted`, `--resume` flags
- `notebooks/05_train.md`
- `tests/test_train.py` — CPU-safe tests; GPU smoke skips without CUDA; 1-step GPT-2 stub test
- **Kaggle run:** `!python -m pwrules.train --data /kaggle/input/pwrules-rules --out /kaggle/working/adapter`
- **Save as:** `yourname/pwrules-adapter`
- **Outputs:** `adapter/`, `checkpoints/`, `training_curves.png`, `training_curves.csv`, `memorisation_report.json`

### Phase 6 — Rule generation ✅
- `pwrules/generate/__init__.py` → `load_model_for_inference`, `generate_untargeted`, `generate_targeted`, `write_rule_file`, `compute_diversity_stats`, `generate_rules`
- `pwrules/generate/__main__.py` → CLI with `--target-users`, `--wordlist`, `--budget`
- `notebooks/06_generate.md`
- `tests/test_generate.py` — CPU-safe tests with mock model; 15 tests
- **Kaggle run:** `!python -m pwrules.generate --adapter /kaggle/input/pwrules-adapter/adapter --out /kaggle/working/rules --target-users /kaggle/input/pwrules-targeted/target_users_test.jsonl --budget 10000`
- **Save as:** `yourname/pwrules-generated-rules`
- **Outputs:** `rules/llm_untargeted.rule`, `rules/llm_targeted/<user_id>.rule`, `generation_stats.json`, `rule_length_dist.png`

---

## Next: Phase 7 — Rule filtering
Run `python -m pwrules.filter` after Phase 6 is complete on Kaggle.
