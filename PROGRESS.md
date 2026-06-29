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
| ☑ | 7 | Rule filtering | Filtered rules + filter-funnel counts |
| ☑ | 8 | **Evaluation + baselines** | Hit@k tables/curves vs best64/RuleForge/MDBSCAN + complementarity |
| ☑ | 9 | Ablations + statistics | Ablation table + variance/significance |
| ☑ | 10 | Results export | Paper-ready figures + CSVs |

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

### Phase 7 — Rule filtering ✅
- `pwrules/filter/__init__.py` → `syntax_check`, `semantic_dedup`, `rank_by_effectiveness`, `filter_rules`
- `pwrules/filter/__main__.py` → CLI `python -m pwrules.filter --rules ... --out ...`
- `notebooks/07_filter.md`
- `tests/test_filter.py` — 20 tests; hashcat probe stage skipped when hashcat absent
- **Kaggle run:** `!python -m pwrules.filter --rules /kaggle/input/pwrules-generated-rules/rules/llm_untargeted.rule --out /kaggle/working/filtered --hashcat-sample 200`
- **Save as:** `yourname/pwrules-filtered`
- **Outputs:** `<stem>_filtered.rule`, `filter_funnel.csv`, `filter_funnel.png`

### Phase 8 — Evaluation + baselines ✅
- `pwrules/eval/__init__.py` → `generate_candidates`, `hit_at_k`, `evaluate_method`, `evaluate_complementarity`, `evaluate_targeted`, `run_eval`
- `pwrules/eval/baselines.py` → `get_best64_rule`, `run_ruleforge` (MDBSCAN → DBSCAN → HAC cascade)
- `pwrules/eval/__main__.py` → CLI `python -m pwrules.eval run ...`
- `notebooks/08_eval.md`
- `tests/test_eval.py` — 20 tests on synthetic data; hashcat calls skipped when absent
- **Kaggle run:** `!python -m pwrules.eval run --wordlist ... --test ... --out /kaggle/working/results --run-ruleforge`
- **Save as:** `yourname/pwrules-results`
- **Outputs:** `results.csv` (method, dataset, k, hit_rate, seed), `guessing_curve.png`, `targeted_results.csv`

### Phase 9 — Ablations + statistics ✅
- `pwrules/eval/ablations.py` → `aggregate_seeds`, `build_ablation_table`, `bootstrap_ci`, `mcnemar_p`, `compute_significance`, `run_ablations`
- `pwrules/eval/__main__.py` → subcommand `python -m pwrules.eval ablate ...`
- `notebooks/09_ablations.md`
- `tests/test_ablations.py` — 15 tests
- **Kaggle run:** `!python -m pwrules.eval ablate --results-dir ... --out /kaggle/working/ablations --baseline best64 --n-bootstrap 10000`
- **Save as:** `yourname/pwrules-ablations`
- **Outputs:** `ablations.csv`, `aggregated_results.csv`, `significance_report.json`

### Phase 10 — Results export ✅
- `pwrules/eval/reporting.py` → `make_hit_at_k_table`, `make_guessing_curve`, `make_targeted_table`, `make_filter_funnel_table`, `make_ablation_table`, `export_paper_artifacts`
- `pwrules/eval/__main__.py` → subcommand `python -m pwrules.eval report ...`
- `notebooks/10_export.md`
- `tests/test_reporting.py` — 18 tests; MISSING marker verified when source files absent
- **Kaggle run:** `!python -m pwrules.eval report --results-dir ... --ablations-dir ... --filter-dir ... --out /kaggle/working/paper`
- **Save as:** `yourname/pwrules-paper`
- **Outputs:** `table_hit_at_k.{csv,tex}`, `guessing_curve.png`, `table_targeted.{csv,tex}`, `table_filter_funnel.{csv,tex}`, `table_ablations.{csv,tex}`, `MISSING.txt`

---

## Hardening pass — audit fixes, path discovery, research figures (2026-06-30)

Full-codebase audit + fixes; all tests green (`255 passed, 4 skipped` — skips are
GPU/hashcat-gated). Local test env: Python 3.12 venv (project default 3.10+).

### New: centralised path discovery — `pwrules/paths.py`
- Slug-agnostic: every CLI finds its inputs by filename under `/kaggle/working`
  then `/kaggle/input`, so notebooks never hard-code a dataset slug.
- Single override point: `PWRULES_<NAME>` env vars (e.g. `PWRULES_ROCKYOU`).
- All `__main__` entrypoints now auto-resolve inputs when flags are omitted.
- `tests/test_paths.py` (10 tests).

### New: research figures — `pwrules/eval/figures.py`
- `pipeline_diagram` (Fig. 1), `rule_op_distribution`, `memorisation_breakdown`,
  `top_rules`, `targeted_vs_untargeted`, `per_user_rule_counts`, `hit_at_k_bars`,
  `complementarity`, `ablation_bars`, and `generate_all_figures` (best-effort).
- Wired into Phase 10 (`export_paper_artifacts` → `figures/`) and Phase 6 emits a
  `rule_op_distribution.png`. `tests/test_figures.py` (9 tests).

### Correctness fixes
- **eval (critical):** Phase 8 crashed on `load_protocol(config_path)` — fixed.
  `hit_at_k` is now true set-intersection (frozen definition); targeted aggregate
  rows are persisted to `results.csv`; `LLM-untargeted` default = unfiltered rules,
  `LLM-filtered` = filtered (so the filtering ablation is real); method name
  standardised to `LLM-targeted`.
- **ablations:** removed the degenerate McNemar surrogate and single-seed
  "significance" (would fabricate p≈1.0 / false significance); significance now
  gated on ≥2 seeds.
- **baselines:** RuleForge Python fallback now infers a real per-cluster rule
  (was emitting a single `c`, which weakened the baseline).
- **clean (critical):** `verify_test_checksum` no longer drops whitespace-only
  passwords (false mismatch on the frozen test split); manifest records the
  *effective* `by_user` mode; UTF-8 decode uses `errors="replace"`.
- **ruleextract:** case ops (`l/u/c/C/t/T`) are ASCII-only to match hashcat on
  non-ASCII base words.
- **conditioning:** empty real-mode `user_id` no longer collapses many users into
  one; `birth_year` normalised to int.
- **filter:** effectiveness ranking counts DISTINCT recovered val passwords; honours
  `top_k` even when hashcat is absent; fixed a wordlist file-handle leak.
- **reporting:** `make_*` create the output dir; LaTeX escapes all cells/headers;
  `Δ` → `Delta` (pdflatex-safe).
- **train/generate (research quality):** response-only SFT loss masking
  (model-agnostic marker derivation); `enable_thinking=False` so Qwen3 emits rules
  not reasoning; generated rules sanitised to one line; explicit stop token;
  early-exit when the unique-rule space saturates; `--targeted` falls back to the
  Phase-3 val split; memorisation check uses diverse probes + sanitised output.

### Tuning — `configs/train.yaml` (not frozen)
- LoRA `r/alpha` 16/32 → 32/64; `max_seq_length` 1024 → 512; batch 16×4 → 8×8
  (same effective 64, safer VRAM); `response_only_loss: true`; generation
  `top_p` 0.95 → 0.9, `max_new_tokens` 64 → 48, added `early_stop_patience`.

## All phases complete ✅
