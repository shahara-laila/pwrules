# Claude Code Prompt Playbook
## Building the Target-Conditioned LLM Password-Rule System

This is a copy-paste playbook. Each phase has (1) a **prompt for Claude Code** that writes the code, (2) **how to run it on Kaggle**, and (3) the **outcome** + a **done-when** gate so you can track progress.

---

## How this works (read once)

Claude Code runs on your Mac and edits a local git repo. It **cannot** run on Kaggle's GPU. So the workflow is:

```
[Your Mac]  Claude Code writes/tests code  →  [GitHub]  push public code-only repo
                                                   │
[Kaggle]  notebook clones repo + installs it + reads data from /kaggle/input + runs on GPU
                                                   │
[Kaggle]  outputs saved to /kaggle/working, then exported as a Kaggle Dataset for persistence
```

Two hard rules that make this safe and clean:

- **Code is public, data is private.** The GitHub repo contains *only code* (no passwords). Leaked corpora are uploaded as **private Kaggle Datasets** and read from `/kaggle/input/...`. `.gitignore` must exclude all data.
- **Never fabricate results.** Claude Code writes code that *produces* numbers; it never hard-codes or invents Hit@k values, and never commits raw passwords.

---

## PART 1 — One-time setup

### 1A. Start Claude Code

On your Mac, in an empty folder:

```bash
mkdir pwrules && cd pwrules
git init
claude
```

Then paste the **Phase 0** prompt first (it scaffolds everything).

### 1B. CLAUDE.md — paste this as your first instruction to Claude Code

Tell Claude Code: *"Create a file CLAUDE.md at the repo root with exactly this content, then follow it for the rest of the project."*

```markdown
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
```

### 1C. Global constraints (these apply to every prompt below)

When in doubt, remind Claude Code of these — they prevent the common failure modes:

- Build only what the current phase asks; don't scaffold future phases.
- Make it runnable on Kaggle (single GPU, `/kaggle/input` read-only, `/kaggle/working` writable).
- Write a test and run it. If it fails, fix before claiming done.
- Update `PROGRESS.md` and stop.

### 1D. Kaggle setup

1. **Upload corpora as private Datasets.** Kaggle → Datasets → New Dataset → upload RockYou (and your other corpora) → set **Private**. Note each slug, e.g. `yourname/rockyou`.
2. **Notebook pattern.** Every Kaggle notebook starts with the standard cell in the Appendix (clone repo, install, install Hashcat, GPU check).
3. **Persistence across the 12h cap.** At the end of a run, save `/kaggle/working/<artifacts>` as a **new Kaggle Dataset** (or a new version of one), then attach it as input to the next notebook. This is how a later phase consumes an earlier phase's output (e.g., the fine-tuned adapter).

---

## PART 2 — Phase tracker

Keep this table in `PROGRESS.md` and tick boxes as you go.

| ✅ | Phase | Builds | Outcome (deliverable) |
|----|-------|--------|-----------------------|
| ☐ | 0 | Repo scaffold + CLAUDE.md + configs + CI | Clean repo, frozen `protocol.yaml`, `PROGRESS.md` |
| ☐ | 1 | Kaggle env + smoke notebook | Notebook: GPU + Hashcat + data read all verified |
| ☐ | 2 | Cleaning/preprocessing + splits | `clean/*` outputs + stats table + train/val/test splits |
| ☐ | 3 | **Rule extraction** | Validated `rules_dataset.jsonl` + coverage% + rule-freq |
| ☐ | 4 | Target-conditioning dataset | `targeted_dataset.jsonl` + held-out target users |
| ☐ | 5 | QLoRA fine-tuning | LoRA adapter + training curves + memorisation check |
| ☐ | 6 | Rule generation (inference) | Untargeted + targeted rule sets |
| ☐ | 7 | Rule filtering | Filtered rules + filter-funnel counts |
| ☐ | 8 | **Evaluation + baselines** | Hit@k tables/curves vs best64/RuleForge/MDBSCAN + complementarity |
| ☐ | 9 | Ablations + statistics | Ablation table + variance/significance |
| ☐ | 10 | Results export | Paper-ready figures + CSVs |

---

## PART 3 — Phase prompts

> Paste the **Prompt** block into Claude Code. Then follow **Run on Kaggle**. Confirm **Done-when** before ticking the box.

---

### Phase 0 — Repo scaffold

**Prompt:**
```
Create the CLAUDE.md exactly as I specified, then scaffold the repo:
- pyproject.toml installing pwrules as an editable package with a CLI; pin
  unsloth, transformers, peft, trl, bitsandbytes, datasets, accelerate, pyyaml,
  pandas, scipy, matplotlib, pytest.
- Package skeleton pwrules/{clean,ruleextract,conditioning,train,generate,filter,eval}/__init__.py
  with empty module stubs and a shared pwrules/config.py that loads YAML + sets SEED.
- configs/protocol.yaml with FROZEN keys: base_wordlist_path, guess_budget
  (list: 10,100,1000,...,1e7), split = {train,val,test ratios, by_user: true},
  hit_at_k_definition (string), seed. Fill sensible defaults.
- configs/train.yaml with model_name (default a small Qwen3 instruct variant),
  qlora params (r, alpha, dropout, 4bit), training params (epochs, lr, batch,
  early_stopping), generation params (budget, temperature, top_p).
- .gitignore excluding data/, model weights, *.txt, /kaggle outputs, __pycache__.
- PROGRESS.md containing the phase tracker table (I'll paste it).
- A trivial pytest that imports the package.
Run the test, then stop.
```

**Run on Kaggle:** none yet.
**Outcome:** clean installable repo, frozen `protocol.yaml`, `PROGRESS.md` tracker.
**Done-when:** `pytest` passes locally; you've pushed the repo to a **public** GitHub.

---

### Phase 1 — Kaggle environment + smoke notebook

**Prompt:**
```
Create notebooks/01_smoke.ipynb (or a markdown file notebooks/01_smoke.md with
the exact cells) that, when run on Kaggle:
1. Clones this repo and pip-installs it editable.
2. Installs Hashcat (apt) and prints `hashcat -I` plus a `--stdout` test applying
   a trivial rule to one probe word.
3. Prints nvidia-smi and runs a 1-step dummy QLoRA forward/backward on the
   configured base model to confirm it fits in VRAM (catch OOM early).
4. Lists /kaggle/input to confirm the private dataset(s) are attached, and prints
   the first 5 lines of the corpus.
Also write pwrules/env.py with a check_env() function the notebook calls. Add a
test for check_env that runs without a GPU (skip GPU-only asserts). Stop.
```

**Run on Kaggle:** new notebook → paste the standard cell (Appendix) → attach your private RockYou dataset → run all. Enable GPU.
**Outcome:** a notebook proving GPU + Hashcat + data access all work.
**Done-when:** smoke notebook runs top-to-bottom on Kaggle with no errors and no OOM.

---

### Phase 2 — Cleaning, preprocessing, splitting

**Prompt:**
```
Implement pwrules/clean with a CLI `python -m pwrules.clean`:
- Input: a raw corpus path (read-only) and an output dir.
- Steps: normalise to UTF-8 (drop/repair invalid bytes), strip control chars,
  remove exact duplicates, optional length/charset filter (configurable, off by
  default), and emit cleaned passwords.
- Compute stats: length histogram, character-class composition, top-N tokens →
  save as CSV + a matplotlib PNG.
- Split into train/val/test per configs/protocol.yaml; if by_user is true and a
  user field exists, split by user and ASSERT zero overlap; else split by row.
  Save split manifests and FREEZE the test split (write a checksum).
Add tests on a tiny synthetic corpus. Update PROGRESS.md. Stop.
```

**Run on Kaggle:**
```python
!python -m pwrules.clean --input /kaggle/input/rockyou/rockyou.txt --out /kaggle/working/clean
```
Then save `/kaggle/working/clean` as a Kaggle Dataset (e.g. `yourname/pwrules-clean`).

**Outcome:** cleaned corpus + dataset-stats table/figure + frozen splits.
**Done-when:** splits are disjoint (assert passes), stats CSV + figure produced.

---

### Phase 3 — Rule extraction (CORE)

**Prompt:**
```
Implement pwrules/ruleextract with CLI `python -m pwrules.ruleextract`. Goal:
convert each cleaned password into a validated (base_word, rule, password) triple.
Algorithm:
1. Base-word selection: strip leading/trailing digits and symbols, lowercase the
   alphabetic core, and match against a reference wordlist (configurable; default
   the unique alphabetic tokens of the attack base wordlist so rules are
   applicable at attack time). If no base found, record as 'no_base' and skip.
2. Transformation inference → ordered Hashcat primitives, detected in this order:
   case op (c / u / t), leetspeak substitutions from a fixed map (a->@, e->3,
   o->0, i->1, s->$, ...), appended suffix ($x per char), prepended prefix (^x),
   reversal (r), duplication (d). Compose the rule string.
3. Reproduce-or-discard validation: apply the inferred rule to the base word and
   keep the triple ONLY if it exactly regenerates the password. Implement the
   rule applier in Python for the supported primitive subset, AND add a parity
   test that cross-checks a random sample against real `hashcat --stdout` to
   confirm identical output. Discard non-reproducing rules.
Outputs: rules_dataset.jsonl ({base, rule, password}); a coverage report (% of
passwords with a validated rule); rule_frequency.csv + a PNG. Also emit the
instruction-format dataset (input=base word, output=rule) as train/val jsonl.
Add tests including the hashcat parity test (skippable if hashcat absent). Update
PROGRESS.md. Stop.
```

**Run on Kaggle:** (Hashcat present from the standard cell)
```python
!python -m pwrules.ruleextract --clean /kaggle/input/pwrules-clean/clean --out /kaggle/working/rules
```
Save `/kaggle/working/rules` as a Kaggle Dataset.

**Outcome:** validated `rules_dataset.jsonl`, coverage %, rule-frequency figure, instruction train/val files.
**Done-when:** hashcat-parity test passes; coverage reported; 30 random triples spot-checked by you and correct.

---

### Phase 4 — Target-conditioning dataset

**Prompt:**
```
Implement pwrules/conditioning with CLI `python -m pwrules.conditioning`.
- Input: rules_dataset.jsonl plus a user-attribute source (real attribute-linked
  data if available; otherwise a synthetic attribute generator that attaches
  name/birth_year/interest to passwords — clearly flag synthetic mode in a
  metadata field). Make real-vs-synthetic a CLI flag.
- Output: targeted instruction examples (input = structured attributes + base
  word; output = the rule) as jsonl, and a held-out target_users_test.jsonl whose
  users do NOT appear in training (assert disjoint).
Add tests. Update PROGRESS.md. Stop.
```

**Run on Kaggle:**
```python
!python -m pwrules.conditioning --rules /kaggle/input/pwrules-rules/rules_dataset.jsonl --mode synthetic --out /kaggle/working/targeted
```

**Outcome:** targeted instruction dataset + held-out target users.
**Done-when:** target users verified disjoint from training; synthetic/real flag recorded in metadata.

---

### Phase 5 — QLoRA fine-tuning

**Prompt:**
```
Implement pwrules/train with CLI `python -m pwrules.train`.
- Use Unsloth to load the configured base model in 4-bit, attach QLoRA adapters
  (r/alpha/dropout from configs/train.yaml) on attention + MLP projections.
- Train with TRL SFTTrainer on the instruction dataset using the model's chat
  template; validation eval every N steps; early stopping on val loss; set SEED.
- Save the adapter to the output dir; save training/val loss curves as PNG+CSV.
- Memorisation check: after training, sample generations and report the fraction
  of produced rules that are NOVEL vs present in the training set; write to a
  report file. Make the run resumable from a checkpoint dir (for the 12h cap).
Add a tiny CPU-only smoke test (1 step on a stub) that's skipped if no GPU.
Update PROGRESS.md. Stop.
```

**Run on Kaggle:** attach the rules dataset + (optionally) a checkpoint dataset, then:
```python
!python -m pwrules.train --data /kaggle/input/pwrules-rules --out /kaggle/working/adapter --config configs/train.yaml
```
Save `/kaggle/working/adapter` as a Kaggle Dataset (this is the input to Phase 6).

**Outcome:** LoRA adapter + training curves + memorisation report.
**Done-when:** val loss plateaus/early-stops; memorisation check shows the model generalises (high novel-rule rate), not memorises.

---

### Phase 6 — Rule generation (inference)

**Prompt:**
```
Implement pwrules/generate with CLI `python -m pwrules.generate`.
- Load base model + the trained adapter. Generate a FIXED budget of rules
  (from config) for two regimes: (a) untargeted, (b) targeted per held-out user
  (input = that user's attributes). Use temperature/top-p (or beam) from config.
- Deduplicate exact rules. Write rules/llm_untargeted.rule and
  rules/llm_targeted/<user>.rule (one rule per line, Hashcat format).
- Report a diversity stat (unique rules, rule-length distribution).
Add a CPU stub test. Update PROGRESS.md. Stop.
```

**Run on Kaggle:** attach the adapter dataset, then:
```python
!python -m pwrules.generate --adapter /kaggle/input/pwrules-adapter/adapter --out /kaggle/working/rules
```

**Outcome:** untargeted + targeted rule sets at the fixed budget.
**Done-when:** rule files produced in valid Hashcat format; diversity stats reported.

---

### Phase 7 — Rule filtering

**Prompt:**
```
Implement pwrules/filter with CLI `python -m pwrules.filter`, a 3-stage funnel:
1. Syntax check: probe each rule via `hashcat -r <rule> --stdout` on a probe word;
   discard errors and no-ops.
2. Deduplicate exact and semantically-equivalent rules (normalise then compare).
3. Optional effectiveness ranking on the VALIDATION dictionary only (never test).
Output filtered .rule files and filter_funnel.csv (generated→valid→unique→
effective counts). Add tests (skip hashcat stage if absent). Update PROGRESS.md.
Stop.
```

**Run on Kaggle:**
```python
!python -m pwrules.filter --rules /kaggle/working/rules --val /kaggle/input/pwrules-rules/val.jsonl --out /kaggle/working/rules_filtered
```

**Outcome:** filtered rule sets + filtering-funnel CSV.
**Done-when:** every surviving rule passes the hashcat syntax probe; funnel CSV complete.

---

### Phase 8 — Evaluation + baselines (CORE)

**Prompt:**
```
Implement pwrules/eval with CLI `python -m pwrules.eval`. Follow protocol.yaml
EXACTLY (base wordlist, guess budget, frozen test split, Hit@k definition).
- Candidate generation: for a given ruleset, run
  `hashcat -r <ruleset> <base_wordlist> --stdout`, deduplicate preserving order,
  truncate to each budget k.
- Hit@k = |candidates[:k] ∩ test_plaintexts| / |test_plaintexts|. Compute for the
  full k schedule; save Hit@k CSV and a guessing-number curve (log-x) PNG.
- Baseline runners (identical wordlist, identical k):
  * best64 (ships with hashcat at /usr/share/hashcat/rules/best64.rule),
  * RuleForge (clone its public repo; expose its clustering variants incl.
    MDBSCAN — note MDBSCAN needs .NET SDK 7.0; make that an optional dependency
    and degrade gracefully to DBSCAN/HAC if .NET is unavailable, logging which
    ran),
  * optional neural baseline producing an equivalent guess count.
- Complementarity: Hit@k of (LLM rules ∪ best64) candidates.
- Cross-dataset: parameterise so rules trained on corpus A can be evaluated on
  corpus B's test set.
- Targeted eval: per held-out user, generate candidates from their rules and
  check membership of their password within budget; aggregate.
Output one tidy results CSV (method, dataset, k, hit_rate, seed) + curves. Add
tests on a tiny synthetic wordlist/test set. Update PROGRESS.md. Stop.
```

**Run on Kaggle:**
```python
!python -m pwrules.eval --rules /kaggle/working/rules_filtered --wordlist <base_wordlist> --test /kaggle/input/pwrules-clean/clean/test.txt --out /kaggle/working/results
```
(Do the RuleForge/.NET baseline in a dedicated notebook if .NET install is slow.)

**Outcome:** Hit@k tables + guessing-number curves for your method and all baselines, plus complementarity, cross-dataset, and targeted results.
**Done-when:** all methods evaluated under identical budgets on ≥2 datasets; results CSV + curves produced.

---

### Phase 9 — Ablations + statistics

**Prompt:**
```
Implement pwrules/eval ablation + stats utilities and a CLI to run them:
- Ablations: (i) target-conditioning on/off, (ii) model size, (iii) with/without
  filtering, (iv) cross-dataset. Reuse the eval harness; never re-implement Hit@k.
- Variance: run each setting over >=3 seeds; report mean ± std. Add significance
  vs the strongest baseline (bootstrap CI or McNemar's). Output ablations.csv and
  a significance report. Add tests. Update PROGRESS.md. Stop.
```

**Run on Kaggle:** re-run eval per seed/setting (script the loop).
**Outcome:** ablation table + variance + significance.
**Done-when:** each contribution isolated by an ablation; mean±std and significance reported.

---

### Phase 10 — Results export

**Prompt:**
```
Add pwrules/eval reporting that reads the results/ablation CSVs and emits
PAPER-READY artifacts: (1) the main Hit@k comparison table as CSV+LaTeX,
(2) the guessing-number curve figure, (3) the targeted table, (4) the filter
funnel, (5) the ablation table. Match the column structure of my paper's Table 3
and Table 4. Do NOT invent numbers — read them from the result files; if a file
is missing, write a clear 'MISSING' marker. Update PROGRESS.md. Stop.
```

**Run on Kaggle / locally:** run the reporting CLI on your collected result CSVs.
**Outcome:** figures + CSV/LaTeX tables ready to drop into the paper's placeholders.
**Done-when:** every paper placeholder has a matching exported artifact (or a clear MISSING marker).

---

## Appendix — Standard Kaggle notebook starter cell

Put this at the top of every Kaggle notebook. Replace placeholders.

```python
# 1. Get the code
!git clone https://github.com/<YOUR_GH_USER>/pwrules.git
%cd pwrules
!pip install -q -e .
# 2. Cracking tool (CPU --stdout is all we need)
!apt-get -qq update && apt-get -qq install -y hashcat
!hashcat -I 2>/dev/null | head -5 || echo "hashcat installed (no GPU backend needed for --stdout)"
# 3. Sanity
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
import os; print("inputs:", os.listdir("/kaggle/input"))
```

**Persistence reminder:** after a run, Kaggle → notebook → *Save Version*, then add `/kaggle/working/<artifact>` as a new **Dataset** (or new version) and attach it as input to the next phase's notebook. That is how Phase 5's adapter reaches Phase 6, etc.

---

## Tips for getting good code out of Claude Code

- One phase per session; let it finish, review, then continue. Bounded tasks beat "build everything."
- If it over-engineers, say: "minimal, just this phase, make the test pass."
- If a Kaggle run errors, paste the full traceback back to Claude Code and ask it to fix the module + test.
- Re-paste the relevant CLAUDE.md non-negotiables if it drifts (especially: no fabricated results, no committed data, protocol is frozen).
- Keep PROGRESS.md as the single source of truth for what's done and where each output lives.
