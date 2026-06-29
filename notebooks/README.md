# notebooks/

Thin Kaggle notebooks only — no business logic (that lives in the `pwrules` package).
Each notebook starts with the standard starter cell from [PLAYBOOK.md](../PLAYBOOK.md)
(clone repo → `pip install -e .` → install Hashcat → GPU check), then calls
`python -m pwrules.<module>`.

Planned:
- `01_smoke.md` / `01_smoke.ipynb` — Phase 1 environment smoke test.
