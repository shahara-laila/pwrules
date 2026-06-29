# notebooks/

Thin Kaggle notebooks only — no business logic (that lives in the `pwrules` package).
Each notebook starts with a 3-cell starter (clone repo → install → import + sanity),
then calls `python -m pwrules.<module>`.

**Install only what the phase needs.** The heavy `.[train]` stack (torch/unsloth,
several GB) is what exhausts a CPU session and silently kills the kernel mid-install.
So:

| Phase | Install | Accelerator |
|-------|---------|-------------|
| 2 clean, 4 condition, 9 ablate, 10 export | `pip install -e .` | None (CPU) |
| 3 extract, 7 filter, 8 eval | `pip install -e .` + hashcat | None (CPU) |
| 5 train, 6 generate | `pip install -e ".[train]"` | **GPU** |

Set the Accelerator (right sidebar) to match BEFORE running. Splitting the starter
into separate cells means a slow/failing step is isolated and shows `[n]` when done.
