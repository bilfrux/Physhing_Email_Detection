# CONTRIBUTIONS.md: module ownership matrix

The pipeline is divided into **4 modules, one owner each**, as required by the governance policy.
The owner can explain every function, hyper-parameter and library call of their module at the oral
defence, and authors at least one entry in `PROMPT_LOG.md` **and** one in `AI_AUDIT.md` about it.
Reviewers re-run the module's checks (see *Evidence* in `AI_AUDIT.md`) before each milestone.

> TODO (team): confirm the owner of each module and fill in the "Main contributions" column with
> what each person actually did. Do not leave a module without an owner.

| Module | Owner | Reviewer | Files | Main contributions (week 2) |
|---|---|---|---|---|
| **M1. Data audit & preprocessing** | _to confirm_ | _to confirm_ | `scripts/01_audit_and_build.py`, `src/phish/data.py`, `src/phish/text.py`, `data_profiler.py` | Per-source loading, missingness, class balance, exact + near-duplicate removal (MinHash-LSH), text cleaning and masking, EDA figures 1–4 |
| **M2. Learning/testing plan & leakage control** | _to confirm_ | _to confirm_ | `scripts/02_split.py`, `scripts/03_shortcut_test.py`, `src/phish/split.py` | Frozen grouped + stratified hold-out, leakage asserts, LOSO folds, source-predictability (shortcut) test |
| **M3. Baseline, metrics & benchmark** | _to confirm_ | _to confirm_ | `scripts/04_baseline.py`, `scripts/06_summary_figures.py`, `scripts/08_uncertainty.py`, `src/phish/metrics.py`, `metrics_evaluation.py`, `demo_metrics.py` | TF-IDF + LR baseline, C selection (1-SE rule), calibration, LOSO, comparison with Bhuiyan & Bhuiyan (2026), confidence intervals |
| **M4. AutoML v0.1** | _to confirm_ | _to confirm_ | `scripts/05_automl_v01.py`, `scripts/05b_recover_best.py`, `scripts/05c_eval_best.py`, `src/phish/features.py` | Optuna search space and objective, recovery of the stopped search, hold-out + LOSO of the best configuration |

**Shared:** report (`reports/latex/main.tex`, `scripts/07_latex_tables.py`), `run_all.sh`,
`requirements.txt`, governance files. Each section of the report is written by the owner of the
corresponding module.

## Workload balance

| Member | Module owned | Module reviewed | PROMPT_LOG entries | AI_AUDIT entries |
|---|---|---|---|---|
| Imene Khelil | | | I-1, I-2, I-3 | #1–#5, #9-#12 |
| Florian | | | F-1 | #6, #7, #8 |
| Ambroise | M1 | | A-1, A-2 | #14–#16 |
| Dylan | | | D-1, D-2, D-3, D-4 | #13 |
