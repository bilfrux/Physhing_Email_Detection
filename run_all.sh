#!/usr/bin/env bash
# Rebuilds every table and figure of the report from data/raw (about 45 min).
# Run it inside the venv:  source .venv/bin/activate && bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

python scripts/01_audit_and_build.py    # audit, EDA figures, dedup
python scripts/02_split.py              # frozen 80/20 split + leakage checks
python scripts/03_shortcut_test.py      # can we guess the source from the text?
python scripts/04_baseline.py           # TF-IDF + LR baseline, LOSO

# AutoML: run 2 was stopped after 34 trials, its log is in reports/tables/.
# 05b gets the parameters back from that log, 05c evaluates the best one.
# For a new search instead: python scripts/05_automl_v01.py 50
python scripts/05b_recover_best.py
python scripts/05c_eval_best.py

python scripts/06_summary_figures.py    # figures 7 and 8
python scripts/08_uncertainty.py        # confidence intervals
python scripts/09_sensitivity_checks.py # extra checks
python scripts/07_latex_tables.py       # LaTeX tables, needs all the above
