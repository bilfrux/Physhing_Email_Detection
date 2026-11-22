#!/usr/bin/env bash
# Reproduces every number and figure of the Week-2 report from the raw CSVs.
# Expected runtime on a 2-core laptop: ~45 min.
set -euo pipefail
cd "$(dirname "$0")"
python scripts/01_audit_and_build.py   # audit, EDA figures, dedup -> data/processed/corpus.parquet
python scripts/02_split.py             # frozen grouped+stratified 80/20 split + leakage asserts
python scripts/03_shortcut_test.py     # can the source corpus be predicted from text?
python scripts/04_baseline.py          # TF-IDF + LR baseline, validation curve, LOSO, dedup inflation
# AutoML v0.1. The reported search (run 2) was stopped after 34 trials; its log is committed in
# reports/tables/log_05_run2_stopped.txt. 05b replays it deterministically and re-verifies the top
# configurations, 05c evaluates the saved best one. To run a fresh search instead:
#   python scripts/05_automl_v01.py 50
python scripts/05b_recover_best.py     # replay log -> automl_trials.csv, automl_best_config.json (verified)
python scripts/05c_eval_best.py        # best config: hold-out + LOSO
python scripts/06_summary_figures.py   # comparison figures used in the report
python scripts/08_uncertainty.py       # bootstrap / Wilson confidence intervals (no refit)
python scripts/09_sensitivity_checks.py # nested-C LOSO, non-ASCII, cross-source dedup
python scripts/07_latex_tables.py      # LaTeX tables from result files (run last)
