# Phish or Not? Does a phishing e-mail detector generalise to unseen e-mail sources?

ESILV Machine Learning 2026 project (major OCC). Team: Imene Khelil, Florian, Ambroise, Dylan.

**Week 2 deliverable:** raw-data audit, learning/testing plan, baseline, state-of-the-art benchmark,
AutoML v0.1. Report: `reports/latex/rapport.pdf` (short version, the one we submit) and `reports/latex/main.pdf` (long version with every table).

## Key results (Week 2)

| | F1 | FPR |
|---|---|---|
| Baseline TF-IDF + LR, frozen hold-out (in-distribution) | 0.991 | 0.84 % |
| Same model, leave-one-source-out (4 binary corpora) | 0.79 – 0.93 | 5.6 – 24.0 % |
| Recall on unseen modern phishing (Nazario 2015–2022) | 0.58 (recall) | – |
| AutoML v0.1 best (ridge, word+char n-grams), hold-out | 0.990 | 0.68 % |
| AutoML v0.1 best, LOSO (4 binary corpora) | 0.73 – 0.92 | 8.5 – 19.1 % |

Numbers regenerated with `run_all.sh` in the pinned environment; 95 % confidence intervals in
`reports/tables/uncertainty_*.csv` and in the report.

* The source corpus is predictable from the text with 98.7 % accuracy (95 % even after cleaning
  and masking), so the classifier can learn *where an e-mail comes from*.
* 13.5 % of the e-mails are exact or near-duplicates (one 2008 fake-CNN malware campaign = 3,919 copies).
* Our LOSO numbers reproduce Bhuiyan & Bhuiyan (2026, *BDCC* 10(7):211) to within 0.05 F1 (FPR within 1.3 points).

## Reproduce

```bash
python3.13 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
# put the six raw CSVs in data/raw/ (see data/README.md)
PATH="$PWD/.venv/bin:$PATH" bash run_all.sh   # ~1 h 30 on a laptop; every table/figure is regenerated
cd reports/latex && tectonic rapport.tex   # or latexmk -pdf rapport.tex
```

| Script | Guideline step | Output |
|---|---|---|
| `01_audit_and_build.py` | 2.1 raw-data analysis, preprocessing | `reports/tables/audit_*`, Fig. 1–4, `data/processed/corpus.parquet` |
| `02_split.py` | 2.2 learning/testing plan | frozen grouped split + leakage asserts |
| `03_shortcut_test.py` | 2.1/2.2 confound analysis | source-predictability table, top artefact tokens |
| `04_baseline.py` | 2.3 baseline + calibration | validation curve, hold-out, LOSO, dedup-inflation test |
| `05_automl_v01.py` | 2.5 AutoML v0.1 search | Optuna search (run 2 log committed) |
| `05b_recover_best.py` | 2.5 | replays the logged search, re-verifies and saves the best config |
| `05c_eval_best.py` | 2.5 | best config: hold-out + LOSO |
| `06_summary_figures.py` | 2.4 benchmark comparison | Fig. 7–8, summary tables |
| `08_uncertainty.py` | report | bootstrap CIs (hold-out, paired AutoML − baseline), Wilson CIs (LOSO) |
| `09_sensitivity_checks.py` | 2.2 / risks | C re-selected without the held-out source, non-ASCII effect on Nazario, cross-source dedup |
| `07_latex_tables.py` | report | LaTeX tables **and every number quoted in the text** (macros) from the result files |

## Layout

```
src/phish/        config, data loading + dedup, text cleaning, splits, features, metrics, plot style
scripts/          numbered pipeline (run in order)
reports/          figures/, tables/ (CSV + logs), latex/ (report)
ai-governance/    PROMPT_LOG.md, AI_AUDIT.md, CONTRIBUTIONS.md
data/             raw/ (not committed), processed/ (generated)
```

## Leakage controls

1. Exact and near-duplicate removal **before** any split (MinHash-LSH, Jaccard ≥ 0.8).
2. Hold-out and CV folds **stratified** on source × label and grouped by near-duplicate cluster (redundant on the final corpus, which keeps one e-mail per cluster; it matters only in the inflation test, where near-duplicates are kept).
3. All fitted transforms (TF-IDF, scaler, χ² selector) inside sklearn `Pipeline`s, refitted per fold.
4. The `source` column is never a feature; metadata fields (missing-not-at-random by source) are not used.
5. Test split frozen in `data/processed/split.csv`, touched once.
