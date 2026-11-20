# Phish or Not? Does a phishing e-mail detector generalise to unseen e-mail sources?

ESILV Machine Learning 2026 project (major OCC). Team: Imene Khelil, Florian, Ambroise, Dylan.

**Week 2 deliverable:** raw-data audit, learning/testing plan, baseline, state-of-the-art benchmark,
AutoML v0.1. Report: `reports/latex/main.pdf` (LaTeX source in `reports/latex/`).

## Key results (Week 2)

| | F1 | FPR |
|---|---|---|
| Baseline TF-IDF + LR, random hold-out (in-distribution) | 0.991 | 0.9 % |
| Same model, leave-one-source-out (4 binary corpora) | 0.80 – 0.94 | 5.3 – 23.1 % |
| Recall on unseen modern phishing (Nazario 2015–2022) | 0.59 (recall) | – |
| AutoML v0.1 best (ridge, word+char n-grams), hold-out | 0.989 | 0.65 % |
| AutoML v0.1 best, LOSO (4 binary corpora) | 0.73 – 0.92 | 8.5 – 19.2 % |

* The source corpus is predictable from the text with 98.7 % accuracy (95 % even after cleaning
  and masking), so the classifier can learn *where an e-mail comes from*.
* 13.5 % of the e-mails are exact or near-duplicates (one 2008 fake-CNN malware campaign = 3,919 copies).
* Our LOSO numbers reproduce Bhuiyan & Bhuiyan (2026, *BDCC* 10(7):211) to within 0.05 F1 (FPR within 1.1 points).

## Reproduce

```bash
pip install -r requirements.txt
# put the six raw CSVs in data/raw/ (see data/README.md)
bash run_all.sh          # ~45 min on a 2-core laptop; every table/figure is regenerated
cd reports/latex && latexmk -pdf main.tex
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
| `07_latex_tables.py` | report | LaTeX tables generated from the result files |

## Layout

```
src/phish/        config, data loading + dedup, text cleaning, splits, features, metrics, plot style
scripts/          numbered pipeline (run in order)
reports/          figures/, tables/ (CSV + logs), latex/ (report)
ai-governance/    PROMPT_LOG.md, AI_AUDIT.md
data/             raw/ (not committed), processed/ (generated)
```

## Leakage controls

1. Exact and near-duplicate removal **before** any split (MinHash-LSH, Jaccard ≥ 0.8).
2. Hold-out and CV folds **grouped** by near-duplicate cluster and **stratified** on source × label.
3. All fitted transforms (TF-IDF, scaler, χ² selector) inside sklearn `Pipeline`s, refitted per fold.
4. The `source` column is never a feature; metadata fields (missing-not-at-random by source) are not used.
5. Test split frozen in `data/processed/split.csv`, touched once.
