"""Uncertainty of the reported numbers (no model is refitted).

1. Hold-out test: 95 % percentile-bootstrap confidence intervals for the baseline
   and the AutoML v0.1 best configuration, plus the PAIRED difference
   (AutoML - baseline) on the same resampled e-mails.
   Resampling unit = near-duplicate cluster. On the de-duplicated corpus every
   cluster holds one e-mail, so this equals an ordinary bootstrap; the unit is
   kept explicit so the script stays correct if near-duplicates are ever kept
   (copies of one campaign are not independent observations).
2. Leave-one-source-out: Wilson 95 % intervals for recall and FPR, computed from
   the counts (n x pos_rate) stored by 04_baseline.py and 05c_eval_best.py.

Input : data/processed/pred_baseline_test.parquet, pred_automl_test.parquet
        reports/tables/baseline_results.json, automl_best_results.json
Output: reports/tables/uncertainty_holdout.csv, uncertainty_loso.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from phish import config as C

B = 2000
rng = np.random.default_rng(C.SEED)

# ----------------------------------------------------------------- inputs
df = pd.read_parquet(C.PROCESSED / "corpus.parquet", columns=["source", "label", "cluster"])
split = pd.read_csv(C.PROCESSED / "split.csv")
te = df[split.split.values == "test"].reset_index(drop=True)
pb = pd.read_parquet(C.PROCESSED / "pred_baseline_test.parquet")
pa = pd.read_parquet(C.PROCESSED / "pred_automl_test.parquet")
for p in (pb, pa):  # predictions must be row-aligned with the frozen test split
    assert len(p) == len(te) and (p.label.values == te.label.values).all() and (p.source.values == te.source.values).all()

y = te.label.values
s_base = pb.p_baseline.values
s_auto = pa.score.values
yhat_base = (s_base >= 0.5).astype(int)  # LR probability, threshold used in 04_baseline.py
# 05c stores scores(): predict_proba for probabilistic models (threshold 0.5), decision_function
# otherwise (ridge_lsq, linear_svm: predict() thresholds it at 0).
best_model = json.load(open(C.TAB / "automl_best_config.json"))["best"]["params"]["model"]
PROBA_MODELS = {"logreg", "complement_nb", "sgd_huber", "decision_tree"}
thr_auto = 0.5 if best_model in PROBA_MODELS else 0.0
yhat_auto = (s_auto >= thr_auto).astype(int)


def metrics(y, yhat, s):
    tp = np.sum((y == 1) & (yhat == 1)); fp = np.sum((y == 0) & (yhat == 1))
    fn = np.sum((y == 1) & (yhat == 0)); tn = np.sum((y == 0) & (yhat == 0))
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    return {"precision": prec, "recall": rec, "f1": 2 * tp / (2 * tp + fp + fn), "fpr": fp / (fp + tn), "roc_auc": roc_auc_score(y, s)}


# Guard: the point estimates must equal the ones already reported (same predictions, same rule).
point = {"baseline": metrics(y, yhat_base, s_base), "automl": metrics(y, yhat_auto, s_auto)}
saved_b = json.load(open(C.TAB / "baseline_results.json"))["results"]["baseline_tfidf_lr_holdout"]
saved_a = json.load(open(C.TAB / "automl_best_results.json"))["holdout"]
for name, saved in [("baseline", saved_b), ("automl", saved_a)]:
    for k in ("f1", "fpr", "recall"):
        assert abs(point[name][k] - saved[k]) < 1e-4, f"{name} {k}: recomputed {point[name][k]:.4f} != saved {saved[k]}"
print(f"point estimates match the saved results (AutoML model = {best_model}, threshold = {thr_auto})")

# ----------------------------------------------------------------- 1. cluster bootstrap on the hold-out
clusters = te.cluster.values
uniq, inv = np.unique(clusters, return_inverse=True)
members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
print(f"test set: {len(te):,} e-mails in {len(uniq):,} near-duplicate clusters; {B} bootstrap replicates")

boot = {"baseline": [], "automl": [], "diff": []}
for _ in range(B):
    idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
    mb = metrics(y[idx], yhat_base[idx], s_base[idx])
    ma = metrics(y[idx], yhat_auto[idx], s_auto[idx])
    boot["baseline"].append(mb)
    boot["automl"].append(ma)
    boot["diff"].append({k: ma[k] - mb[k] for k in mb})

rows = []
for name in ("baseline", "automl", "diff"):
    bd = pd.DataFrame(boot[name])
    for k in bd.columns:
        est = point["automl"][k] - point["baseline"][k] if name == "diff" else point[name][k]
        lo, hi = np.percentile(bd[k], [2.5, 97.5])
        row = {"model": name, "metric": k, "estimate": est, "ci_low": lo, "ci_high": hi}
        if name == "diff":  # two-sided bootstrap p-value for H0: no difference
            row["p_value"] = min(1.0, 2 * min((bd[k] <= 0).mean(), (bd[k] >= 0).mean()))
        rows.append(row)
hold = pd.DataFrame(rows).round(5)
hold.to_csv(C.TAB / "uncertainty_holdout.csv", index=False)
print(hold.to_string(index=False))


# ----------------------------------------------------------------- 2. Wilson intervals on LOSO
def wilson(k, n, z=norm.ppf(0.975)):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


loso_sources = {
    "baseline": json.load(open(C.TAB / "baseline_results.json"))["loso"],
    "automl": json.load(open(C.TAB / "automl_best_results.json"))["loso"],
}
rows = []
for model, loso in loso_sources.items():
    for src, r in loso.items():
        n_pos = round(r["n"] * r["pos_rate"])
        n_neg = r["n"] - n_pos
        lo, hi = wilson(round(r["recall"] * n_pos), n_pos)
        row = {"model": model, "held_out_source": src, "n_pos": n_pos, "n_neg": n_neg, "recall": r["recall"], "recall_ci_low": lo, "recall_ci_high": hi}
        if n_neg > 0:
            lo, hi = wilson(round(r["fpr"] * n_neg), n_neg)
            row.update(fpr=r["fpr"], fpr_ci_low=lo, fpr_ci_high=hi)
        rows.append(row)
loso_ci = pd.DataFrame(rows).round(4)
loso_ci.to_csv(C.TAB / "uncertainty_loso.csv", index=False)
print(loso_ci.to_string(index=False))
