"""Confidence intervals (no model is refitted, we reuse the saved predictions).

1. Hold-out: 95% bootstrap intervals for the baseline and the AutoML model, and
   the paired difference AutoML - baseline on the same resamples (+ p-value).
   We resample clusters; after dedup each cluster is one mail, so this is a
   normal bootstrap, but it stays right if near-duplicates are kept one day.
2. Leave-one-source-out: Wilson intervals for recall and FPR from the counts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from phish import config as cfg
from phish.data import load_split

N_BOOT = 2000
rng = np.random.default_rng(cfg.SEED)

_, _, test = load_split(columns=["source", "label", "cluster"])
pred_base = pd.read_parquet(cfg.PROCESSED / "pred_baseline_test.parquet")
pred_auto = pd.read_parquet(cfg.PROCESSED / "pred_automl_test.parquet")
for p in (pred_base, pred_auto):  # same rows, same order as the test split
    assert len(p) == len(test) and (p.label.values == test.label.values).all() and (p.source.values == test.source.values).all()

y = test.label.values
s_base = pred_base.p_baseline.values
s_auto = pred_auto.score.values
yhat_base = (s_base >= 0.5).astype(int)
# 05c saved predict_proba for models that have it, else decision_function (threshold 0)
auto_model = json.load(open(cfg.TAB / "automl_best_config.json"))["best"]["params"]["model"]
threshold = 0.5 if auto_model in {"logreg", "complement_nb", "sgd_huber", "decision_tree"} else 0.0
yhat_auto = (s_auto >= threshold).astype(int)


def metrics(y, yhat, s):
    tp = np.sum((y == 1) & (yhat == 1))
    fp = np.sum((y == 0) & (yhat == 1))
    fn = np.sum((y == 1) & (yhat == 0))
    tn = np.sum((y == 0) & (yhat == 0))
    return {
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "f1": 2 * tp / (2 * tp + fp + fn),
        "fpr": fp / (fp + tn),
        "roc_auc": roc_auc_score(y, s),
    }


# the point estimates must be the ones already in the report
point = {"baseline": metrics(y, yhat_base, s_base), "automl": metrics(y, yhat_auto, s_auto)}
saved = {"baseline": json.load(open(cfg.TAB / "baseline_results.json"))["results"]["baseline_tfidf_lr_holdout"],
         "automl": json.load(open(cfg.TAB / "automl_best_results.json"))["holdout"]}
for name in point:
    for k in ("f1", "fpr", "recall"):
        assert abs(point[name][k] - saved[name][k]) < 1e-4, f"{name} {k}: {point[name][k]:.4f} != saved {saved[name][k]}"
print(f"point estimates match the saved results (AutoML model = {auto_model}, threshold = {threshold})")

# --- 1. bootstrap on the hold-out
_, cluster_id = np.unique(test.cluster.values, return_inverse=True)
n_clusters = cluster_id.max() + 1
members = [np.flatnonzero(cluster_id == k) for k in range(n_clusters)]
print(f"test set: {len(test):,} e-mails in {n_clusters:,} near-duplicate clusters; {N_BOOT} bootstrap replicates")

boot = {"baseline": [], "automl": [], "diff": []}
for _ in range(N_BOOT):
    idx = np.concatenate([members[k] for k in rng.integers(0, n_clusters, n_clusters)])
    mb = metrics(y[idx], yhat_base[idx], s_base[idx])
    ma = metrics(y[idx], yhat_auto[idx], s_auto[idx])
    boot["baseline"].append(mb)
    boot["automl"].append(ma)
    boot["diff"].append({k: ma[k] - mb[k] for k in mb})

rows = []
for name, reps in boot.items():
    reps = pd.DataFrame(reps)
    for k in reps.columns:
        est = point["automl"][k] - point["baseline"][k] if name == "diff" else point[name][k]
        lo, hi = np.percentile(reps[k], [2.5, 97.5])
        row = {"model": name, "metric": k, "estimate": est, "ci_low": lo, "ci_high": hi}
        if name == "diff":  # two-sided p-value for "no difference"
            row["p_value"] = min(1.0, 2 * min((reps[k] <= 0).mean(), (reps[k] >= 0).mean()))
        rows.append(row)
hold = pd.DataFrame(rows).round(5)
hold.to_csv(cfg.TAB / "uncertainty_holdout.csv", index=False)
print(hold.to_string(index=False))


# --- 2. Wilson intervals for leave-one-source-out
def wilson(k, n, z=norm.ppf(0.975)):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


loso = {"baseline": json.load(open(cfg.TAB / "baseline_results.json"))["loso"],
        "automl": json.load(open(cfg.TAB / "automl_best_results.json"))["loso"]}
rows = []
for model, per_src in loso.items():
    for src, r in per_src.items():
        n_pos = round(r["n"] * r["pos_rate"])
        n_neg = r["n"] - n_pos
        lo, hi = wilson(round(r["recall"] * n_pos), n_pos)
        row = {"model": model, "held_out_source": src, "n_pos": n_pos, "n_neg": n_neg,
               "recall": r["recall"], "recall_ci_low": lo, "recall_ci_high": hi}
        if n_neg > 0:
            lo, hi = wilson(round(r["fpr"] * n_neg), n_neg)
            row.update(fpr=r["fpr"], fpr_ci_low=lo, fpr_ci_high=hi)
        rows.append(row)
loso_ci = pd.DataFrame(rows).round(4)
loso_ci.to_csv(cfg.TAB / "uncertainty_loso.csv", index=False)
print(loso_ci.to_string(index=False))
