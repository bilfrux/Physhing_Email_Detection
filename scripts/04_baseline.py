"""Baseline: TF-IDF (unigrams) + logistic regression.

Same model family as Bhuiyan & Bhuiyan (2026), so we can compare, and the
coefficients show which words it uses. C is chosen by 5-fold CV on the training
split (one-standard-error rule), then the model is tested on the hold-out, in
leave-one-source-out, and with / without near-duplicates.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, f1_score

from phish import config as cfg
from phish import viz
from phish.data import load_split
from phish.features import BASELINE_TEXT as TEXT, BASELINE_TFIDF, baseline_lr, baseline_model
from phish.metrics import full_report
from phish.split import cv_folds, holdout_split, loso_folds

viz.setup()
C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]

df, train, test = load_split()
results = {}

dummy = DummyClassifier(strategy="most_frequent").fit(train[[TEXT]], train.label)
results["dummy_most_frequent"] = full_report(test.label, dummy.predict(test[[TEXT]]))

# --- choose C with 5-fold CV on train
rows = []
for k, (a, b) in enumerate(cv_folds(train, n_splits=5)):
    vec = TfidfVectorizer(**BASELINE_TFIDF)
    Xa = vec.fit_transform(train[TEXT].iloc[a])
    Xb = vec.transform(train[TEXT].iloc[b])
    ya, yb = train.label.iloc[a], train.label.iloc[b]
    for c in C_GRID:
        m = baseline_lr(c).fit(Xa, ya)
        rows.append({"fold": k, "C": c, "train_f1": f1_score(ya, m.predict(Xa)), "val_f1": f1_score(yb, m.predict(Xb))})
curve = pd.DataFrame(rows).groupby("C")[["train_f1", "val_f1"]].agg(["mean", "std"])
curve.columns = ["_".join(c) for c in curve.columns]
curve.round(4).to_csv(cfg.TAB / "baseline_validation_curve.csv")

# train F1 hits 1.0 for big C while val F1 stays flat, so we take the smallest C
# within one standard error of the best one
c_best = float(curve.val_f1_mean.idxmax())
se = curve.loc[c_best, "val_f1_std"] / np.sqrt(5)
limit = curve.val_f1_mean.max() - se
best_C = float(curve.index[curve.val_f1_mean >= limit].min())
print(curve.round(4))
print(f"argmax C = {c_best:g}, 1-SE threshold = {limit:.4f}, selected C = {best_C:g}")

fig, ax = plt.subplots(figsize=(5.6, 3.2))
for col, color, label in [("train_f1", viz.COLORS[0], "Training folds"), ("val_f1", viz.COLORS[1], "Validation folds")]:
    mean, std = curve[f"{col}_mean"], curve[f"{col}_std"]
    ax.plot(curve.index, mean, marker="o", ms=4, color=color, label=label)
    ax.fill_between(curve.index, mean - std, mean + std, color=color, alpha=0.15, lw=0)
ax.axvline(best_C, color=viz.INK2, lw=1, ls="--")
ax.text(best_C / 1.2, curve.val_f1_mean.min(), f"selected C = {best_C:g}\n(one-standard-error rule)", fontsize=8, color=viz.INK2, ha="right")
ax.set_xscale("log")
ax.set_xlabel("C (inverse regularisation strength, log scale)")
ax.set_ylabel("F1 (malicious class)")
ax.set_title("Baseline validation curve (5-fold grouped CV)")
ax.legend(loc="center right")
fig.savefig(cfg.FIG / "fig5_validation_curve.png")
plt.close(fig)

# --- final model on the whole train split, tested once on the hold-out
model = baseline_model(best_C).fit(train[TEXT], train.label)
p = model.predict_proba(test[TEXT])[:, 1]
pred = (p >= 0.5).astype(int)
results["baseline_tfidf_lr_holdout"] = full_report(test.label, pred, p)
results["baseline_tfidf_lr_train"] = full_report(train.label, model.predict(train[TEXT]), model.predict_proba(train[TEXT])[:, 1])
per_src = {s: full_report(test.label[test.source == s], pred[test.source == s], p[test.source == s]) for s in cfg.SOURCES}
pd.DataFrame(per_src).T.to_csv(cfg.TAB / "baseline_holdout_per_source.csv")
pd.DataFrame({"source": test.source, "label": test.label, "p_baseline": p}).to_parquet(cfg.PROCESSED / "pred_baseline_test.parquet")
print(json.dumps(results, indent=1))

fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
ConfusionMatrixDisplay.from_predictions(test.label, pred, display_labels=["Legit", "Malicious"], cmap="Blues", ax=axes[0], colorbar=False)
axes[0].grid(False)
axes[0].set_title("Confusion matrix (hold-out test)")
frac_pos, mean_pred = calibration_curve(test.label, p, n_bins=10, strategy="quantile")
brier = results["baseline_tfidf_lr_holdout"]["brier"]
axes[1].plot([0, 1], [0, 1], color=viz.INK2, lw=1, ls="--", label="Perfect calibration")
axes[1].plot(mean_pred, frac_pos, marker="o", ms=4, color=viz.COLORS[0], label=f"TF-IDF + LR (Brier {brier:.3f})")
axes[1].set_xlabel("Mean predicted probability")
axes[1].set_ylabel("Observed fraction malicious")
axes[1].set_title("Reliability diagram")
axes[1].legend(loc="upper left", fontsize=7.5)
fig.suptitle("Baseline on the hold-out test set", fontweight="bold", fontsize=10)
fig.tight_layout()
fig.savefig(cfg.FIG / "fig6_baseline_cm_calibration.png")
plt.close(fig)

# words with the largest weights, to spot shortcuts
vocab = np.array(model.named_steps["tfidf"].get_feature_names_out())
order = np.argsort(model.named_steps["clf"].coef_[0])
top_mal, top_legit = vocab[order[::-1][:25]], vocab[order[:25]]
pd.DataFrame({"top_malicious_tokens": top_mal, "top_legit_tokens": top_legit}).to_csv(cfg.TAB / "baseline_top_tokens.csv", index=False)
print("malicious:", ", ".join(top_mal))
print("legit    :", ", ".join(top_legit))

# --- leave-one-source-out
loso = {}
for src, a, b in loso_folds(df):
    m = baseline_model(best_C).fit(df[TEXT].iloc[a], df.label.iloc[a])
    pp = m.predict_proba(df[TEXT].iloc[b])[:, 1]
    loso[src] = full_report(df.label.iloc[b], (pp >= 0.5).astype(int), pp)
    print("LOSO", src, loso[src])
pd.DataFrame(loso).T.to_csv(cfg.TAB / "baseline_loso.csv")

# --- do near-duplicates make the score look better? (corpus with only exact copies removed)
ex = pd.read_parquet(cfg.PROCESSED / "corpus_exact_dedup.parquet")
inflation = {}
for name, grouped in [("random split (near-duplicates may cross)", False), ("grouped split (clusters kept together)", True)]:
    is_test = holdout_split(ex, grouped=grouped)
    m = baseline_model(best_C).fit(ex[TEXT][~is_test], ex.label[~is_test])
    pp = m.predict_proba(ex[TEXT][is_test])[:, 1]
    crossing = len(set(ex.cluster[~is_test]) & set(ex.cluster[is_test]))
    inflation[name] = {**full_report(ex.label[is_test], (pp >= 0.5).astype(int), pp), "clusters_crossing_split": crossing}
    print(name, inflation[name])
pd.DataFrame(inflation).T.to_csv(cfg.TAB / "dedup_inflation.csv")

out = {"best_C": best_C, "vectorizer": {k: str(v) for k, v in BASELINE_TFIDF.items()},
       "results": results, "loso": loso, "inflation": inflation}
json.dump(out, open(cfg.TAB / "baseline_results.json", "w"), indent=1)
