"""Step 2.3 - baseline: TF-IDF (unigrams) + Logistic Regression.

Why this baseline: it is the standard text-classification baseline, it is the
exact model family of the state-of-the-art reference (Bhuiyan & Bhuiyan 2026),
so our numbers are directly comparable, it is fast, and its coefficients are
interpretable (we can SEE shortcut tokens).

Calibration of the model (hyper-parameter C = inverse L2 regularisation
strength) is chosen by 5-fold grouped CV on the training split only, with a
validation curve to diagnose under-/over-fitting. Probability calibration is
checked with a reliability diagram + Brier score.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from sklearn.pipeline import Pipeline

from phish import config as C
from phish import viz
from phish.metrics import full_report
from phish.split import cv_folds, holdout_split, loso_folds

viz.setup()
TEXT = "text_min"
VEC = dict(ngram_range=(1, 1), min_df=3, max_df=0.95, sublinear_tf=True, max_features=100_000)
C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]


def make_model(c):
    return Pipeline([("tfidf", TfidfVectorizer(**VEC)), ("clf", LogisticRegression(C=c, solver="liblinear", max_iter=2000, random_state=C.SEED))])


df = pd.read_parquet(C.PROCESSED / "corpus.parquet")
split = pd.read_csv(C.PROCESSED / "split.csv")
tr = df[split.split.values == "train"].reset_index(drop=True)
te = df[split.split.values == "test"].reset_index(drop=True)
results = {}

# ------------------------------------------------------------- 0. trivial baselines
dummy = DummyClassifier(strategy="most_frequent").fit(tr[[TEXT]], tr.label)
results["dummy_most_frequent"] = full_report(te.label, dummy.predict(te[[TEXT]]))

# ------------------------------------------------------------- 1. choose C by grouped CV
rows = []
for k, (f_tr, f_va) in enumerate(cv_folds(tr, n_splits=5)):
    vec = TfidfVectorizer(**VEC)
    Xtr = vec.fit_transform(tr[TEXT].iloc[f_tr])  # fitted on the training fold ONLY
    Xva = vec.transform(tr[TEXT].iloc[f_va])
    for c in C_GRID:
        m = LogisticRegression(C=c, solver="liblinear", max_iter=2000, random_state=C.SEED).fit(Xtr, tr.label.iloc[f_tr])
        rows.append({"fold": k, "C": c, "train_f1": f1_score(tr.label.iloc[f_tr], m.predict(Xtr)), "val_f1": f1_score(tr.label.iloc[f_va], m.predict(Xva))})
cvres = pd.DataFrame(rows)
curve = cvres.groupby("C")[["train_f1", "val_f1"]].agg(["mean", "std"])
curve.columns = ["_".join(c) for c in curve.columns]
curve.round(4).to_csv(C.TAB / "baseline_validation_curve.csv")
# One-standard-error rule: the most regularised C whose mean validation F1 is
# within one standard error of the best one. Rationale: train F1 reaches 1.000 for
# large C (memorisation) while validation F1 is flat -> prefer the simpler model.
c_max = float(curve.val_f1_mean.idxmax())
se = curve.loc[c_max, "val_f1_std"] / np.sqrt(5)
best_C = float(curve.index[curve.val_f1_mean >= curve.val_f1_mean.max() - se].min())
print(curve.round(4))
print(f"argmax C = {c_max:g}, 1-SE threshold = {curve.val_f1_mean.max() - se:.4f}, selected C = {best_C:g}")

fig, ax = plt.subplots(figsize=(5.6, 3.2))
for col, color, lab in [("train_f1", viz.SERIES[0], "Training folds"), ("val_f1", viz.SERIES[1], "Validation folds")]:
    ax.plot(curve.index, curve[f"{col}_mean"], marker="o", ms=4, color=color, label=lab)
    ax.fill_between(curve.index, curve[f"{col}_mean"] - curve[f"{col}_std"], curve[f"{col}_mean"] + curve[f"{col}_std"], color=color, alpha=0.15, lw=0)
ax.axvline(best_C, color=viz.INK2, lw=1, ls="--")
ax.text(best_C / 1.2, curve.val_f1_mean.min(), f"selected C = {best_C:g}\n(one-standard-error rule)", fontsize=8, color=viz.INK2, ha="right")
ax.set_xscale("log")
ax.set_xlabel("C (inverse regularisation strength, log scale)")
ax.set_ylabel("F1 (malicious class)")
ax.set_title("Baseline validation curve (5-fold grouped CV)")
ax.legend(loc="center right")
fig.savefig(C.FIG / "fig5_validation_curve.png")
plt.close(fig)

# ------------------------------------------------------------- 2. final fit -> hold-out test
model = make_model(best_C).fit(tr[TEXT], tr.label)
p = model.predict_proba(te[TEXT])[:, 1]
yhat = (p >= 0.5).astype(int)
results["baseline_tfidf_lr_holdout"] = full_report(te.label, yhat, p)
results["baseline_tfidf_lr_train"] = full_report(tr.label, model.predict(tr[TEXT]), model.predict_proba(tr[TEXT])[:, 1])
per_src = {s: full_report(te.label[te.source == s], yhat[te.source == s], p[te.source == s]) for s in C.SOURCES}
pd.DataFrame(per_src).T.to_csv(C.TAB / "baseline_holdout_per_source.csv")
pd.DataFrame({"source": te.source, "label": te.label, "p_baseline": p}).to_parquet(C.PROCESSED / "pred_baseline_test.parquet")
print(json.dumps(results, indent=1))

fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
ConfusionMatrixDisplay.from_predictions(te.label, yhat, display_labels=["Legit", "Malicious"], cmap="Blues", ax=axes[0], colorbar=False)
axes[0].grid(False)
axes[0].set_title("Confusion matrix (hold-out test)")
frac_pos, mean_pred = calibration_curve(te.label, p, n_bins=10, strategy="quantile")
axes[1].plot([0, 1], [0, 1], color=viz.INK2, lw=1, ls="--", label="Perfect calibration")
axes[1].plot(mean_pred, frac_pos, marker="o", ms=4, color=viz.SERIES[0], label=f"TF-IDF + LR (Brier {results['baseline_tfidf_lr_holdout']['brier']:.3f})")
axes[1].set_xlabel("Mean predicted probability")
axes[1].set_ylabel("Observed fraction malicious")
axes[1].set_title("Reliability diagram")
axes[1].legend(loc="upper left", fontsize=7.5)
fig.suptitle("Baseline on the hold-out test set", fontweight="bold", fontsize=10)
fig.tight_layout()
fig.savefig(C.FIG / "fig6_baseline_cm_calibration.png")
plt.close(fig)

# Most influential tokens (shortcut inspection)
vocab = np.array(model.named_steps["tfidf"].get_feature_names_out())
coef = model.named_steps["clf"].coef_[0]
pd.DataFrame({
    "top_malicious_tokens": vocab[np.argsort(coef)[::-1][:25]],
    "top_legit_tokens": vocab[np.argsort(coef)[:25]],
}).to_csv(C.TAB / "baseline_top_tokens.csv", index=False)
print("malicious:", ", ".join(vocab[np.argsort(coef)[::-1][:25]]))
print("legit    :", ", ".join(vocab[np.argsort(coef)[:25]]))

# ------------------------------------------------------------- 3. leave-one-source-out
loso = {}
for src, a, b in loso_folds(df):
    m = make_model(best_C).fit(df[TEXT].iloc[a], df.label.iloc[a])
    pp = m.predict_proba(df[TEXT].iloc[b])[:, 1]
    loso[src] = full_report(df.label.iloc[b], (pp >= 0.5).astype(int), pp)
    print("LOSO", src, loso[src])
pd.DataFrame(loso).T.to_csv(C.TAB / "baseline_loso.csv")

# ------------------------------------------------------------- 4. how much do near-duplicates inflate scores?
ex = pd.read_parquet(C.PROCESSED / "corpus_exact_dedup.parquet")
infl = {}
for name, grouped in [("random split (near-duplicates may cross)", False), ("grouped split (clusters kept together)", True)]:
    m_te = holdout_split(ex, grouped=grouped)
    m = make_model(best_C).fit(ex[TEXT][~m_te], ex.label[~m_te])
    pp = m.predict_proba(ex[TEXT][m_te])[:, 1]
    r = full_report(ex.label[m_te], (pp >= 0.5).astype(int), pp)
    cross = len(set(ex.cluster[~m_te]) & set(ex.cluster[m_te]))
    infl[name] = {**r, "clusters_crossing_split": cross}
    print(name, infl[name])
pd.DataFrame(infl).T.to_csv(C.TAB / "dedup_inflation.csv")

json.dump({"best_C": best_C, "vectorizer": {k: str(v) for k, v in VEC.items()}, "results": results, "loso": loso, "inflation": infl},
          open(C.TAB / "baseline_results.json", "w"), indent=1)
