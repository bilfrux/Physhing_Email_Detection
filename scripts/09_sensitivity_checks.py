"""Sensitivity checks on three choices that could bias the week-2 conclusions.

1. Selection leakage in LOSO: 04_baseline.py chooses C by CV on the training split,
   which contains all six sources, then reuses it in every LOSO fold. Here C is
   re-chosen for each fold WITHOUT the held-out source (3-fold grouped CV, same
   one-standard-error rule) and the LOSO scores are compared.
2. Non-ASCII text: clean_text() drops every non-ASCII character. Does that explain
   part of the low Nazario recall? Recall on mostly non-ASCII vs other Nazario mails.
3. Cross-source near-duplicates: one representative per cluster is kept, always in
   the source listed first in config.SOURCES. Which sources lose e-mails to which?

Output: reports/tables/check_loso_nested_C.csv, check_nazario_nonascii.csv, check_cross_source_dedup.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from phish import config as C
from phish.metrics import full_report
from phish.split import cv_folds, loso_folds

TEXT = "text_min"
B = json.load(open(C.TAB / "baseline_results.json"))
VEC = dict(ngram_range=(1, 1), min_df=3, max_df=0.95, sublinear_tf=True, max_features=100_000)  # same as 04_baseline.py
C_GRID = [3, 10, 30, 100, 300, 1000]  # 04's grid restricted to where its validation F1 is within 0.003 of the best
GLOBAL_C = B["best_C"]


def lr(c):
    return LogisticRegression(C=c, solver="liblinear", max_iter=2000, random_state=C.SEED)


def choose_C(d):
    """Same protocol as 04_baseline.py (grouped CV, one-standard-error rule), 3 folds."""
    folds = cv_folds(d, n_splits=3)
    scores = {c: [] for c in C_GRID}
    for a, b in folds:
        vec = TfidfVectorizer(**VEC)
        Xa, Xb = vec.fit_transform(d[TEXT].iloc[a]), vec.transform(d[TEXT].iloc[b])
        for c in C_GRID:
            scores[c].append(f1_score(d.label.iloc[b], lr(c).fit(Xa, d.label.iloc[a]).predict(Xb)))
    mean = {c: np.mean(v) for c, v in scores.items()}
    best = max(mean, key=mean.get)
    se = np.std(scores[best]) / np.sqrt(len(folds))
    return float(min(c for c in C_GRID if mean[c] >= mean[best] - se))


df = pd.read_parquet(C.PROCESSED / "corpus.parquet")

# ------------------------------------------------------------------ 1. nested C in LOSO
rows, nazario_scores = [], None
for src, a, b in loso_folds(df):
    tr, te = df.iloc[a], df.iloc[b]
    c_nested = choose_C(tr)
    m = Pipeline([("tfidf", TfidfVectorizer(**VEC)), ("clf", lr(c_nested))]).fit(tr[TEXT], tr.label)
    p = m.predict_proba(te[TEXT])[:, 1]
    r = full_report(te.label, (p >= 0.5).astype(int), p)
    g = B["loso"][src]
    rows.append({"held_out_source": src, "C_global": GLOBAL_C, "C_nested": c_nested,
                 "recall_global": g["recall"], "recall_nested": r["recall"],
                 "f1_global": g.get("f1"), "f1_nested": r.get("f1"), "fpr_global": g.get("fpr"), "fpr_nested": r.get("fpr")})
    print(rows[-1], flush=True)
    if src == "Nazario" and c_nested == GLOBAL_C:
        nazario_scores = p
out = pd.DataFrame(rows).round(4)
out.to_csv(C.TAB / "check_loso_nested_C.csv", index=False)

# ------------------------------------------------------------------ 2. Nazario recall vs non-ASCII share
naz = df[df.source == "Nazario"]
if nazario_scores is None:  # nested C differed: refit with the global C used in the report
    rest = df[df.source != "Nazario"]
    m = Pipeline([("tfidf", TfidfVectorizer(**VEC)), ("clf", lr(GLOBAL_C))]).fit(rest[TEXT], rest.label)
    nazario_scores = m.predict_proba(naz[TEXT])[:, 1]
share = naz.raw_text.str.count(r"[^\x00-\x7f]") / naz.raw_text.str.len().clip(lower=1)
grp = np.where(share > 0.3, "mostly non-ASCII (>30% of characters)", "mostly ASCII")
na = (pd.DataFrame({"group": grp, "detected": nazario_scores >= 0.5, "words_left": naz[TEXT].str.split().str.len()})
      .groupby("group").agg(n=("detected", "size"), recall=("detected", "mean"), median_words_after_cleaning=("words_left", "median")).round(4))
na.to_csv(C.TAB / "check_nazario_nonascii.csv")
print(na)

# ------------------------------------------------------------------ 3. cross-source near-duplicates
ex = pd.read_parquet(C.PROCESSED / "corpus_exact_dedup.parquet", columns=["source", "cluster"])
multi = ex.groupby("cluster").source.nunique()
cross = ex[ex.cluster.isin(multi[multi > 1].index)]
kept = df.set_index("cluster").source  # the representative's source
cross = cross.assign(kept_in=cross.cluster.map(kept))
lost = cross[cross.source != cross.kept_in]
tab = pd.crosstab(lost.source, lost.kept_in).reindex(index=C.SOURCES, columns=C.SOURCES, fill_value=0)
tab["total_lost"] = tab.sum(axis=1)
tab["pct_of_source_after_exact_dedup"] = (100 * tab.total_lost / ex.source.value_counts().reindex(C.SOURCES)).round(2)
tab.to_csv(C.TAB / "check_cross_source_dedup.csv")
print(f"{len(multi[multi > 1])} cross-source clusters; e-mails moved to another source:\n", tab)
