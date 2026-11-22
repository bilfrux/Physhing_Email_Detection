"""Three checks on choices that could change our conclusions.

1. In 04, C is chosen on a CV that contains all six sources, then reused in
   leave-one-source-out. Here we choose C again in each fold without the
   held-out source and compare.
2. clean_text() drops non-ASCII characters. Is that why Nazario recall is low?
3. When a near-duplicate cluster spans several sources, we keep the copy of
   the first source in the list. Which sources lose mails to which?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

from phish import config as cfg
from phish.features import BASELINE_TEXT as TEXT, BASELINE_TFIDF, baseline_lr, baseline_model
from phish.metrics import full_report
from phish.split import cv_folds, loso_folds

baseline = json.load(open(cfg.TAB / "baseline_results.json"))
GLOBAL_C = baseline["best_C"]
C_GRID = [3, 10, 30, 100, 300, 1000]  # part of 04's grid where val F1 is within 0.003 of the best


def choose_C(d):
    """Same rule as 04 (one standard error), with 3 folds."""
    folds = cv_folds(d, n_splits=3)
    f1s = {c: [] for c in C_GRID}
    for a, b in folds:
        vec = TfidfVectorizer(**BASELINE_TFIDF)
        Xa, Xb = vec.fit_transform(d[TEXT].iloc[a]), vec.transform(d[TEXT].iloc[b])
        for c in C_GRID:
            f1s[c].append(f1_score(d.label.iloc[b], baseline_lr(c).fit(Xa, d.label.iloc[a]).predict(Xb)))
    mean = {c: np.mean(v) for c, v in f1s.items()}
    best = max(mean, key=mean.get)
    se = np.std(f1s[best]) / np.sqrt(len(folds))
    return float(min(c for c in C_GRID if mean[c] >= mean[best] - se))


df = pd.read_parquet(cfg.PROCESSED / "corpus.parquet")

# --- 1. C chosen inside each LOSO fold
rows, nazario_scores = [], None
for src, a, b in loso_folds(df):
    train, test = df.iloc[a], df.iloc[b]
    c = choose_C(train)
    p = baseline_model(c).fit(train[TEXT], train.label).predict_proba(test[TEXT])[:, 1]
    r = full_report(test.label, (p >= 0.5).astype(int), p)
    g = baseline["loso"][src]
    rows.append({"held_out_source": src, "C_global": GLOBAL_C, "C_nested": c,
                 "recall_global": g["recall"], "recall_nested": r["recall"],
                 "f1_global": g.get("f1"), "f1_nested": r.get("f1"), "fpr_global": g.get("fpr"), "fpr_nested": r.get("fpr")})
    print(rows[-1], flush=True)
    if src == "Nazario" and c == GLOBAL_C:
        nazario_scores = p
pd.DataFrame(rows).round(4).to_csv(cfg.TAB / "check_loso_nested_C.csv", index=False)

# --- 2. Nazario recall, mostly non-ASCII mails vs the rest
naz = df[df.source == "Nazario"]
if nazario_scores is None:  # C was different, refit with the C used in the report
    rest = df[df.source != "Nazario"]
    nazario_scores = baseline_model(GLOBAL_C).fit(rest[TEXT], rest.label).predict_proba(naz[TEXT])[:, 1]
share = naz.raw_text.str.count(r"[^\x00-\x7f]") / naz.raw_text.str.len().clip(lower=1)
group = np.where(share > 0.3, "mostly non-ASCII (>30% of characters)", "mostly ASCII")
nonascii = (pd.DataFrame({"group": group, "detected": nazario_scores >= 0.5, "words_left": naz[TEXT].str.split().str.len()})
            .groupby("group")
            .agg(n=("detected", "size"), recall=("detected", "mean"), median_words_after_cleaning=("words_left", "median"))
            .round(4))
nonascii.to_csv(cfg.TAB / "check_nazario_nonascii.csv")
print(nonascii)

# --- 3. cross-source near-duplicates
ex = pd.read_parquet(cfg.PROCESSED / "corpus_exact_dedup.parquet", columns=["source", "cluster"])
n_src = ex.groupby("cluster").source.nunique()
cross = ex[ex.cluster.isin(n_src[n_src > 1].index)]
kept_in = df.set_index("cluster").source  # source of the copy we kept
cross = cross.assign(kept_in=cross.cluster.map(kept_in))
lost = cross[cross.source != cross.kept_in]
tab = pd.crosstab(lost.source, lost.kept_in).reindex(index=cfg.SOURCES, columns=cfg.SOURCES, fill_value=0)
tab["total_lost"] = tab.sum(axis=1)
tab["pct_of_source_after_exact_dedup"] = (100 * tab.total_lost / ex.source.value_counts().reindex(cfg.SOURCES)).round(2)
tab.to_csv(cfg.TAB / "check_cross_source_dedup.csv")
print(f"{(n_src > 1).sum()} cross-source clusters; e-mails moved to another source:\n", tab)
