"""Can we guess the source corpus from the text?

If yes, the spam classifier can learn "which corpus is this" (Ling is 16% malicious,
Nazario 100%) instead of "is this a scam". Training split only, 5-fold CV.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from phish import config as cfg
from phish.data import load_split
from phish.split import cv_folds

_, train, _ = load_split()
folds = cv_folds(train, n_splits=5)
y = train["source"].values


def style_stats(s):
    n = s.str.len().clip(lower=1)
    return np.c_[np.log10(n), s.str.count(r"[A-Z]") / n, s.str.count(r" [.,:;!?] ") / n,
                 s.str.count(r"\d") / n, s.str.count(r"\n") / n]


def cv_predict(make_model, X):
    pred = np.empty(len(train), dtype=object)
    for a, b in folds:
        pred[b] = make_model().fit(X[a], y[a]).predict(X[b])
    return pred


def tfidf():
    return TfidfVectorizer(max_features=50000, sublinear_tf=True, min_df=3)


def raw_tfidf():  # keeps case and punctuation
    return TfidfVectorizer(lowercase=False, token_pattern=r"(?u)\b\w+\b|[^\w\s]", max_features=50000, sublinear_tf=True, min_df=3)


inputs = [
    ("raw text (case + punctuation kept)", train["raw_text"].to_numpy(), raw_tfidf),
    ("minimal cleaning (lower-case, no punctuation)", train["text_min"].to_numpy(), tfidf),
    ("masked (URL/e-mail/digits -> tokens)", train["text_mask"].to_numpy(), tfidf),
]
rows = []
for name, X, vec in inputs:
    pred = cv_predict(lambda: make_pipeline(vec(), LinearSVC(C=0.5, random_state=cfg.SEED)), X)
    rows.append({"input": name, "source_accuracy": accuracy_score(y, pred), "source_macro_f1": f1_score(y, pred, average="macro")})
    print(rows[-1])

pred = cv_predict(lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=cfg.SEED)),
                  style_stats(train["raw_text"]))
rows.append({"input": "5 style statistics only (length, case, spacing, digits, newlines)",
             "source_accuracy": accuracy_score(y, pred), "source_macro_f1": f1_score(y, pred, average="macro")})
print(rows[-1])
rows.append({"input": "chance (majority source)", "source_accuracy": pd.Series(y).value_counts(normalize=True).iloc[0],
             "source_macro_f1": np.nan})
res = pd.DataFrame(rows).round(4)
res.to_csv(cfg.TAB / "shortcut_source_prediction.csv", index=False)
print(res.to_string())

# label accuracy if we only know the source: predict the majority label of each source
majority = train.groupby("source").label.agg(lambda s: int(s.mean() >= 0.5))
guess = train.source.map(majority)
src_only = {
    "accuracy_label_from_source_only": round(accuracy_score(train.label, guess), 4),
    "f1_label_from_source_only": round(f1_score(train.label, guess), 4),
}
print(src_only)
pd.Series(src_only).to_csv(cfg.TAB / "shortcut_label_from_source.csv", header=["value"])

# which words give the source away after masking?
vec = tfidf()
X = vec.fit_transform(train["text_mask"])
clf = LinearSVC(C=0.5, random_state=cfg.SEED).fit(X, y)
vocab = np.array(vec.get_feature_names_out())
top = {src: ", ".join(vocab[np.argsort(clf.coef_[i])[::-1][:12]]) for i, src in enumerate(clf.classes_)}
pd.Series(top).to_csv(cfg.TAB / "shortcut_top_tokens_per_source.csv", header=["top tokens identifying the source (masked text)"])
for k, v in top.items():
    print(f"{k:15s} {v}")
