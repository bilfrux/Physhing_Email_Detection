"""Step 2.1/2.2 - "shortcut test": can the SOURCE corpus be predicted from the text?

If yes, a malicious/legit classifier can partly learn "which corpus is this
from" (whose class mix differs: Ling 16% malicious, Nazario 100%) instead of
"is this a scam". We measure source predictability for each text variant,
and the accuracy of a model that knows ONLY the source (no text at all).
Training split only, 5-fold grouped CV.
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

from phish import config as C
from phish.split import cv_folds

df = pd.read_parquet(C.PROCESSED / "corpus.parquet")
split = pd.read_csv(C.PROCESSED / "split.csv")
tr = df[split.split.values == "train"].reset_index(drop=True)
folds = cv_folds(tr, n_splits=5)
y = tr["source"].values


def style_feats(s: pd.Series) -> np.ndarray:
    L = s.str.len().clip(lower=1)
    return np.c_[
        np.log10(L),
        s.str.count(r"[A-Z]") / L,
        s.str.count(r" [.,:;!?] ") / L,
        s.str.count(r"\d") / L,
        s.str.count(r"\n") / L,
    ]


variants = {
    "raw text (case + punctuation kept)": (
        tr["raw_text"],
        lambda: TfidfVectorizer(lowercase=False, token_pattern=r"(?u)\b\w+\b|[^\w\s]", max_features=50000, sublinear_tf=True, min_df=3),
    ),
    "minimal cleaning (lower-case, no punctuation)": (tr["text_min"], lambda: TfidfVectorizer(max_features=50000, sublinear_tf=True, min_df=3)),
    "masked (URL/e-mail/digits -> tokens)": (tr["text_mask"], lambda: TfidfVectorizer(max_features=50000, sublinear_tf=True, min_df=3)),
}

rows = []
for name, (X, vec) in variants.items():
    pred = np.empty(len(tr), dtype=object)
    for f_tr, f_te in folds:
        m = make_pipeline(vec(), LinearSVC(C=0.5))
        m.fit(X.iloc[f_tr], y[f_tr])
        pred[f_te] = m.predict(X.iloc[f_te])
    rows.append({"input": name, "source_accuracy": accuracy_score(y, pred), "source_macro_f1": f1_score(y, pred, average="macro")})
    print(rows[-1])

S = style_feats(tr["raw_text"])
pred = np.empty(len(tr), dtype=object)
for f_tr, f_te in folds:
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    m.fit(S[f_tr], y[f_tr])
    pred[f_te] = m.predict(S[f_te])
rows.append({"input": "5 style statistics only (length, case, spacing, digits, newlines)", "source_accuracy": accuracy_score(y, pred), "source_macro_f1": f1_score(y, pred, average="macro")})
print(rows[-1])
rows.append({"input": "chance (majority source)", "source_accuracy": pd.Series(y).value_counts(normalize=True).iloc[0], "source_macro_f1": np.nan})
res = pd.DataFrame(rows).round(4)
res.to_csv(C.TAB / "shortcut_source_prediction.csv", index=False)
print(res.to_string())

# Label accuracy reachable from the source identity alone (no text): majority label per source.
maj = tr.groupby("source").label.agg(lambda s: int(s.mean() >= 0.5))
p = tr.source.map(maj)
src_only = {
    "accuracy_label_from_source_only": round(accuracy_score(tr.label, p), 4),
    "f1_label_from_source_only": round(f1_score(tr.label, p), 4),
}
print(src_only)
pd.Series(src_only).to_csv(C.TAB / "shortcut_label_from_source.csv", header=["value"])

# Residual source markers in the masked text: top tokens of a one-vs-rest model.
vec = TfidfVectorizer(max_features=50000, sublinear_tf=True, min_df=3)
Xm = vec.fit_transform(tr["text_mask"])
clf = LinearSVC(C=0.5).fit(Xm, y)
vocab = np.array(vec.get_feature_names_out())
top = {cls: ", ".join(vocab[np.argsort(clf.coef_[i])[::-1][:12]]) for i, cls in enumerate(clf.classes_)}
pd.Series(top).to_csv(C.TAB / "shortcut_top_tokens_per_source.csv", header=["top tokens identifying the source (masked text)"])
for k, v in top.items():
    print(f"{k:15s} {v}")
