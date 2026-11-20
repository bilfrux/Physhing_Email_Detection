"""Feature construction blocks shared by the AutoML search and the final refit
(the exact same objects are used in both, so what is evaluated is what is shipped)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from . import config as C


class EmailStats(BaseEstimator, TransformerMixin):
    """Hand-crafted, source-agnostic statistics.

    Deliberately EXCLUDED: upper-case ratio and punctuation spacing - the audit
    showed they identify Enron/Ling (lower-cased, tokenised) rather than intent.
    All features are non-negative (required by chi2 selection and ComplementNB).
    Input: DataFrame with `text_mask` and `raw_text`.
    """

    names = ["log_chars", "log_words", "n_url", "n_email", "num_per_word", "excl_per_kchar", "money_per_kchar", "urgent_words"]
    _URGENT = r"\b(urgent|immediately|verify|suspend|suspended|password|account|confirm|click|expire|expires|limited)\b"

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        m, r = X["text_mask"], X["raw_text"]
        words = m.str.count(" ") + 1
        chars = r.str.len().clip(lower=1)
        out = np.c_[
            np.log1p(chars),
            np.log1p(words),
            m.str.count(r"\burltoken\b"),
            m.str.count(r"\bemailtoken\b"),
            m.str.count(r"\bnumtoken\b") / words,
            1000 * r.str.count("!") / chars,
            1000 * r.str.count(r"[$€£]") / chars,
            m.str.count(self._URGENT),
        ]
        return out.astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.names)


def build_features(p: dict) -> ColumnTransformer:
    blocks = [
        (
            "word",
            TfidfVectorizer(ngram_range=(1, p["ngram_max"]), min_df=p["min_df"], max_df=0.95, sublinear_tf=True, max_features=p["max_features"], dtype=np.float32),
            p["text"],
        )
    ]
    if p["use_char"]:
        blocks.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, sublinear_tf=True, max_features=50_000, dtype=np.float32), p["text"]))
    if p["use_stats"]:
        blocks.append(("stats", Pipeline([("ext", EmailStats()), ("scale", MaxAbsScaler())]), ["text_mask", "raw_text"]))
    return ColumnTransformer(blocks, sparse_threshold=1.0)


def build_model(p: dict):
    m = p["model"]
    if m == "logreg":
        return LogisticRegression(C=p["C"], solver="liblinear", max_iter=3000)
    if m == "linear_svm":
        return LinearSVC(C=p["C"], max_iter=5000)
    if m == "complement_nb":
        return ComplementNB(alpha=p["alpha"])
    if m == "sgd_huber":
        return SGDClassifier(loss="modified_huber", alpha=p["alpha"], max_iter=50, tol=1e-4, random_state=C.SEED)
    if m == "ridge_lsq":  # least-squares linear regression on {-1,+1} targets, L2-penalised
        return RidgeClassifier(alpha=p["alpha"], solver="sparse_cg")
    if m == "decision_tree":
        return DecisionTreeClassifier(max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"], random_state=C.SEED)
    raise ValueError(m)


def build_head(p: dict) -> Pipeline:
    """Optional chi2 feature selection (keep the top `select_frac` of features) + classifier."""
    steps = []
    if p["select_frac"] < 1.0:
        steps.append(("select", SelectPercentile(chi2, percentile=100 * p["select_frac"])))
    steps.append(("model", build_model(p)))
    return Pipeline(steps)


def build_full_pipeline(p: dict) -> Pipeline:
    """Feature extraction + selection + model as ONE pipeline (fit on training data only)."""
    return Pipeline([("features", build_features(p))] + build_head(p).steps)


def scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)
