"""Pipelines used by the baseline and by the AutoML search."""
import numpy as np
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

from . import config as cfg

# ---------------------------------------------------------------- baseline
BASELINE_TEXT = "text_min"
BASELINE_TFIDF = dict(ngram_range=(1, 1), min_df=3, max_df=0.95, sublinear_tf=True, max_features=100_000)


def baseline_lr(c):
    return LogisticRegression(C=c, solver="liblinear", max_iter=2000, random_state=cfg.SEED)


def baseline_model(c):
    return Pipeline([("tfidf", TfidfVectorizer(**BASELINE_TFIDF)), ("clf", baseline_lr(c))])


# ---------------------------------------------------------------- AutoML
class EmailStats(BaseEstimator, TransformerMixin):
    """A few hand-made counts. No upper-case ratio or punctuation spacing: those
    only tell Enron/Ling apart from the rest. All values are >= 0 (needed by chi2
    and ComplementNB)."""

    names = ["log_chars", "log_words", "n_url", "n_email", "num_per_word", "excl_per_kchar", "money_per_kchar", "urgent_words"]
    URGENT = r"\b(urgent|immediately|verify|suspend|suspended|password|account|confirm|click|expire|expires|limited)\b"

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        masked, raw = X["text_mask"], X["raw_text"]
        words = masked.str.count(" ") + 1
        chars = raw.str.len().clip(lower=1)
        return np.c_[
            np.log1p(chars),
            np.log1p(words),
            masked.str.count(r"\burltoken\b"),
            masked.str.count(r"\bemailtoken\b"),
            masked.str.count(r"\bnumtoken\b") / words,
            1000 * raw.str.count("!") / chars,
            1000 * raw.str.count(r"[$€£]") / chars,
            masked.str.count(self.URGENT),
        ].astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.names)


def build_features(p):
    word = TfidfVectorizer(ngram_range=(1, p["ngram_max"]), min_df=p["min_df"], max_df=0.95, sublinear_tf=True,
                           max_features=p["max_features"], dtype=np.float32)
    blocks = [("word", word, p["text"])]
    if p["use_char"]:
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, sublinear_tf=True,
                               max_features=50_000, dtype=np.float32)
        blocks.append(("char", char, p["text"]))
    if p["use_stats"]:
        blocks.append(("stats", Pipeline([("ext", EmailStats()), ("scale", MaxAbsScaler())]), ["text_mask", "raw_text"]))
    return ColumnTransformer(blocks, sparse_threshold=1.0)


def build_model(p):
    m = p["model"]
    if m == "logreg":
        return LogisticRegression(C=p["C"], solver="liblinear", max_iter=3000, random_state=cfg.SEED)
    if m == "linear_svm":
        return LinearSVC(C=p["C"], max_iter=5000, random_state=cfg.SEED)
    if m == "complement_nb":
        return ComplementNB(alpha=p["alpha"])
    if m == "sgd_huber":
        return SGDClassifier(loss="modified_huber", alpha=p["alpha"], max_iter=50, tol=1e-4, random_state=cfg.SEED)
    if m == "ridge_lsq":  # least squares on -1/+1 targets with L2
        return RidgeClassifier(alpha=p["alpha"], solver="sparse_cg")
    if m == "decision_tree":
        return DecisionTreeClassifier(max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"], random_state=cfg.SEED)
    raise ValueError(m)


def build_head(p):
    """Optional chi2 selection + the model (everything after the features)."""
    steps = []
    if p["select_frac"] < 1.0:
        steps.append(("select", SelectPercentile(chi2, percentile=100 * p["select_frac"])))
    steps.append(("model", build_model(p)))
    return Pipeline(steps)


def build_full_pipeline(p):
    return Pipeline([("features", build_features(p))] + build_head(p).steps)


def scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


# ---------------------------------------------------------------- search space (used by 05 and 05b)
def suggest(trial):
    p = {
        "text": trial.suggest_categorical("text", ["text_min", "text_mask"]),
        "ngram_max": trial.suggest_int("ngram_max", 1, 2),
        "min_df": trial.suggest_categorical("min_df", [2, 5, 10]),
        "max_features": trial.suggest_categorical("max_features", [20_000, 50_000, 100_000]),
        "use_char": trial.suggest_categorical("use_char", [False, True]),
        "use_stats": trial.suggest_categorical("use_stats", [False, True]),
        "select_frac": trial.suggest_categorical("select_frac", [1.0, 0.5, 0.2]),
        "model": trial.suggest_categorical("model", ["logreg", "linear_svm", "complement_nb", "sgd_huber", "ridge_lsq", "decision_tree"]),
    }
    m = p["model"]
    if m == "logreg":
        p["C"] = trial.suggest_float("C", 1e-2, 1e3, log=True)
    elif m == "linear_svm":  # above C=100 liblinear did not converge (run 1)
        p["C"] = trial.suggest_float("C_svm", 1e-2, 1e2, log=True)
    elif m == "complement_nb":
        p["alpha"] = trial.suggest_float("alpha_nb", 1e-3, 1.0, log=True)
    elif m == "sgd_huber":
        p["alpha"] = trial.suggest_float("alpha_sgd", 1e-7, 1e-3, log=True)
    elif m == "ridge_lsq":
        p["alpha"] = trial.suggest_float("alpha_ridge", 1e-2, 1e2, log=True)
    else:
        p["max_depth"] = trial.suggest_int("max_depth", 5, 80, log=True)
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 50, log=True)
    return p


def new_study():
    """Same study as the logged run: TPE seed 42, one default trial per model first."""
    import optuna

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    base = {"text": "text_mask", "ngram_max": 1, "min_df": 5, "max_features": 50_000,
            "use_char": False, "use_stats": False, "select_frac": 1.0}
    defaults = [("logreg", {"C": 10.0}), ("linear_svm", {"C_svm": 1.0}), ("complement_nb", {"alpha_nb": 0.1}),
                ("sgd_huber", {"alpha_sgd": 1e-5}), ("ridge_lsq", {"alpha_ridge": 1.0}),
                ("decision_tree", {"max_depth": 30, "min_samples_leaf": 5})]
    for model, hp in defaults:
        study.enqueue_trial({**base, "model": model, **hp})
    return study


def search_sample(train):
    """The 20k stratified training subsample used by the search."""
    frac = 20_000 / len(train)
    return train.groupby(["source", "label"]).sample(frac=frac, random_state=cfg.SEED).reset_index(drop=True)
