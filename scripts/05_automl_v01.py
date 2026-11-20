"""Step 2.5 - AutoML pipeline v0.1.

Searches jointly over
  data preparation    : minimal cleaning vs masking (URL / e-mail / digits)
  feature engineering : word n-grams, char n-grams, hand-crafted stats (scaled),
                        chi2 feature selection
  model selection     : logistic regression, linear SVM, complement naive Bayes,
                        SGD (modified Huber), ridge least-squares, decision tree
  hyper-parameters    : per-model (C, alpha, depth, ...)
with Optuna's TPE sampler + median pruning.

Search objective = mean F1 over 3-fold grouped CV on a stratified subsample of
the TRAINING split (the test split is never used for selection).
Afterwards the best configuration of each model family is refitted on the full
training split, then evaluated (a) on the hold-out test and (b) in
leave-one-source-out (LOSO).

Known limitation (documented in the report): the search objective is
in-distribution, so the selected configuration has "seen" every source;
v0.2 will use nested LOSO as the selection criterion.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import time
from collections import OrderedDict

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score

from phish import config as C
from phish.features import build_features, build_full_pipeline, build_head, scores
from phish.metrics import full_report
from phish.split import cv_folds, loso_folds

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
SUBSAMPLE = 20_000
SEARCH_TIMEOUT = 25 * 60  # seconds; run 1 had an unbounded tail (non-converging linear SVM at large C)
optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_parquet(C.PROCESSED / "corpus.parquet")
split = pd.read_csv(C.PROCESSED / "split.csv")
tr = df[split.split.values == "train"].reset_index(drop=True)
te = df[split.split.values == "test"].reset_index(drop=True)

# Stratified (source x label) subsample of the training split for the search.
rng = np.random.default_rng(C.SEED)
frac = SUBSAMPLE / len(tr)
sub = tr.groupby(["source", "label"]).sample(frac=frac, random_state=C.SEED).reset_index(drop=True)
folds = cv_folds(sub, n_splits=3)
print(f"search on {len(sub)} training e-mails, {len(folds)} grouped folds, {N_TRIALS} trials")

_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
FEAT_KEYS = ["text", "ngram_max", "min_df", "max_features", "use_char", "use_stats"]


def fold_features(p, k):
    key = (k,) + tuple(p[x] for x in FEAT_KEYS)
    if key not in _cache:
        a, b = folds[k]
        ct = build_features(p)
        _cache[key] = (ct.fit_transform(sub.iloc[a]), ct.transform(sub.iloc[b]))  # fit on fold-train only
        if len(_cache) > 9:  # ~3 configs x 3 folds; char n-gram matrices are large (7 GB RAM box)
            _cache.popitem(last=False)
    return _cache[key]


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
    elif m == "linear_svm":  # capped: liblinear did not converge for C > ~100 (run 1)
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


def objective(trial):
    p = suggest(trial)
    val, trn = [], []
    for k, (a, b) in enumerate(folds):
        Xa, Xb = fold_features(p, k)
        head = build_head(p).fit(Xa, sub.label.iloc[a])
        val.append(f1_score(sub.label.iloc[b], head.predict(Xb)))
        trn.append(f1_score(sub.label.iloc[a], head.predict(Xa)))
        trial.report(float(np.mean(val)), k)
        if trial.should_prune():
            raise optuna.TrialPruned()
    trial.set_user_attr("train_f1", float(np.mean(trn)))
    trial.set_user_attr("val_f1_std", float(np.std(val)))
    trial.set_user_attr("params_full", p)
    return float(np.mean(val))


# Seed the search with one default config per family so every family is evaluated at least once.
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.SEED), pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
base = {"text": "text_mask", "ngram_max": 1, "min_df": 5, "max_features": 50_000, "use_char": False, "use_stats": False, "select_frac": 1.0}
for fam, hp in [("logreg", {"C": 10.0}), ("linear_svm", {"C_svm": 1.0}), ("complement_nb", {"alpha_nb": 0.1}), ("sgd_huber", {"alpha_sgd": 1e-5}), ("ridge_lsq", {"alpha_ridge": 1.0}), ("decision_tree", {"max_depth": 30, "min_samples_leaf": 5})]:
    study.enqueue_trial({**base, "model": fam, **hp})
t0 = time.time()
def _log(study, t):
    print(f"trial {t.number:3d} {t.state.name:8s} value={t.value if t.value is None else round(t.value, 4)} "
          f"model={t.params.get('model')} best={round(study.best_value, 4) if study.best_trial else None} t={time.time() - t0:.0f}s", flush=True)


study.optimize(objective, n_trials=N_TRIALS, timeout=SEARCH_TIMEOUT, show_progress_bar=False, callbacks=[_log])
print(f"search time {time.time() - t0:.0f}s")

trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
trials.to_csv(C.TAB / "automl_trials.csv", index=False)
done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
print("completed", len(done), "pruned", sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials))

best_per_family = {}
for t in done:
    p = t.user_attrs["params_full"]
    f = p["model"]
    if f not in best_per_family or t.value > best_per_family[f]["cv_f1"]:
        best_per_family[f] = {"cv_f1": t.value, "cv_f1_std": t.user_attrs["val_f1_std"], "train_f1": t.user_attrs["train_f1"], "params": p, "trial": t.number}
json.dump(best_per_family, open(C.TAB / "automl_best_per_family.json", "w"), indent=1, default=str)
overall = max(best_per_family, key=lambda f: best_per_family[f]["cv_f1"])
print("best family:", overall, best_per_family[overall])

# Refit best config per family on the full training split -> hold-out test + LOSO.
# Families are evaluated in parallel (2 cores); every fit re-learns its features on its own training data.
from joblib import Parallel, delayed

_cache.clear()


def eval_family(fam, p):
    t1 = time.time()
    pipe = build_full_pipeline(p).fit(tr, tr.label)
    s = scores(pipe, te)
    r = full_report(te.label, pipe.predict(te), s)
    info = best_per_family[fam]
    row = {"family": fam, "cv_f1": round(info["cv_f1"], 4), "search_train_f1": round(info["train_f1"], 4), **{f"test_{k}": v for k, v in r.items() if k not in ("n", "pos_rate")}}
    lrows = []
    for src, a, b in loso_folds(df):
        m = build_full_pipeline(p).fit(df.iloc[a], df.label.iloc[a])
        rr = full_report(df.label.iloc[b], m.predict(df.iloc[b]), scores(m, df.iloc[b]))
        lrows.append({"family": fam, "held_out": src, **rr})
    row["eval_s"] = round(time.time() - t1)
    return row, lrows, (s if fam == overall else None)


fams = sorted(best_per_family, key=lambda f: -best_per_family[f]["cv_f1"])
out = Parallel(n_jobs=2, verbose=5)(delayed(eval_family)(f, best_per_family[f]["params"]) for f in fams)
rows, loso_rows = [], []
for fam, (row, lrows, s) in zip(fams, out):
    rows.append(row)
    loso_rows += lrows
    if s is not None:
        pd.DataFrame({"source": te.source, "label": te.label, "score": s}).to_parquet(C.PROCESSED / "pred_automl_test.parquet")
    print(fam, row, flush=True)

pd.DataFrame(rows).to_csv(C.TAB / "automl_family_holdout.csv", index=False)
lo = pd.DataFrame(loso_rows)
lo.to_csv(C.TAB / "automl_family_loso.csv", index=False)
print(pd.DataFrame(rows).to_string())
print(lo.pivot(index="family", columns="held_out", values="f1").round(4).to_string())
print(lo.pivot(index="family", columns="held_out", values="recall").round(4).to_string())
json.dump({"best_family": overall, "n_trials": N_TRIALS, "subsample": len(sub)}, open(C.TAB / "automl_summary.json", "w"), indent=1)
