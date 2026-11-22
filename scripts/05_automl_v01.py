"""AutoML v0.1 with Optuna (TPE + median pruning).

Searches the cleaning (minimal / masked), the features (word n-grams, char n-grams,
hand-made stats, chi2 selection), the model and its hyper-parameters. Objective:
mean F1 over 3-fold CV on a 20k subsample of the training split. Then the best
config of each model is refitted on the full train split and tested on the
hold-out and in leave-one-source-out.

Limit: the objective is in-distribution, every source is seen during the search.
v0.2 will select with nested leave-one-source-out.

Usage: python scripts/05_automl_v01.py [n_trials]
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
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

from phish import config as cfg
from phish.data import load_split
from phish.features import build_features, build_full_pipeline, build_head, new_study, scores, search_sample, suggest
from phish.metrics import full_report
from phish.split import cv_folds, loso_folds

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
TIMEOUT = 25 * 60  # run 1 had no limit and got stuck on slow SVM trials
optuna.logging.set_verbosity(optuna.logging.WARNING)

df, train, test = load_split()
sub = search_sample(train)
folds = cv_folds(sub, n_splits=3)
print(f"search on {len(sub)} training e-mails, {len(folds)} grouped folds, {N_TRIALS} trials")

# the TF-IDF matrices are slow to build, keep the last few (char n-grams use a lot of RAM)
cache = OrderedDict()
FEATURE_KEYS = ["text", "ngram_max", "min_df", "max_features", "use_char", "use_stats"]


def fold_features(p, k):
    key = (k,) + tuple(p[x] for x in FEATURE_KEYS)
    if key not in cache:
        a, b = folds[k]
        ct = build_features(p)
        cache[key] = (ct.fit_transform(sub.iloc[a]), ct.transform(sub.iloc[b]))
        if len(cache) > 9:
            cache.popitem(last=False)
    return cache[key]


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


start = time.time()


def log_trial(study, t):
    value = None if t.value is None else round(t.value, 4)
    best = round(study.best_value, 4) if study.best_trial else None
    print(f"trial {t.number:3d} {t.state.name:8s} value={value} model={t.params.get('model')} best={best} t={time.time() - start:.0f}s", flush=True)


study = new_study()
study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT, show_progress_bar=False, callbacks=[log_trial])
print(f"search time {time.time() - start:.0f}s")

trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
trials.to_csv(cfg.TAB / "automl_trials.csv", index=False)
done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
print("completed", len(done), "pruned", sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials))

best_per_model = {}
for t in done:
    p = t.user_attrs["params_full"]
    m = p["model"]
    if m not in best_per_model or t.value > best_per_model[m]["cv_f1"]:
        best_per_model[m] = {"cv_f1": t.value, "cv_f1_std": t.user_attrs["val_f1_std"],
                             "train_f1": t.user_attrs["train_f1"], "params": p, "trial": t.number}
json.dump(best_per_model, open(cfg.TAB / "automl_best_per_family.json", "w"), indent=1, default=str)
best_model = max(best_per_model, key=lambda m: best_per_model[m]["cv_f1"])
print("best family:", best_model, best_per_model[best_model])

# --- refit the best config of each model on the full train split (2 in parallel)
cache.clear()


def evaluate(model_name, p):
    t1 = time.time()
    pipe = build_full_pipeline(p).fit(train, train.label)
    s = scores(pipe, test)
    r = full_report(test.label, pipe.predict(test), s)
    info = best_per_model[model_name]
    row = {"family": model_name, "cv_f1": round(info["cv_f1"], 4), "search_train_f1": round(info["train_f1"], 4),
           **{f"test_{k}": v for k, v in r.items() if k not in ("n", "pos_rate")}}
    loso_rows = []
    for src, a, b in loso_folds(df):
        m = build_full_pipeline(p).fit(df.iloc[a], df.label.iloc[a])
        loso_rows.append({"family": model_name, "held_out": src, **full_report(df.label.iloc[b], m.predict(df.iloc[b]), scores(m, df.iloc[b]))})
    row["eval_s"] = round(time.time() - t1)
    return row, loso_rows, (s if model_name == best_model else None)


order = sorted(best_per_model, key=lambda m: -best_per_model[m]["cv_f1"])
out = Parallel(n_jobs=2, verbose=5)(delayed(evaluate)(m, best_per_model[m]["params"]) for m in order)
rows, loso_rows = [], []
for m, (row, lrows, s) in zip(order, out):
    rows.append(row)
    loso_rows += lrows
    if s is not None:
        pd.DataFrame({"source": test.source, "label": test.label, "score": s}).to_parquet(cfg.PROCESSED / "pred_automl_test.parquet")
    print(m, row, flush=True)

pd.DataFrame(rows).to_csv(cfg.TAB / "automl_family_holdout.csv", index=False)
loso = pd.DataFrame(loso_rows)
loso.to_csv(cfg.TAB / "automl_family_loso.csv", index=False)
print(pd.DataFrame(rows).to_string())
print(loso.pivot(index="family", columns="held_out", values="f1").round(4).to_string())
print(loso.pivot(index="family", columns="held_out", values="recall").round(4).to_string())
json.dump({"best_family": best_model, "n_trials": N_TRIALS, "subsample": len(sub)}, open(cfg.TAB / "automl_summary.json", "w"), indent=1)
