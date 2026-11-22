"""Recover the best AutoML v0.1 configuration from the search log of run 2
(the run was stopped during the search, before it saved its trials).

Optuna's TPE sampler is deterministic given its seed and the history of
(params, values), so we replay the logged history with ask/tell to recover the
exact parameters of every trial, then RE-EVALUATE the best ones with the same
3-fold grouped CV to prove the reconstruction is correct.

Outputs: reports/tables/automl_trials.csv, automl_best_config.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import re

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score

from phish import config as C
from phish.features import build_features, build_head
from phish.split import cv_folds

optuna.logging.set_verbosity(optuna.logging.WARNING)
LOG = C.TAB / "log_05_run2_stopped.txt"
pat = re.compile(r"trial\s+(\d+) (COMPLETE|PRUNED)\s+value=([\d.]+) model=(\w+)")
logged = [(int(n), s, float(v), m) for n, s, v, m in pat.findall(LOG.read_text())]
print(f"{len(logged)} trials in log")


def suggest(trial):  # identical to scripts/05_automl_v01.py (run 2)
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
    elif m == "linear_svm":
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


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.SEED),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
base = {"text": "text_mask", "ngram_max": 1, "min_df": 5, "max_features": 50_000, "use_char": False, "use_stats": False, "select_frac": 1.0}
for fam, hp in [("logreg", {"C": 10.0}), ("linear_svm", {"C_svm": 1.0}), ("complement_nb", {"alpha_nb": 0.1}), ("sgd_huber", {"alpha_sgd": 1e-5}), ("ridge_lsq", {"alpha_ridge": 1.0}), ("decision_tree", {"max_depth": 30, "min_samples_leaf": 5})]:
    study.enqueue_trial({**base, "model": fam, **hp})

rows, diverged = [], None
for n, state, value, model in logged:
    t = study.ask()
    p = suggest(t)
    if p["model"] != model:
        diverged = f"replay diverged at trial {n}: log says {model}, replay gives {p['model']}"
        break
    if state == "COMPLETE":
        study.tell(t, value)
    else:
        t.report(value, 1)
        study.tell(t, state=optuna.trial.TrialState.PRUNED)
    rows.append({"number": n, "state": state, "value": value, **{f"params_{k}": v for k, v in p.items()}})

if diverged is None:
    print("replay consistent with the log for all", len(rows), "trials (model family matches at every step)")
    trials = pd.DataFrame(rows)
    trials.to_csv(C.TAB / "automl_trials.csv", index=False)
else:
    # The TPE replay is only bit-reproducible on the platform / library build that ran the search
    # (tiny floating-point differences can flip a sampled category). Fall back to the parameters
    # recovered on that platform (committed automl_trials.csv), after checking they match the log;
    # the re-evaluation below is what actually proves they are the logged configurations.
    print(f"WARNING: {diverged}. Using the committed reports/tables/automl_trials.csv instead.")
    trials = pd.read_csv(C.TAB / "automl_trials.csv")
    got = list(zip(trials.number, trials.state, trials.value.round(4), trials.params_model))
    want = [(n, s, round(v, 4), m) for n, s, v, m in logged]
    if got != want:
        raise SystemExit("committed automl_trials.csv does not match the search log")
    print(f"committed automl_trials.csv matches the log for all {len(trials)} trials")

# ---- re-evaluate the top configurations on the same CV to verify the reconstruction
df = pd.read_parquet(C.PROCESSED / "corpus.parquet")
split = pd.read_csv(C.PROCESSED / "split.csv")
tr = df[split.split.values == "train"].reset_index(drop=True)
sub = tr.groupby(["source", "label"]).sample(frac=20_000 / len(tr), random_state=C.SEED).reset_index(drop=True)
folds = cv_folds(sub, n_splits=3)

top = trials[trials.state == "COMPLETE"].sort_values(["value", "number"], ascending=[False, True]).head(3)
check = []
for r in top.itertuples():
    p = {k.replace("params_", ""): v for k, v in r._asdict().items() if k.startswith("params_") and not (isinstance(v, float) and np.isnan(v))}
    for k in ("ngram_max", "min_df", "max_features"):
        p[k] = int(p[k])
    for k in ("use_char", "use_stats"):
        p[k] = bool(p[k])
    val, trn = [], []
    for a, b in folds:
        ct = build_features(p)
        Xa, Xb = ct.fit_transform(sub.iloc[a]), ct.transform(sub.iloc[b])
        h = build_head(p).fit(Xa, sub.label.iloc[a])
        val.append(f1_score(sub.label.iloc[b], h.predict(Xb)))
        trn.append(f1_score(sub.label.iloc[a], h.predict(Xa)))
    check.append({"trial": r.number, "logged_cv_f1": r.value, "recomputed_cv_f1": round(float(np.mean(val)), 4),
                  "cv_f1_std": round(float(np.std(val)), 4), "train_f1": round(float(np.mean(trn)), 4), "params": p})
    print(check[-1], flush=True)

ok = [c for c in check if abs(c["recomputed_cv_f1"] - c["logged_cv_f1"]) < 1e-4]
print(f"{len(ok)}/{len(check)} configurations reproduce their logged score (|difference| < 1e-4)")
# Best = highest score; ties broken in favour of the ridge_lsq family as recorded, then the earliest trial.
best = sorted(ok, key=lambda c: (-c["recomputed_cv_f1"], c["params"]["model"] != "ridge_lsq", c["trial"]))[0]
json.dump({"source": "AutoML v0.1 run 2 (stopped after 34 trials, 23 min of search)", "best": best, "verified_top3": check},
          open(C.TAB / "automl_best_config.json", "w"), indent=1, default=str)
print("BEST:", best)
