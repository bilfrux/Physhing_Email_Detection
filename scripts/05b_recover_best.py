"""Get back the parameters of AutoML run 2.

We stopped run 2 before it saved its trials, only the log (scores + model name)
is left. The TPE sampler is deterministic given its seed and the history, so we
replay the log with ask/tell and get the same parameters. Then we recompute the CV
score of the 3 best configs to check they really match the log.

Writes reports/tables/automl_trials.csv and automl_best_config.json.
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

from phish import config as cfg
from phish.data import load_split
from phish.features import build_features, build_head, new_study, search_sample, suggest
from phish.split import cv_folds

optuna.logging.set_verbosity(optuna.logging.WARNING)
log = (cfg.TAB / "log_05_run2_stopped.txt").read_text()
pattern = r"trial\s+(\d+) (COMPLETE|PRUNED)\s+value=([\d.]+) model=(\w+)"
logged = [(int(n), state, float(v), model) for n, state, v, model in re.findall(pattern, log)]
print(f"{len(logged)} trials in log")

study = new_study()
rows, problem = [], None
for n, state, value, model in logged:
    t = study.ask()
    p = suggest(t)
    if p["model"] != model:
        problem = f"replay diverged at trial {n}: log says {model}, replay gives {p['model']}"
        break
    if state == "COMPLETE":
        study.tell(t, value)
    else:
        t.report(value, 1)
        study.tell(t, state=optuna.trial.TrialState.PRUNED)
    rows.append({"number": n, "state": state, "value": value, **{f"params_{k}": v for k, v in p.items()}})

if problem is None:
    print("replay consistent with the log for all", len(rows), "trials (model family matches at every step)")
    trials = pd.DataFrame(rows)
    trials.to_csv(cfg.TAB / "automl_trials.csv", index=False)
else:
    # the replay is only exact on the machine that ran the search (tiny float differences
    # can change a sampled choice). Then we use the committed csv, after checking it
    # matches the log. The re-evaluation below is the real check anyway.
    print(f"WARNING: {problem}. Using the committed reports/tables/automl_trials.csv instead.")
    trials = pd.read_csv(cfg.TAB / "automl_trials.csv")
    got = list(zip(trials.number, trials.state, trials.value.round(4), trials.params_model))
    if got != [(n, s, round(v, 4), m) for n, s, v, m in logged]:
        raise SystemExit("committed automl_trials.csv does not match the search log")
    print(f"committed automl_trials.csv matches the log for all {len(trials)} trials")

# --- recompute the CV score of the top 3
_, train, _ = load_split()
sub = search_sample(train)
folds = cv_folds(sub, n_splits=3)


def params_from_row(r):
    p = {k.replace("params_", ""): v for k, v in r._asdict().items()
         if k.startswith("params_") and not (isinstance(v, float) and np.isnan(v))}
    for k in ("ngram_max", "min_df", "max_features"):
        p[k] = int(p[k])
    for k in ("use_char", "use_stats"):
        p[k] = bool(p[k])
    return p


top3 = trials[trials.state == "COMPLETE"].sort_values(["value", "number"], ascending=[False, True]).head(3)
checks = []
for r in top3.itertuples():
    p = params_from_row(r)
    val, trn = [], []
    for a, b in folds:
        ct = build_features(p)
        Xa, Xb = ct.fit_transform(sub.iloc[a]), ct.transform(sub.iloc[b])
        head = build_head(p).fit(Xa, sub.label.iloc[a])
        val.append(f1_score(sub.label.iloc[b], head.predict(Xb)))
        trn.append(f1_score(sub.label.iloc[a], head.predict(Xa)))
    checks.append({"trial": r.number, "logged_cv_f1": r.value, "recomputed_cv_f1": round(float(np.mean(val)), 4),
                   "cv_f1_std": round(float(np.std(val)), 4), "train_f1": round(float(np.mean(trn)), 4), "params": p})
    print(checks[-1], flush=True)

ok = [c for c in checks if abs(c["recomputed_cv_f1"] - c["logged_cv_f1"]) < 1e-4]
print(f"{len(ok)}/{len(checks)} configurations reproduce their logged score (|difference| < 1e-4)")
# ties: keep ridge_lsq as in the original run, then the earliest trial
best = sorted(ok, key=lambda c: (-c["recomputed_cv_f1"], c["params"]["model"] != "ridge_lsq", c["trial"]))[0]
json.dump({"source": "AutoML v0.1 run 2 (stopped after 34 trials, 23 min of search)", "best": best, "verified_top3": checks},
          open(cfg.TAB / "automl_best_config.json", "w"), indent=1, default=str)
print("BEST:", best)
