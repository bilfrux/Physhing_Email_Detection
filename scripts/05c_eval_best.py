"""Evaluate the saved best AutoML v0.1 configuration (automl_best_config.json):
refit on the full training split -> hold-out test, then leave-one-source-out.
Same protocol as the baseline (04), so the two are directly comparable."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import time

import pandas as pd

from phish import config as C
from phish.features import build_full_pipeline, scores
from phish.metrics import full_report
from phish.split import loso_folds

best = json.load(open(C.TAB / "automl_best_config.json"))["best"]
p = best["params"]
print("config:", p, flush=True)

df = pd.read_parquet(C.PROCESSED / "corpus.parquet")
split = pd.read_csv(C.PROCESSED / "split.csv")
tr = df[split.split.values == "train"].reset_index(drop=True)
te = df[split.split.values == "test"].reset_index(drop=True)

t0 = time.time()
pipe = build_full_pipeline(p).fit(tr, tr.label)
s = scores(pipe, te)
res = {"holdout": full_report(te.label, pipe.predict(te), s), "train": full_report(tr.label, pipe.predict(tr), scores(pipe, tr))}
pd.DataFrame({"source": te.source, "label": te.label, "score": s}).to_parquet(C.PROCESSED / "pred_automl_test.parquet")
print("holdout", res["holdout"], f"{time.time() - t0:.0f}s", flush=True)

loso = {}
for src, a, b in loso_folds(df):
    m = build_full_pipeline(p).fit(df.iloc[a], df.label.iloc[a])  # features re-learnt on the 5 training sources only
    loso[src] = full_report(df.label.iloc[b], m.predict(df.iloc[b]), scores(m, df.iloc[b]))
    print("LOSO", src, loso[src], f"{time.time() - t0:.0f}s", flush=True)
res["loso"] = loso
json.dump({"config": p, "trial": best["trial"], "cv_f1": best["recomputed_cv_f1"], **res}, open(C.TAB / "automl_best_results.json", "w"), indent=1)
print("done")
