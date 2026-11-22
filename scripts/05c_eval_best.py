"""Best AutoML config: refit on the train split, test on the hold-out and in
leave-one-source-out. Same protocol as the baseline (04)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import time

import pandas as pd

from phish import config as cfg
from phish.data import load_split
from phish.features import build_full_pipeline, scores
from phish.metrics import full_report
from phish.split import loso_folds

best = json.load(open(cfg.TAB / "automl_best_config.json"))["best"]
p = best["params"]
print("config:", p, flush=True)

df, train, test = load_split()
start = time.time()
pipe = build_full_pipeline(p).fit(train, train.label)
s = scores(pipe, test)
res = {"holdout": full_report(test.label, pipe.predict(test), s),
       "train": full_report(train.label, pipe.predict(train), scores(pipe, train))}
pd.DataFrame({"source": test.source, "label": test.label, "score": s}).to_parquet(cfg.PROCESSED / "pred_automl_test.parquet")
print("holdout", res["holdout"], f"{time.time() - start:.0f}s", flush=True)

res["loso"] = {}
for src, a, b in loso_folds(df):
    m = build_full_pipeline(p).fit(df.iloc[a], df.label.iloc[a])
    res["loso"][src] = full_report(df.label.iloc[b], m.predict(df.iloc[b]), scores(m, df.iloc[b]))
    print("LOSO", src, res["loso"][src], f"{time.time() - start:.0f}s", flush=True)

json.dump({"config": p, "trial": best["trial"], "cv_f1": best["recomputed_cv_f1"], **res},
          open(cfg.TAB / "automl_best_results.json", "w"), indent=1)
print("done")
