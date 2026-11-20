"""Step 2.2 - learning/testing plan: build and freeze the hold-out split.

Writes data/processed/split.csv (row id -> train/test) so that every later
script uses exactly the same test set (reproducibility, no peeking).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from phish import config as C
from phish.split import holdout_split

df = pd.read_parquet(C.PROCESSED / "corpus.parquet")
is_test = holdout_split(df)
df["split"] = ["test" if t else "train" for t in is_test]
df[["source", "row_in_source", "split"]].to_csv(C.PROCESSED / "split.csv", index_label="row_id")

tab = df.groupby(["source", "split"]).label.agg(n="size", pct_malicious=lambda s: round(100 * s.mean(), 1)).unstack("split")
tab.loc["TOTAL"] = [
    (df.split == "test").sum(), (df.split == "train").sum(),
    round(100 * df[df.split == "test"].label.mean(), 1), round(100 * df[df.split == "train"].label.mean(), 1),
]
tab.to_csv(C.TAB / "split_composition.csv")
print(tab)

# Leakage sanity checks -------------------------------------------------------
tr, te = df[df.split == "train"], df[df.split == "test"]
assert set(tr.cluster).isdisjoint(te.cluster), "near-duplicate cluster crosses the split"
from phish.text import basic_normalize
import hashlib
h = lambda s: hashlib.md5(basic_normalize(s).encode()).hexdigest()
assert set(tr.raw_text.map(h)).isdisjoint(te.raw_text.map(h)), "exact duplicate crosses the split"
print("leakage checks passed: no shared cluster, no shared normalised text between train and test")
print(f"test share = {len(te) / len(df):.3f}")
