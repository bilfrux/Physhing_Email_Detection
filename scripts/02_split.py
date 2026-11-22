"""Frozen 80/20 split, saved in data/processed/split.csv so every script uses the same test set."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from phish import config as cfg
from phish.data import text_hash
from phish.split import holdout_split

df = pd.read_parquet(cfg.PROCESSED / "corpus.parquet")
df["split"] = ["test" if t else "train" for t in holdout_split(df)]
df[["source", "row_in_source", "split"]].to_csv(cfg.PROCESSED / "split.csv", index_label="row_id")

train, test = df[df.split == "train"], df[df.split == "test"]
tab = df.groupby(["source", "split"]).label.agg(n="size", pct_malicious=lambda s: round(100 * s.mean(), 1)).unstack("split")
tab.loc["TOTAL"] = [len(test), len(train), round(100 * test.label.mean(), 1), round(100 * train.label.mean(), 1)]
tab.to_csv(cfg.TAB / "split_composition.csv")
print(tab)

# leakage checks
assert set(train.cluster).isdisjoint(test.cluster), "a near-duplicate cluster is on both sides"
assert set(train.raw_text.map(text_hash)).isdisjoint(test.raw_text.map(text_hash)), "an exact duplicate is on both sides"
print("leakage checks ok: no shared cluster and no shared text between train and test")
print(f"test share = {len(test) / len(df):.3f}")
