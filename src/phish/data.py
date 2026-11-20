"""Loading, harmonisation and de-duplication of the six source corpora."""
import hashlib

import numpy as np
import pandas as pd

from . import config as C
from .text import basic_normalize, clean_text


def load_sources() -> pd.DataFrame:
    """Load the six per-source CSVs and harmonise them to one schema.

    We deliberately do NOT use the pre-merged phishing_email.csv: it has no
    `source` column (needed for leave-one-source-out evaluation) and its text
    was already pre-processed by the dataset authors in an undocumented way.
    """
    frames = []
    for src in C.SOURCES:
        d = pd.read_csv(C.RAW / f"{src}.csv")
        d["source"] = src
        d["row_in_source"] = np.arange(len(d))
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df


def build_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["raw_text"] = (df["subject"] + " \n " + df["body"]).str.slice(0, C.MAX_CHARS)
    df["norm_hash"] = df["raw_text"].map(
        lambda s: hashlib.md5(basic_normalize(s).encode()).hexdigest()
    )
    df["text_min"] = df["raw_text"].map(lambda s: clean_text(s, mask=False))
    df["text_mask"] = df["raw_text"].map(lambda s: clean_text(s, mask=True))
    return df


def near_duplicate_clusters(texts: pd.Series) -> np.ndarray:
    """Cluster near-duplicates (Jaccard >= threshold on word 5-shingles) with
    MinHash-LSH + union-find. Returns one cluster id per row."""
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=C.NEARDUP_THRESHOLD, num_perm=C.NEARDUP_NUM_PERM)
    sigs = []
    for i, t in enumerate(texts):
        w = basic_normalize(t).split()[: C.NEARDUP_MAX_WORDS]
        shingles = {" ".join(w[j : j + 5]) for j in range(max(1, len(w) - 4))}
        m = MinHash(num_perm=C.NEARDUP_NUM_PERM, seed=C.SEED)
        m.update_batch([s.encode() for s in shingles])
        lsh.insert(i, m)
        sigs.append(m)

    parent = np.arange(len(sigs))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, m in enumerate(sigs):
        for j in lsh.query(m):
            if j != i:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(len(sigs))])
