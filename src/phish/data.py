import hashlib

import numpy as np
import pandas as pd

from . import config as cfg
from .text import clean_text, normalize


def load_sources():
    """Read the six per-source CSVs. We don't use the merged phishing_email.csv:
    it has no source column and its text was already pre-processed."""
    frames = []
    for src in cfg.SOURCES:
        d = pd.read_csv(cfg.RAW / f"{src}.csv")
        d["source"] = src
        d["row_in_source"] = np.arange(len(d))
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df


def text_hash(s):
    return hashlib.md5(normalize(s).encode()).hexdigest()


def build_text(df):
    df = df.copy()
    df["raw_text"] = (df["subject"] + " \n " + df["body"]).str.slice(0, cfg.MAX_CHARS)
    df["norm_hash"] = df["raw_text"].map(text_hash)
    df["text_min"] = df["raw_text"].map(lambda s: clean_text(s, mask=False))
    df["text_mask"] = df["raw_text"].map(lambda s: clean_text(s, mask=True))
    return df


def near_duplicate_clusters(texts):
    """One cluster id per text. Two texts are linked when MinHash-LSH says their
    word 5-shingles have Jaccard >= threshold, then we merge with union-find."""
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=cfg.NEARDUP_THRESHOLD, num_perm=cfg.NEARDUP_NUM_PERM)
    hashes = []
    for i, t in enumerate(texts):
        words = normalize(t).split()[:cfg.NEARDUP_MAX_WORDS]
        shingles = {" ".join(words[j:j + 5]) for j in range(max(1, len(words) - 4))}
        m = MinHash(num_perm=cfg.NEARDUP_NUM_PERM, seed=cfg.SEED)
        m.update_batch([s.encode() for s in shingles])
        lsh.insert(i, m)
        hashes.append(m)

    parent = np.arange(len(hashes))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, m in enumerate(hashes):
        for j in lsh.query(m):
            if j != i:
                ri, rj = root(i), root(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    return np.array([root(i) for i in range(len(hashes))])


def load_split(columns=None):
    """Corpus + frozen split (from 02_split.py). Returns (df, train, test)."""
    df = pd.read_parquet(cfg.PROCESSED / "corpus.parquet", columns=columns)
    split = pd.read_csv(cfg.PROCESSED / "split.csv").split.values
    train = df[split == "train"].reset_index(drop=True)
    test = df[split == "test"].reset_index(drop=True)
    return df, train, test
