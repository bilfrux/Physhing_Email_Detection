"""Splits. Duplicates are removed before (script 01), folds keep near-duplicate
clusters together and are stratified on source x label."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from . import config as cfg


def holdout_split(df, grouped=True):
    """True for the ~20% test rows (first fold of a 5-fold split)."""
    strata = df["source"] + "_" + df["label"].astype(str)
    if grouped:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg.SEED)
        folds = list(cv.split(df, strata, groups=df["cluster"]))
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.SEED)
        folds = list(cv.split(df, strata))
    is_test = np.zeros(len(df), dtype=bool)
    is_test[folds[0][1]] = True
    return is_test


def cv_folds(df, n_splits=5):
    strata = df["source"] + "_" + df["label"].astype(str)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.SEED)
    return list(cv.split(df, strata, groups=df["cluster"]))


def loso_folds(df):
    """Leave-one-source-out: yields (source, train_idx, test_idx) for each source."""
    for src in cfg.SOURCES:
        is_src = df["source"].values == src
        yield src, np.where(~is_src)[0], np.where(is_src)[0]
