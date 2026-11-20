"""Train/test construction.

Rules (each one closes a leakage path):
1. De-duplication happens BEFORE any split (script 01).
2. Splits are grouped by near-duplicate cluster -> two near-copies can never
   sit on both sides of the split.
3. Splits are stratified on (source, label) so every source keeps its class
   mix in train and test.
4. All fitted transforms (TF-IDF vocabulary/IDF, scalers, feature selectors)
   live inside an sklearn Pipeline -> they are re-fitted on the training fold
   only, never on test data.
5. The test split is touched once, at the end; model selection uses CV on train.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from . import config as C


def holdout_split(df: pd.DataFrame, test_fold: int = 0, n_splits: int = 5, grouped: bool = True) -> np.ndarray:
    """Return a boolean mask: True = test (~20%)."""
    strata = df["source"] + "_" + df["label"].astype(str)
    if grouped:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=C.SEED)
        folds = list(cv.split(df, strata, groups=df["cluster"]))
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=C.SEED)
        folds = list(cv.split(df, strata))
    is_test = np.zeros(len(df), dtype=bool)
    is_test[folds[test_fold][1]] = True
    return is_test


def cv_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = C.SEED):
    """Grouped + stratified CV folds for model selection (training data only)."""
    strata = df["source"] + "_" + df["label"].astype(str)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(df, strata, groups=df["cluster"]))


def loso_folds(df: pd.DataFrame):
    """Leave-one-source-out: yields (held_out_source, train_idx, test_idx).

    Every source is held out once. Positive-only sources (Nazario, Nigerian)
    are evaluated with recall only (no negatives -> precision/F1/FPR undefined).
    """
    for src in C.SOURCES:
        test = np.where(df["source"].values == src)[0]
        train = np.where(df["source"].values != src)[0]
        yield src, train, test
