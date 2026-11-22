import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def precision_at_prevalence(tpr, fpr, prevalence):
    """Precision we would get if only `prevalence` of the mails were malicious (Bayes rule)."""
    hits = tpr * prevalence
    alerts = hits + fpr * (1 - prevalence)
    return float(hits / alerts) if alerts > 0 else float("nan")


def full_report(y_true, y_pred, y_score=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    res = {"n": int(len(y_true)), "pos_rate": float(y_true.mean())}
    res["recall"] = recall_score(y_true, y_pred, zero_division=0)
    if len(np.unique(y_true)) < 2:  # Nazario / Nigerian: no legit mail, only recall
        return res
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    res["precision"] = precision_score(y_true, y_pred, zero_division=0)
    res["f1"] = f1_score(y_true, y_pred, zero_division=0)
    res["balanced_acc"] = balanced_accuracy_score(y_true, y_pred)
    res["accuracy"] = float((tp + tn) / len(y_true))
    res["fpr"] = float(fp / (fp + tn)) if (fp + tn) else float("nan")
    res["precision_at_5pct"] = precision_at_prevalence(res["recall"], res["fpr"], 0.05)
    if y_score is not None:
        res["roc_auc"] = roc_auc_score(y_true, y_score)
        res["pr_auc"] = average_precision_score(y_true, y_score)
        if np.min(y_score) >= 0 and np.max(y_score) <= 1:  # only for probabilities
            res["brier"] = brier_score_loss(y_true, y_score)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}
