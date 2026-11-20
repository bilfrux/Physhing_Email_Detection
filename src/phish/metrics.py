"""Evaluation metrics.

Accuracy alone is never reported: the deployment question is
"how many legitimate mails do we wrongly quarantine (FPR) and how many
malicious mails do we miss (recall)?".
"""
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


def precision_at_prevalence(tpr: float, fpr: float, prevalence: float) -> float:
    """Precision the same classifier would have if malicious mails were only
    `prevalence` of the traffic (Bayes rule; TPR/FPR are prevalence-free)."""
    num = tpr * prevalence
    den = num + fpr * (1 - prevalence)
    return float(num / den) if den > 0 else float("nan")


def full_report(y_true, y_pred, y_score=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {"n": int(len(y_true)), "pos_rate": float(y_true.mean())}
    out["recall"] = recall_score(y_true, y_pred, zero_division=0)
    if len(np.unique(y_true)) < 2:  # positive-only corpus: only recall is defined
        return out
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update(
        precision=precision_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        balanced_acc=balanced_accuracy_score(y_true, y_pred),
        accuracy=float((tp + tn) / len(y_true)),
        fpr=float(fp / (fp + tn)) if (fp + tn) else float("nan"),
    )
    out["precision_at_5pct"] = precision_at_prevalence(out["recall"], out["fpr"], 0.05)
    if y_score is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)
        if np.min(y_score) >= 0 and np.max(y_score) <= 1:
            out["brier"] = brier_score_loss(y_true, y_score)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
