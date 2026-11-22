"""Metrics and plots to compare phishing classifiers: confusion matrix, precision,
recall, F1 / F0.5 / F2, MCC, ROC and PR curves, FPR at 95% recall."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def fpr_at_recall(y_true, y_proba, target=0.95):
    """Lowest FPR we can get while keeping recall >= target, and the threshold for it."""
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    ok = np.where(tpr >= target)[0]
    i = ok[np.argmin(fpr[ok])]
    return float(fpr[i]), float(thr[i])


def evaluate(y_true, y_proba, name, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr95, thr95 = fpr_at_recall(y_true, y_proba)
    return {
        "model": name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f0.5": fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "fpr": fp / (fp + tn),
        "fpr_at_95_recall": fpr95,
        "threshold_at_95_recall": thr95,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def plot_dashboard(y_true, y_proba, name, threshold=0.5, save_path=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ax = axes[0, 0]
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Legit", "Phishing"])
    ax.set_yticks([0, 1], ["Legit", "Phishing"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (threshold {threshold})")

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    axes[0, 1].plot(fpr, tpr)
    axes[0, 1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0, 1].set_xlabel("False positive rate")
    axes[0, 1].set_ylabel("Recall")
    axes[0, 1].set_title(f"ROC (AUC {roc_auc_score(y_true, y_proba):.3f})")

    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    axes[1, 0].plot(rec, prec)
    axes[1, 0].set_xlabel("Recall")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title(f"Precision-recall (AP {average_precision_score(y_true, y_proba):.3f})")

    thresholds = np.linspace(0.01, 0.99, 99)
    rows = [evaluate_at(y_true, y_proba, t) for t in thresholds]
    for k in ("precision", "recall", "f1", "fpr"):
        axes[1, 1].plot(thresholds, [r[k] for r in rows], label=k)
    axes[1, 1].axvline(threshold, color="grey", ls=":")
    axes[1, 1].set_xlabel("Threshold")
    axes[1, 1].set_title("Effect of the threshold")
    axes[1, 1].legend()

    fig.suptitle(name)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print("saved", save_path)
    return fig


def evaluate_at(y_true, y_proba, t):
    y_pred = (np.asarray(y_proba) >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"precision": tp / (tp + fp) if tp + fp else 1.0, "recall": tp / (tp + fn),
            "f1": 2 * tp / (2 * tp + fp + fn), "fpr": fp / (fp + tn)}


def plot_comparison(results, metrics=("precision", "recall", "f1", "f2", "mcc", "pr_auc"), save_path=None):
    table = pd.DataFrame(results).set_index("model")[list(metrics)]
    ax = table.T.plot.bar(figsize=(10, 5), rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison (same test set)")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=7)
    fig = ax.get_figure()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print("saved", save_path)
    return fig
