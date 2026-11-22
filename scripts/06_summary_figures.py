"""Figures 7 and 8 of the report, and the baseline vs state-of-the-art table."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phish import config as cfg
from phish import viz
from phish.metrics import precision_at_prevalence

viz.setup()
baseline = json.load(open(cfg.TAB / "baseline_results.json"))
hold = baseline["results"]["baseline_tfidf_lr_holdout"]
loso = baseline["loso"]

SOTA = cfg.SOTA_LOCO

# --- fig 7: FPR in-distribution vs on an unseen source
fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
ax = axes[0]
srcs = cfg.BINARY_SOURCES
y = np.arange(len(srcs) + 1)
ax.scatter([hold["fpr"] * 100], [0], color=viz.COLORS[0], s=40, zorder=3, label="Ours, random hold-out (in-distribution)")
ax.scatter([loso[s]["fpr"] * 100 for s in srcs], y[1:], color=viz.COLORS[1], s=40, zorder=3, label="Ours, leave-one-source-out")
ax.scatter([SOTA[s]["fpr"] * 100 for s in srcs], y[1:], marker="D", facecolor="none", edgecolor=viz.INK2, s=36, zorder=3,
           label="Bhuiyan & Bhuiyan 2026 (LOCO, same model)")
for i, s in enumerate(srcs, start=1):
    ax.plot([hold["fpr"] * 100, loso[s]["fpr"] * 100], [i, i], color=viz.GRID, lw=1.5, zorder=1)
    ax.text(loso[s]["fpr"] * 100 + 0.8, i - 0.32, f"{loso[s]['fpr'] * 100:.1f}%", fontsize=7.5, color=viz.INK2)
ax.text(hold["fpr"] * 100 + 0.8, -0.32, f"{hold['fpr'] * 100:.1f}%", fontsize=7.5, color=viz.INK2)
ax.set_yticks(y, ["Mixed test set"] + [f"held-out {s}" for s in srcs])
ax.invert_yaxis()
ax.set_xlabel("False-positive rate: legitimate mails flagged (%)")
ax.set_xlim(0, 30)
ax.set_title("Legit mails wrongly flagged")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), fontsize=7.5)
ax.grid(axis="y", visible=False)

ax = axes[1]
prev = np.logspace(-2, np.log10(0.5), 60)
ax.plot(prev * 100, [precision_at_prevalence(hold["recall"], hold["fpr"], p) for p in prev], color=viz.COLORS[0], label="In-distribution (hold-out)")
for s, color, ls in [("CEAS_08", viz.COLORS[1], "-"), ("Enron", viz.COLORS[2], "-"), ("SpamAssasin", viz.COLORS[3], "--")]:
    label = f"LOSO held-out {s}" + (" (overlaps CEAS_08)" if s == "SpamAssasin" else "")
    ax.plot(prev * 100, [precision_at_prevalence(loso[s]["recall"], loso[s]["fpr"], p) for p in prev], color=color, ls=ls, label=label)
ax.set_xscale("log")
ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
ax.set_xlabel("Share of malicious mails in the traffic (%, log)")
ax.set_ylabel("Precision of the alerts")
ax.set_ylim(0, 1.02)
ax.set_title("Precision if malicious mail is rare")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7.5)
fig.suptitle("Baseline (TF-IDF + LR): in-distribution vs unseen source", fontweight="bold", fontsize=10)
fig.tight_layout()
fig.savefig(cfg.FIG / "fig7_generalisation_gap.png")
plt.close(fig)

cols = ["precision", "recall", "f1", "fpr", "roc_auc", "precision_at_5pct"]
rows = [{"evaluation": "Random hold-out (ours)", **{k: hold[k] for k in cols}}]
for s in cfg.SOURCES:
    rows.append({"evaluation": f"LOSO: {s}", **{k: loso[s].get(k, np.nan) for k in cols},
                 "sota_f1": SOTA.get(s, {}).get("f1", np.nan), "sota_fpr": SOTA.get(s, {}).get("fpr", np.nan)})
pd.DataFrame(rows).round(4).to_csv(cfg.TAB / "summary_baseline_vs_sota.csv", index=False)

# --- fig 8: AutoML search + baseline vs AutoML
if (cfg.TAB / "automl_trials.csv").exists() and (cfg.TAB / "automl_best_results.json").exists():
    trials = pd.read_csv(cfg.TAB / "automl_trials.csv")
    automl = json.load(open(cfg.TAB / "automl_best_results.json"))
    names = {"logreg": "Logistic regression", "linear_svm": "Linear SVM", "complement_nb": "Complement NB",
             "sgd_huber": "SGD (mod. Huber)", "ridge_lsq": "Ridge least-squares", "decision_tree": "Decision tree"}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    done = trials[trials.state == "COMPLETE"]
    for i, m in enumerate(names):
        t = done[done.params_model == m]
        ax.scatter(t.number, t.value, s=20, color=viz.COLORS[i], label=names[m], zorder=3)
    pruned = trials[trials.state == "PRUNED"]
    ax.scatter(pruned.number, pruned.value, s=14, marker="x", color=viz.INK2, label="Pruned (partial CV)", zorder=2, linewidths=0.8)
    best_so_far = done.set_index("number").value.reindex(trials.number).ffill().cummax()
    ax.step(trials.number, best_so_far.values, where="post", color=viz.INK, lw=1.2, label="Best so far")
    ax.set_ylim(0.88, 0.99)
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Mean CV F1 (3-fold grouped)")
    ax.set_title("Search history, run 2 (34 trials)")
    ax.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17))

    ax = axes[1]
    labels = ["Hold-out"] + [f"LOSO {s}" for s in cfg.BINARY_SOURCES] + ["Nazario (recall)", "Nigerian (recall)"]

    def values(h, l):
        return [h["f1"]] + [l[s]["f1"] for s in cfg.BINARY_SOURCES] + [l["Nazario"]["recall"], l["Nigerian_Fraud"]["recall"]]

    vb, va = values(hold, loso), values(automl["holdout"], automl["loso"])
    yy = np.arange(len(labels))
    for i in range(len(labels)):
        ax.plot([min(vb[i], va[i]), max(vb[i], va[i])], [i, i], color=viz.GRID, lw=2, zorder=1)
    ax.scatter(vb, yy, color=viz.COLORS[0], s=36, zorder=3, label="Baseline TF-IDF + LR")
    ax.scatter(va, yy, color=viz.COLORS[1], s=36, zorder=3, marker="D", label=f"AutoML v0.1 best ({names[automl['config']['model']]})")
    ax.set_yticks(yy, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("F1 (recall for positive-only sources)")
    ax.set_title("Baseline vs AutoML best, full refit")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.4, -0.17))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(cfg.FIG / "fig8_automl.png")
    plt.close(fig)
print("figures written")
