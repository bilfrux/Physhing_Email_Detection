"""Writes the LaTeX tables and the numbers used in the report text
(reports/latex/tables/*.tex) from the result files. Run it last."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json

import numpy as np
import pandas as pd

from phish import config as cfg

OUT = cfg.ROOT / "reports" / "latex" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
NAMES = {"logreg": "Logistic regression", "linear_svm": "Linear SVM", "complement_nb": "Complement NB",
         "sgd_huber": "SGD (mod.\\ Huber)", "ridge_lsq": "Ridge least-squares", "decision_tree": "Decision tree"}


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("->", r"$\rightarrow$").replace("&", r"\&")


def missing(x):
    return x is None or (isinstance(x, float) and np.isnan(x))


def f(x, d=3):
    return "--" if missing(x) else f"{x:.{d}f}"


def pct(x, d=1):
    return "--" if missing(x) else f"{100 * x:.{d}f}"


def table(name, cols, header, rows):
    body = "\n".join(rows)
    (OUT / f"{name}.tex").write_text(
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{header}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}")
    print("wrote", name)


def macros(d):
    return "".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in d.items())


def load(name):
    path = cfg.TAB / name
    return json.load(open(path)) if name.endswith(".json") else pd.read_csv(path, index_col=0)


baseline = load("baseline_results.json")
B_HOLD, B_LOSO = baseline["results"]["baseline_tfidf_lr_holdout"], baseline["loso"]
has_automl = (cfg.TAB / "automl_best_results.json").exists()

# --- data audit
comp, final = load("audit_composition.csv"), load("final_corpus_composition.csv")
rows = []
for s in cfg.SOURCES + ["TOTAL"]:
    raw, after = int(comp.loc[s, "n"]), int(final.loc[s, "n"])
    rows.append(f"{esc(s)} & {raw:,} & {comp.loc[s, 'pct_malicious']:.1f} & {after:,} & {100 * (1 - after / raw):.1f} & {final.loc[s, 'pct_malicious']:.1f} \\\\")
rows.insert(-1, r"\midrule")
table("tab_composition", "lrrrrr", r"Source & Raw e-mails & \% malicious & After dedup & \% removed & \% malicious \\", rows)

miss = load("audit_missingness_pct.csv")
rows = [f"{esc(s)} & " + " & ".join(f"{miss.loc[s, c]:.1f}" for c in miss.columns) + r" \\" for s in cfg.SOURCES]
table("tab_missing", "l" + "r" * len(miss.columns), "Source & " + " & ".join(esc(c) for c in miss.columns) + r" \\", rows)

style = load("audit_style_artifacts.csv")
rows = []
for s in cfg.SOURCES:
    r = style.loc[s]
    years = "--" if np.isnan(r.year_median) else f"{int(r.year_p05)}--{int(r.year_p95)}"
    rows.append(f"{esc(s)} & {int(r.median_chars):,} & {int(r.max_chars):,} & {r.uppercase_char_ratio_pct:.1f} & {r.pct_spaced_punct:.1f} & {r.pct_with_http:.1f} & {years} \\\\")
table("tab_style", "lrrrrrc", r"Source & Median chars & Max chars & \% upper-case chars & \% with ``\,.\,'' spacing & \% with http & Years (p5--p95) \\", rows)

# --- shortcut test
shortcut = pd.read_csv(cfg.TAB / "shortcut_source_prediction.csv")
rows = [f"{esc(r.input)} & {pct(r.source_accuracy)} & {pct(r.source_macro_f1)} \\\\" for r in shortcut.itertuples()]
table("tab_shortcut", "lrr", r"Input used to predict the \emph{source} (6 classes) & Accuracy (\%) & Macro-F1 (\%) \\", rows)
label_from_src = load("shortcut_label_from_source.csv")["value"]
(OUT / "val_label_from_source.tex").write_text(f"{100 * label_from_src['accuracy_label_from_source_only']:.1f}")

tokens = load("shortcut_top_tokens_per_source.csv").iloc[:, 0]
short = lambda t: t if len(t) <= 16 else t[:14] + "..."
rows = [f"{esc(s)} & \\texttt{{{esc(', '.join(short(t) for t in tokens[s].split(', ')[:9]))}}} \\\\" for s in cfg.SOURCES]
table("tab_top_tokens", r"lp{0.72\linewidth}", r"Source & Most source-identifying tokens after masking \\", rows)

# --- split
split = pd.read_csv(cfg.TAB / "split_composition.csv", header=[0, 1], index_col=0)
rows = []
for s in cfg.SOURCES + ["TOTAL"]:
    r = split.loc[s]
    rows.append(f"{esc(s)} & {int(r[('n', 'train')]):,} & {int(r[('n', 'test')]):,} & {r[('pct_malicious', 'train')]:.1f} & {r[('pct_malicious', 'test')]:.1f} \\\\")
rows.insert(-1, r"\midrule")
table("tab_split", "lrrrr", r"Source & Train & Test & \% malicious (train) & \% malicious (test) \\", rows)

# --- baseline
C = baseline["best_C"]


def baseline_row(name, r):
    return (f"{name} & {f(r.get('precision'))} & {f(r.get('recall'))} & {f(r.get('f1'))} & {pct(r.get('fpr'), 2)} "
            f"& {f(r.get('roc_auc'), 4)} & {f(r.get('precision_at_5pct'))} \\\\")


res = baseline["results"]
table("tab_baseline", "lrrrrrr", r"Model (evaluated on) & Precision & Recall & F1 & FPR (\%) & ROC-AUC & Prec.\ @5\% \\", [
    baseline_row("Majority class (test)", res["dummy_most_frequent"]),
    baseline_row(f"TF-IDF + LR, C={C:g} (train)", res["baseline_tfidf_lr_train"]),
    baseline_row(f"TF-IDF + LR, C={C:g} (test)", res["baseline_tfidf_lr_holdout"]),
])
(OUT / "val_best_C.tex").write_text(f"{C:g}")

rows = [f"{esc(k)} & {v['clusters_crossing_split']:,} & {f(v['f1'], 4)} & {pct(v['fpr'], 2)} & {f(v['precision_at_5pct'])} \\\\"
        for k, v in baseline["inflation"].items()]
table("tab_inflation", "lrrrr", r"Split of the exact-dedup corpus & Clusters crossing & F1 & FPR (\%) & Prec.\ @5\% \\", rows)

summary = pd.read_csv(cfg.TAB / "summary_baseline_vs_sota.csv")
rows = [f"{esc(r.evaluation)} & {f(r.precision)} & {f(r.recall)} & {f(r.f1)} & {pct(r.fpr)} & {f(r.precision_at_5pct)} "
        f"& {f(getattr(r, 'sota_f1', np.nan))} & {pct(getattr(r, 'sota_fpr', np.nan))} \\\\" for r in summary.itertuples()]
rows.insert(1, r"\midrule")
table("tab_loso", "lrrrrrrr", r""" & \multicolumn{5}{c}{Ours: TF-IDF + LR} & \multicolumn{2}{c}{Bhuiyan \& Bhuiyan 2026} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-8}
Evaluation & Prec. & Recall & F1 & FPR (\%) & Prec.\ @5\% & F1 & FPR (\%) \\""", rows)

# --- AutoML
if has_automl:
    trials = pd.read_csv(cfg.TAB / "automl_trials.csv")
    automl = load("automl_best_results.json")
    completed = trials.state == "COMPLETE"
    per_model = trials.groupby("params_model").agg(n=("number", "size"), completed=("state", lambda s: (s == "COMPLETE").sum()),
                                                   best=("value", lambda v: v[completed[v.index]].max()))
    rows = [f"{NAMES[k]} & {int(r.n)} & {int(r.completed)} & {f(r.best, 4)} \\\\" for k, r in per_model.sort_values("best", ascending=False).iterrows()]
    table("tab_automl_families", "lrrr", r"Model family & Trials & Completed & Best CV F1 \\", rows)

    p = automl["config"]
    desc = ["masked" if p["text"] == "text_mask" else "minimal cleaning", f"word 1--{p['ngram_max']}-grams",
            f"min\\_df={p['min_df']}", f"{p['max_features'] // 1000}k features",
            "+ char 3--5-grams" if p["use_char"] else "no char n-grams",
            "+ hand-crafted stats" if p["use_stats"] else "no stats",
            "no selection" if p["select_frac"] >= 1 else f"$\\chi^2$ keep {int(100 * p['select_frac'])}\\%"]
    desc += [f"{'$\\alpha$' if k == 'alpha' else k}={v:.3g}" for k, v in p.items() if k in ("C", "alpha", "max_depth", "min_samples_leaf")]
    (OUT / "val_automl_cfg.tex").write_text(", ".join(desc))

    def short_row(name, r):
        return f"{name} & {f(r.get('precision'))} & {f(r.get('recall'))} & {f(r.get('f1'))} & {pct(r.get('fpr'))} & {f(r.get('precision_at_5pct'))}"

    rows = []
    pairs = [("Hold-out", B_HOLD, automl["holdout"])] + [(f"LOSO: {esc(s)}", B_LOSO[s], automl["loso"][s]) for s in cfg.SOURCES]
    for name, b, a in pairs:
        rows.append(short_row(name + " -- baseline", b) + " \\\\")
        rows.append(short_row(name + " -- AutoML", a) + " \\\\ \\addlinespace")
    table("tab_automl_best", "lrrrrr", r"Evaluation -- model & Prec. & Recall & F1 & FPR (\%) & Prec.\ @5\% \\", rows)
    (OUT / "val_automl.tex").write_text(macros({
        "nTrials": len(trials), "nComplete": completed.sum(), "nPruned": (trials.state == "PRUNED").sum(),
        "bestFamily": NAMES[p["model"]], "bestCV": f"{automl['cv_f1']:.4f}", "bestTrial": automl["trial"]}))

# --- numbers quoted in the text of main.tex / automl_discussion.tex
binary = [B_LOSO[s] for s in cfg.BINARY_SOURCES]
curve = load("baseline_validation_curve.csv")
sota = cfg.SOTA_LOCO
text = {
    "baseFone": f(B_HOLD["f1"]), "baseFPR": pct(B_HOLD["fpr"]), "basePrecFive": f(B_HOLD["precision_at_5pct"]), "baseBrier": f(B_HOLD["brier"]),
    "losoFoneMin": f(min(r["f1"] for r in binary), 2), "losoFoneMax": f(max(r["f1"] for r in binary), 2),
    "losoFPRmin": pct(min(r["fpr"] for r in binary)), "losoFPRmax": pct(max(r["fpr"] for r in binary)),
    "fprRatioMin": f"{min(r['fpr'] for r in binary) / B_HOLD['fpr']:.0f}", "fprRatioMax": f"{max(r['fpr'] for r in binary) / B_HOLD['fpr']:.0f}",
    "nazRecallPct": f"{100 * B_LOSO['Nazario']['recall']:.0f}", "nazRecall": f(B_LOSO["Nazario"]["recall"], 2),
    "nigRecallPct": f"{100 * B_LOSO['Nigerian_Fraud']['recall']:.0f}",
    "sotaFoneDiffMax": f(max(abs(B_LOSO[s]["f1"] - sota[s]["f1"]) for s in sota), 2),
    "sotaFPRDiffMax": f"{100 * max(abs(B_LOSO[s]['fpr'] - sota[s]['fpr']) for s in sota):.1f}",
    "valCtrainOne": f"{curve.index[curve.train_f1_mean >= 0.99995].min():g}",
    "valArgmaxC": f"{curve.val_f1_mean.idxmax():g}",
}
if has_automl:
    A_HOLD, A_LOSO = automl["holdout"], automl["loso"]
    text.update(
        autoFone=f(A_HOLD["f1"]), autoFPR=pct(A_HOLD["fpr"], 2), autoPrecFive=f(A_HOLD["precision_at_5pct"]),
        baseFPRtwo=pct(B_HOLD["fpr"], 2), autoTrainFone=f(automl["train"]["f1"]),
        autoNazRecall=f(A_LOSO["Nazario"]["recall"]), baseNazRecall=f(B_LOSO["Nazario"]["recall"]),
        autoFoneWorse=str(sum(A_LOSO[s]["f1"] < B_LOSO[s]["f1"] for s in cfg.BINARY_SOURCES)),
        autoFPRworse=", ".join(esc(s) for s in cfg.BINARY_SOURCES if A_LOSO[s]["fpr"] > B_LOSO[s]["fpr"]),
        autoFPRbetter=", ".join(esc(s) for s in cfg.BINARY_SOURCES if A_LOSO[s]["fpr"] < B_LOSO[s]["fpr"]))
    for s, k in [("CEAS_08", "Ceas"), ("Enron", "Enron"), ("Ling", "Ling"), ("SpamAssasin", "Spam")]:
        text[f"dFone{k}"] = f"{A_LOSO[s]['f1'] - B_LOSO[s]['f1']:+.3f}".replace("-", "$-$")
if (cfg.TAB / "check_loso_nested_C.csv").exists():  # from 09
    nested_all = pd.read_csv(cfg.TAB / "check_loso_nested_C.csv")
    nested = nested_all.set_index("held_out_source").loc[cfg.BINARY_SOURCES]
    naz = load("check_nazario_nonascii.csv")
    non = naz.loc[[i for i in naz.index if i.startswith("mostly non")][0]]
    asc = naz.loc["mostly ASCII"]
    share_non = non.n / (non.n + asc.n)
    cross = load("check_cross_source_dedup.csv")
    text.update(
        nestedMaxDFone=f"{(nested.f1_nested - nested.f1_global).abs().max():.3f}",
        nestedMaxDFPR=f"{100 * (nested.fpr_nested - nested.fpr_global).abs().max():.1f}",
        nestedCvalues=", ".join(f"{c:g}" for c in sorted(set(nested_all.C_nested))),
        nazNonAsciiN=str(int(non.n)), nazNonAsciiPct=f"{100 * share_non:.1f}",
        nazNonAsciiRecall=f(non.recall, 2), nazAsciiRecall=f(asc.recall, 2),
        nazNonAsciiEffect=f"{100 * (asc.recall - non.recall) * share_non:.1f}",
        xsSpamLost=str(int(cross.loc["SpamAssasin", "total_lost"])), xsSpamPct=f"{cross.loc['SpamAssasin', 'pct_of_source_after_exact_dedup']:.1f}",
        xsNigLost=str(int(cross.loc["Nigerian_Fraud", "total_lost"])), xsNigPct=f"{cross.loc['Nigerian_Fraud', 'pct_of_source_after_exact_dedup']:.1f}",
    )
(OUT / "val_text.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in text.items()))
print("wrote val_text", text)

# --- confidence intervals (from 08)
if (cfg.TAB / "uncertainty_holdout.csv").exists():
    unc = pd.read_csv(cfg.TAB / "uncertainty_holdout.csv").set_index(["model", "metric"])

    def ci(model, metric, d=3, scale=1):
        r = unc.loc[(model, metric)]
        return f"{scale * r.estimate:.{d}f} [{scale * r.ci_low:.{d}f}, {scale * r.ci_high:.{d}f}]"

    rows = [f"{name} & {ci(m, 'precision')} & {ci(m, 'recall')} & {ci(m, 'f1')} & {ci(m, 'fpr', 2, 100)} \\\\"
            for m, name in [("baseline", "Baseline (TF-IDF + LR)"), ("automl", "AutoML v0.1 best")]]
    rows.append(r"\midrule")
    # 5 decimals for F1: its upper bound is -0.00004 and would print as "-0.0000"
    rows.append(f"AutoML $-$ baseline (paired) & {ci('diff', 'precision', 4)} & {ci('diff', 'recall', 4)} & {ci('diff', 'f1', 5)} & {ci('diff', 'fpr', 2, 100)} \\\\")
    table("tab_uncertainty", "lllll", r"Hold-out test & Precision & Recall & F1 & FPR (\%) \\", rows)

    def p_value(metric):
        p = unc.loc[("diff", metric), "p_value"]
        return f"{p:.3f}" if p >= 0.001 else "<0.001"

    (OUT / "val_uncertainty.tex").write_text(macros({"pDiffFone": p_value("f1"), "pDiffFPR": p_value("fpr"), "pDiffRecall": p_value("recall")}))

    loso_ci = pd.read_csv(cfg.TAB / "uncertainty_loso.csv")
    rows = []
    for src in cfg.SOURCES:
        cells = []
        for m in ("baseline", "automl"):
            r = loso_ci[(loso_ci.model == m) & (loso_ci.held_out_source == src)].iloc[0]
            cells.append(f"{r.recall:.3f} [{r.recall_ci_low:.3f}, {r.recall_ci_high:.3f}]")
            cells.append("--" if pd.isna(r.get("fpr")) else f"{100 * r.fpr:.1f} [{100 * r.fpr_ci_low:.1f}, {100 * r.fpr_ci_high:.1f}]")
        rows.append(f"{esc(src)} & " + " & ".join(cells) + r" \\")
    table("tab_uncertainty_loso", "lllll", r"""& \multicolumn{2}{c}{Baseline} & \multicolumn{2}{c}{AutoML v0.1} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Held-out source & Recall & FPR (\%) & Recall & FPR (\%) \\""", rows)
print("done")
