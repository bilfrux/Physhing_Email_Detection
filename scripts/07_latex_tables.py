"""Generate the LaTeX tables of the report directly from the result CSVs
(no number is typed by hand -> report and code cannot drift apart)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json

import numpy as np
import pandas as pd

from phish import config as C

OUT = C.ROOT / "reports" / "latex" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("->", r"$\rightarrow$").replace("&", r"\&")


def f(x, d=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def pct(x, d=1):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}"


def write(name, body):
    (OUT / f"{name}.tex").write_text(body)
    print("wrote", name)


# 1 - composition + dedup
comp = pd.read_csv(C.TAB / "audit_composition.csv", index_col=0)
dd = pd.read_csv(C.TAB / "audit_dedup_by_source.csv", index_col=0)
fc = pd.read_csv(C.TAB / "final_corpus_composition.csv", index_col=0)
rows = []
for s in C.SOURCES + ["TOTAL"]:
    raw = int(comp.loc[s, "n"])
    after = int(fc.loc[s, "n"])
    rows.append(f"{esc(s)} & {raw:,} & {comp.loc[s, 'pct_malicious']:.1f} & {after:,} & {100 * (1 - after / raw):.1f} & {fc.loc[s, 'pct_malicious']:.1f} \\\\")
rows.insert(-1, r"\midrule")
write("tab_composition", r"""\begin{tabular}{lrrrrr}
\toprule
Source & Raw e-mails & \% malicious & After dedup & \% removed & \% malicious \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# 2 - missingness
miss = pd.read_csv(C.TAB / "audit_missingness_pct.csv", index_col=0)
rows = [f"{esc(s)} & " + " & ".join(f"{miss.loc[s, c]:.1f}" for c in miss.columns) + r" \\" for s in C.SOURCES]
write("tab_missing", r"""\begin{tabular}{l""" + "r" * len(miss.columns) + r"""}
\toprule
Source & """ + " & ".join(esc(c) for c in miss.columns) + r""" \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# 3 - style artefacts
st = pd.read_csv(C.TAB / "audit_style_artifacts.csv", index_col=0)
rows = []
for s in C.SOURCES:
    r = st.loc[s]
    yr = "--" if np.isnan(r.year_median) else f"{int(r.year_p05)}--{int(r.year_p95)}"
    rows.append(f"{esc(s)} & {int(r.median_chars):,} & {int(r.max_chars):,} & {r.uppercase_char_ratio_pct:.1f} & {r.pct_spaced_punct:.1f} & {r.pct_with_http:.1f} & {yr} \\\\")
write("tab_style", r"""\begin{tabular}{lrrrrrc}
\toprule
Source & Median chars & Max chars & \% upper-case chars & \% with ``\,.\,'' spacing & \% with http & Years (p5--p95) \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# 4 - shortcut test
sc = pd.read_csv(C.TAB / "shortcut_source_prediction.csv")
lab = pd.read_csv(C.TAB / "shortcut_label_from_source.csv", index_col=0)["value"]
rows = [f"{esc(r.input)} & {pct(r.source_accuracy)} & {pct(r.source_macro_f1)} \\\\" for r in sc.itertuples()]
write("tab_shortcut", r"""\begin{tabular}{lrr}
\toprule
Input used to predict the \emph{source} (6 classes) & Accuracy (\%) & Macro-F1 (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")
(OUT / "val_label_from_source.tex").write_text(f"{100 * lab['accuracy_label_from_source_only']:.1f}")

top = pd.read_csv(C.TAB / "shortcut_top_tokens_per_source.csv", index_col=0).iloc[:, 0]
short = lambda t: t if len(t) <= 16 else t[:14] + "..."
rows = [f"{esc(s)} & \\texttt{{{esc(', '.join(short(t) for t in top[s].split(', ')[:9]))}}} \\\\" for s in C.SOURCES]
write("tab_top_tokens", r"""\begin{tabular}{lp{0.72\linewidth}}
\toprule
Source & Most source-identifying tokens after masking \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# 5 - split
sp = pd.read_csv(C.TAB / "split_composition.csv", header=[0, 1], index_col=0)
rows = []
for s in C.SOURCES + ["TOTAL"]:
    r = sp.loc[s]
    rows.append(f"{esc(s)} & {int(r[('n', 'train')]):,} & {int(r[('n', 'test')]):,} & {r[('pct_malicious', 'train')]:.1f} & {r[('pct_malicious', 'test')]:.1f} \\\\")
rows.insert(-1, r"\midrule")
write("tab_split", r"""\begin{tabular}{lrrrr}
\toprule
Source & Train & Test & \% malicious (train) & \% malicious (test) \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# 6 - baseline hold-out
B = json.load(open(C.TAB / "baseline_results.json"))
R = B["results"]
def line(name, r):
    return f"{name} & {f(r.get('precision'))} & {f(r.get('recall'))} & {f(r.get('f1'))} & {pct(r.get('fpr'), 2)} & {f(r.get('roc_auc'), 4)} & {f(r.get('precision_at_5pct'))} \\\\"
write("tab_baseline", r"""\begin{tabular}{lrrrrrr}
\toprule
Model (evaluated on) & Precision & Recall & F1 & FPR (\%) & ROC-AUC & Prec.\ @5\% \\
\midrule
""" + "\n".join([
    line("Majority class (test)", R["dummy_most_frequent"]),
    line(f"TF-IDF + LR, C={B['best_C']:g} (train)", R["baseline_tfidf_lr_train"]),
    line(f"TF-IDF + LR, C={B['best_C']:g} (test)", R["baseline_tfidf_lr_holdout"]),
]) + "\n\\bottomrule\n\\end{tabular}")
(OUT / "val_best_C.tex").write_text(f"{B['best_C']:g}")

inf = B["inflation"]
rows = [f"{esc(k)} & {v['clusters_crossing_split']:,} & {f(v['f1'], 4)} & {pct(v['fpr'], 2)} & {f(v['precision_at_5pct'])} \\\\" for k, v in inf.items()]
write("tab_inflation", r"""\begin{tabular}{lrrrr}
\toprule
Split of the exact-dedup corpus & Clusters crossing & F1 & FPR (\%) & Prec.\ @5\% \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# 7 - LOSO vs SOTA
sm = pd.read_csv(C.TAB / "summary_baseline_vs_sota.csv")
rows = []
for r in sm.itertuples():
    rows.append(f"{esc(r.evaluation)} & {f(r.precision)} & {f(r.recall)} & {f(r.f1)} & {pct(r.fpr)} & {f(r.precision_at_5pct)} & {f(getattr(r, 'sota_f1', np.nan))} & {pct(getattr(r, 'sota_fpr', np.nan))} \\\\")
rows.insert(1, r"\midrule")
write("tab_loso", r"""\begin{tabular}{lrrrrrrr}
\toprule
 & \multicolumn{5}{c}{Ours: TF-IDF + LR} & \multicolumn{2}{c}{Bhuiyan \& Bhuiyan 2026} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-8}
Evaluation & Prec. & Recall & F1 & FPR (\%) & Prec.\ @5\% & F1 & FPR (\%) \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# 8 - AutoML
if (C.TAB / "automl_best_results.json").exists():
    names = {"logreg": "Logistic regression", "linear_svm": "Linear SVM", "complement_nb": "Complement NB", "sgd_huber": "SGD (mod.\\ Huber)", "ridge_lsq": "Ridge least-squares", "decision_tree": "Decision tree"}
    tr = pd.read_csv(C.TAB / "automl_trials.csv")
    A = json.load(open(C.TAB / "automl_best_results.json"))
    fam = tr.groupby("params_model").agg(n=("number", "size"), completed=("state", lambda s: (s == "COMPLETE").sum()),
                                        best=("value", lambda v: v[tr.loc[v.index, "state"] == "COMPLETE"].max()))
    fam = fam.sort_values("best", ascending=False)
    rows = [f"{names[k]} & {int(r.n)} & {int(r.completed)} & {f(r.best, 4)} \\\\" for k, r in fam.iterrows()]
    write("tab_automl_families", r"""\begin{tabular}{lrrr}
\toprule
Model family & Trials & Completed & Best CV F1 \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

    def cfg(p):
        s = [("masked" if p["text"] == "text_mask" else "minimal cleaning"), f"word 1--{p['ngram_max']}-grams", f"min\\_df={p['min_df']}", f"{p['max_features'] // 1000}k features"]
        s.append("+ char 3--5-grams" if p["use_char"] else "no char n-grams")
        s.append("+ hand-crafted stats" if p["use_stats"] else "no stats")
        s.append("no selection" if p["select_frac"] >= 1 else f"$\\chi^2$ keep {int(100 * p['select_frac'])}\\%")
        hp = {k: v for k, v in p.items() if k in ("C", "alpha", "max_depth", "min_samples_leaf")}
        s += [f"{'$\\alpha$' if k == 'alpha' else k}={v:.3g}" for k, v in hp.items()]
        return ", ".join(s)
    (OUT / "val_automl_cfg.tex").write_text(cfg(A["config"]))
    b = json.load(open(C.TAB / "baseline_results.json"))
    H, L = b["results"]["baseline_tfidf_lr_holdout"], b["loso"]
    def line(name, r):
        return f"{name} & {f(r.get('precision'))} & {f(r.get('recall'))} & {f(r.get('f1'))} & {pct(r.get('fpr'))} & {f(r.get('precision_at_5pct'))}"
    rows = []
    for ev, rb, ra in [("Hold-out", H, A["holdout"])] + [(f"LOSO: {esc(s)}", L[s], A["loso"][s]) for s in C.SOURCES]:
        rows.append(line(ev + " -- baseline", rb) + " \\\\")
        rows.append(line(ev + " -- AutoML", ra) + " \\\\ \\addlinespace")
    write("tab_automl_best", r"""\begin{tabular}{lrrrrr}
\toprule
Evaluation -- model & Prec. & Recall & F1 & FPR (\%) & Prec.\ @5\% \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    (OUT / "val_automl.tex").write_text(
        f"\\newcommand{{\\nTrials}}{{{len(tr)}}}\\newcommand{{\\nComplete}}{{{(tr.state == 'COMPLETE').sum()}}}"
        f"\\newcommand{{\\nPruned}}{{{(tr.state == 'PRUNED').sum()}}}\\newcommand{{\\bestFamily}}{{{names[A['config']['model']]}}}"
        f"\\newcommand{{\\bestCV}}{{{A['cv_f1']:.4f}}}\\newcommand{{\\bestTrial}}{{{A['trial']}}}"
    )

# 10 - every number quoted in the prose (main.tex, automl_discussion.tex) -> macros, so the text
#      cannot drift from the results when the pipeline is re-run.
SOTA = {"CEAS_08": {"f1": 0.8994, "fpr": 0.1137}, "Enron": {"f1": 0.8201, "fpr": 0.2266}, "Ling": {"f1": 0.8368, "fpr": 0.0512}, "SpamAssasin": {"f1": 0.8433, "fpr": 0.1269}}  # same as 06
Bres = json.load(open(C.TAB / "baseline_results.json"))
BH, BL = Bres["results"]["baseline_tfidf_lr_holdout"], Bres["loso"]
binl = [BL[s] for s in C.BINARY_SOURCES]
vc = pd.read_csv(C.TAB / "baseline_validation_curve.csv", index_col=0)
mac = {
    "baseFone": f(BH["f1"]), "baseFPR": pct(BH["fpr"]), "basePrecFive": f(BH["precision_at_5pct"]), "baseBrier": f(BH["brier"]),
    "losoFoneMin": f(min(r["f1"] for r in binl), 2), "losoFoneMax": f(max(r["f1"] for r in binl), 2),
    "losoFPRmin": pct(min(r["fpr"] for r in binl)), "losoFPRmax": pct(max(r["fpr"] for r in binl)),
    "fprRatioMin": f"{min(r['fpr'] for r in binl) / BH['fpr']:.0f}", "fprRatioMax": f"{max(r['fpr'] for r in binl) / BH['fpr']:.0f}",
    "nazRecallPct": f"{100 * BL['Nazario']['recall']:.0f}", "nazRecall": f(BL["Nazario"]["recall"], 2),
    "nigRecallPct": f"{100 * BL['Nigerian_Fraud']['recall']:.0f}",
    "sotaFoneDiffMax": f(max(abs(BL[s]["f1"] - SOTA[s]["f1"]) for s in SOTA), 2),
    "sotaFPRDiffMax": f"{100 * max(abs(BL[s]['fpr'] - SOTA[s]['fpr']) for s in SOTA):.1f}",
    "valCtrainOne": f"{vc.index[vc.train_f1_mean >= 0.99995].min():g}",
    "valArgmaxC": f"{vc.val_f1_mean.idxmax():g}",
}
if (C.TAB / "automl_best_results.json").exists():
    A = json.load(open(C.TAB / "automl_best_results.json"))
    mac.update(autoFone=f(A["holdout"]["f1"]), autoFPR=pct(A["holdout"]["fpr"], 2), autoPrecFive=f(A["holdout"]["precision_at_5pct"]),
               baseFPRtwo=pct(BH["fpr"], 2), autoTrainFone=f(A["train"]["f1"]),
               autoNazRecall=f(A["loso"]["Nazario"]["recall"]), baseNazRecall=f(BL["Nazario"]["recall"]),
               autoFoneWorse=str(sum(A["loso"][s]["f1"] < BL[s]["f1"] for s in C.BINARY_SOURCES)),
               autoFPRworse=", ".join(esc(s) for s in C.BINARY_SOURCES if A["loso"][s]["fpr"] > BL[s]["fpr"]),
               autoFPRbetter=", ".join(esc(s) for s in C.BINARY_SOURCES if A["loso"][s]["fpr"] < BL[s]["fpr"]))
    for s, k in [("CEAS_08", "Ceas"), ("Enron", "Enron"), ("Ling", "Ling"), ("SpamAssasin", "Spam")]:
        mac[f"dFone{k}"] = f"{A['loso'][s]['f1'] - BL[s]['f1']:+.3f}".replace("-", "$-$")
if (C.TAB / "check_loso_nested_C.csv").exists():  # scripts/09_sensitivity_checks.py
    N = pd.read_csv(C.TAB / "check_loso_nested_C.csv").set_index("held_out_source").loc[C.BINARY_SOURCES]
    na = pd.read_csv(C.TAB / "check_nazario_nonascii.csv", index_col=0)
    xs = pd.read_csv(C.TAB / "check_cross_source_dedup.csv", index_col=0)
    non = na.loc[[i for i in na.index if i.startswith("mostly non")][0]]
    asc = na.loc["mostly ASCII"]
    mac.update(
        nestedMaxDFone=f"{(N.f1_nested - N.f1_global).abs().max():.3f}",
        nestedMaxDFPR=f"{100 * (N.fpr_nested - N.fpr_global).abs().max():.1f}",
        nestedCvalues=", ".join(f"{c:g}" for c in sorted(set(pd.read_csv(C.TAB / "check_loso_nested_C.csv").C_nested))),
        nazNonAsciiN=str(int(non.n)), nazNonAsciiPct=f"{100 * non.n / (non.n + asc.n):.1f}",
        nazNonAsciiRecall=f(non.recall, 2), nazAsciiRecall=f(asc.recall, 2),
        nazNonAsciiEffect=f"{100 * (asc.recall - non.recall) * non.n / (non.n + asc.n):.1f}",
        xsSpamLost=str(int(xs.loc["SpamAssasin", "total_lost"])), xsSpamPct=f"{xs.loc['SpamAssasin', 'pct_of_source_after_exact_dedup']:.1f}",
        xsNigLost=str(int(xs.loc["Nigerian_Fraud", "total_lost"])), xsNigPct=f"{xs.loc['Nigerian_Fraud', 'pct_of_source_after_exact_dedup']:.1f}",
    )
(OUT / "val_text.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
print("wrote val_text", mac)

# 9 - uncertainty (scripts/08_uncertainty.py)
if (C.TAB / "uncertainty_holdout.csv").exists():
    U = pd.read_csv(C.TAB / "uncertainty_holdout.csv").set_index(["model", "metric"])
    def ci(model, m, d=3, scale=1):
        r = U.loc[(model, m)]
        return f"{scale * r.estimate:.{d}f} [{scale * r.ci_low:.{d}f}, {scale * r.ci_high:.{d}f}]"
    rows = [f"{lab} & {ci(m, 'precision')} & {ci(m, 'recall')} & {ci(m, 'f1')} & {ci(m, 'fpr', 2, 100)} \\\\"
            for m, lab in [("baseline", "Baseline (TF-IDF + LR)"), ("automl", "AutoML v0.1 best")]]
    rows.append(r"\midrule")
    rows.append(f"AutoML $-$ baseline (paired) & {ci('diff', 'precision')} & {ci('diff', 'recall')} & {ci('diff', 'f1')} & {ci('diff', 'fpr', 2, 100)} \\\\")
    write("tab_uncertainty", r"""\begin{tabular}{lllll}
\toprule
Hold-out test & Precision & Recall & F1 & FPR (\%) \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    pv = lambda m: f"{U.loc[('diff', m), 'p_value']:.3f}" if U.loc[("diff", m), "p_value"] >= 0.001 else "<0.001"
    (OUT / "val_uncertainty.tex").write_text(
        f"\\newcommand{{\\pDiffFone}}{{{pv('f1')}}}\\newcommand{{\\pDiffFPR}}{{{pv('fpr')}}}\\newcommand{{\\pDiffRecall}}{{{pv('recall')}}}"
    )
    L = pd.read_csv(C.TAB / "uncertainty_loso.csv")
    rows = []
    for src in C.SOURCES:
        cells = []
        for m in ("baseline", "automl"):
            r = L[(L.model == m) & (L.held_out_source == src)].iloc[0]
            rec = f"{r.recall:.3f} [{r.recall_ci_low:.3f}, {r.recall_ci_high:.3f}]"
            fpr = "--" if pd.isna(r.get("fpr")) else f"{100 * r.fpr:.1f} [{100 * r.fpr_ci_low:.1f}, {100 * r.fpr_ci_high:.1f}]"
            cells += [rec, fpr]
        rows.append(f"{esc(src)} & " + " & ".join(cells) + r" \\")
    write("tab_uncertainty_loso", r"""\begin{tabular}{lllll}
\toprule
& \multicolumn{2}{c}{Baseline} & \multicolumn{2}{c}{AutoML v0.1} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Held-out source & Recall & FPR (\%) & Recall & FPR (\%) \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
print("done")
