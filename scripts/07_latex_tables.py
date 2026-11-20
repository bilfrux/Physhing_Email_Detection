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
print("done")
