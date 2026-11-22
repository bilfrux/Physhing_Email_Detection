"""Step 2.1 - raw data audit + construction of the clean corpus.

Outputs
  reports/tables/audit_*.csv          per-source quality statistics
  reports/figures/fig1..fig4          EDA figures
  data/processed/corpus.parquet              final de-duplicated corpus used for modelling
  data/processed/corpus_exact_dedup.parquet  exact-dedup only (to measure near-duplicate inflation)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from phish import config as C
from phish import viz
from phish.data import build_text, load_sources, near_duplicate_clusters

viz.setup()
df = load_sources()
print("raw rows:", len(df))

# ---------------------------------------------------------------- 1. composition
comp = (
    df.groupby("source")["label"]
    .agg(n="size", n_malicious="sum")
    .assign(n_legit=lambda t: t.n - t.n_malicious, pct_malicious=lambda t: (100 * t.n_malicious / t.n).round(1))
    .reindex(C.SOURCES)
)
comp.loc["TOTAL"] = [comp.n.sum(), comp.n_malicious.sum(), comp.n_legit.sum(), round(100 * comp.n_malicious.sum() / comp.n.sum(), 1)]
comp.to_csv(C.TAB / "audit_composition.csv")
print(comp)

# ---------------------------------------------------------------- 2. missingness
by_src = df.groupby("source")
miss = pd.DataFrame({c: by_src[c].apply(lambda s: s.isna().mean() * 100) for c in ["sender", "receiver", "date", "urls"]})
# subject/body were filled with "" in load_sources -> missing = empty
for c in ["subject", "body"]:
    miss[c] = by_src[c].apply(lambda s: (s.str.strip() == "").mean() * 100)
miss = miss.reindex(C.SOURCES).round(1)
miss.to_csv(C.TAB / "audit_missingness_pct.csv")
print(miss)

# ---------------------------------------------------------------- 3. style / artefacts
body = by_src["body"]
style = pd.DataFrame(
    {
        "median_chars": body.apply(lambda s: s.str.len().median()),
        "max_chars": body.apply(lambda s: s.str.len().max()),
        "pct_truncated": body.apply(lambda s: (s.str.len() > C.MAX_CHARS).mean() * 100),
        "uppercase_char_ratio_pct": body.apply(lambda s: 100 * s.str.count(r"[A-Z]").sum() / max(1, s.str.len().sum())),
        "pct_with_http": body.apply(lambda s: s.str.contains("http", case=False).mean() * 100),
        "pct_spaced_punct": body.apply(lambda s: s.str.contains(r" [.,:] ").mean() * 100),
    }
).reindex(C.SOURCES).round(2)
# Strip trailing RFC 2822 zone comments like "+0100 (CET)": the numeric offset already carries
# the information, and pandas 2.x crashes on them (parses "CET" as tzlocal).
dates = df["date"].str.replace(r"\s*\([^)]*\)\s*$", "", regex=True)
df["year"] = pd.to_datetime(dates, errors="coerce", utc=True, format="mixed").dt.year
yr = df.groupby("source")["year"].agg(
    year_p05=lambda s: s.quantile(0.05), year_median="median", year_p95=lambda s: s.quantile(0.95), pct_unparsed=lambda s: s.isna().mean() * 100
).reindex(C.SOURCES).round(1)
style = style.join(yr)
style.to_csv(C.TAB / "audit_style_artifacts.csv")
print(style)

# ---------------------------------------------------------------- 4. duplicates
df = build_text(df)
df["exact_dup"] = df.duplicated("norm_hash", keep="first")
grp = df.groupby("norm_hash").agg(n_src=("source", "nunique"), n_lab=("label", "nunique"), n=("label", "size"))
multi = grp[grp.n > 1]
exact = {
    "rows_raw": len(df),
    "extra_copies_exact": int(df.exact_dup.sum()),
    "dup_groups": len(multi),
    "cross_source_groups": int((multi.n_src > 1).sum()),
    "label_conflict_groups": int((multi.n_lab > 1).sum()),
}
print("exact duplicates:", exact)

uniq = df[~df.exact_dup].reset_index()  # keeps original index in column "index"
print("computing near-duplicate clusters on", len(uniq), "rows ...")
uniq["cluster"] = near_duplicate_clusters(uniq["raw_text"])
cs = uniq.groupby("cluster").agg(n=("label", "size"), n_lab=("label", "nunique"), n_src=("source", "nunique"))
big = cs[cs.n > 1]
near = {
    "rows_after_exact": len(uniq),
    "clusters": int(len(cs)),
    "rows_in_multi_clusters": int(big.n.sum()),
    "multi_clusters": int(len(big)),
    "cross_source_clusters": int((big.n_src > 1).sum()),
    "label_conflict_clusters": int((big.n_lab > 1).sum()),
    "largest_clusters": big.n.sort_values(ascending=False).head(5).tolist(),
}
print("near duplicates:", near)
pd.Series({**exact, **{k: str(v) if isinstance(v, list) else v for k, v in near.items()}}).to_csv(
    C.TAB / "audit_duplicates.csv", header=["value"]
)

top = cs.n.sort_values(ascending=False).head(6).index
top_tab = []
for k in top:
    s = uniq[uniq.cluster == k]
    top_tab.append({"size": len(s), "source": s.source.mode()[0], "label": int(s.label.mode()[0]), "subject_example": s.subject.iloc[0][:60]})
pd.DataFrame(top_tab).to_csv(C.TAB / "audit_top_clusters.csv", index=False)
print(pd.DataFrame(top_tab))

dup_by_src = pd.DataFrame(
    {
        "raw": df.groupby("source").size(),
        "after_exact": uniq.groupby("source").size(),
        "after_near": uniq.drop_duplicates("cluster").groupby("source").size(),
    }
).reindex(C.SOURCES)
dup_by_src["pct_removed"] = (100 * (1 - dup_by_src.after_near / dup_by_src.raw)).round(1)
dup_by_src.to_csv(C.TAB / "audit_dedup_by_source.csv")
print(dup_by_src)

# Final corpus: one representative per near-duplicate cluster (first occurrence).
corpus = uniq.drop_duplicates("cluster", keep="first").drop(columns=["index"]).reset_index(drop=True)
corpus = corpus[corpus.text_min.str.len() > 0].reset_index(drop=True)
corpus["length_chars"] = corpus.raw_text.str.len()
keep = ["source", "row_in_source", "label", "subject", "raw_text", "text_min", "text_mask", "cluster", "length_chars", "year"]
corpus[keep].to_parquet(C.PROCESSED / "corpus.parquet")
# Exact-dedup-only version (kept to *measure* the effect of near-duplicates; groups = cluster).
uniq["length_chars"] = uniq.raw_text.str.len()
uniq[uniq.text_min.str.len() > 0][keep].reset_index(drop=True).to_parquet(C.PROCESSED / "corpus_exact_dedup.parquet")
fc = corpus.groupby("source").label.agg(["size", "sum"]).reindex(C.SOURCES)
fc.columns = ["n", "n_malicious"]
fc["pct_malicious"] = (100 * fc.n_malicious / fc.n).round(1)
fc.loc["TOTAL"] = [fc.n.sum(), fc.n_malicious.sum(), round(100 * fc.n_malicious.sum() / fc.n.sum(), 1)]
fc.to_csv(C.TAB / "final_corpus_composition.csv")
print("final corpus:\n", fc)

# ---------------------------------------------------------------- figures
# Fig 1 - composition per source (raw vs deduplicated), stacked horizontal bars
fig, ax = plt.subplots(figsize=(7, 3.2))
ypos = np.arange(len(C.SOURCES))
legit = comp.loc[C.SOURCES, "n_legit"].values.astype(int)
mal = comp.loc[C.SOURCES, "n_malicious"].values.astype(int)
ax.barh(ypos, legit, color=viz.LEGIT, height=0.6, label="Legitimate (0)", edgecolor="white", linewidth=1)
ax.barh(ypos, mal, left=legit, color=viz.MALICIOUS, height=0.6, label="Spam / phishing (1)", edgecolor="white", linewidth=1)
for i, (l, m) in enumerate(zip(legit, mal)):
    ax.text(l + m + 400, i, f"{l + m:,}  ({100 * m / (l + m):.0f}% malicious)", va="center", fontsize=8, color=viz.INK2)
ax.set_yticks(ypos, C.SOURCES)
ax.invert_yaxis()
ax.set_xlabel("Number of e-mails (raw)")
ax.set_xlim(0, 52000)
ax.set_title("Class composition per source corpus")
ax.legend(loc="lower right")
ax.grid(axis="y", visible=False)
fig.savefig(C.FIG / "fig1_composition.png")
plt.close(fig)

# Fig 2 - length distribution by source and class (log scale)
fig, ax = plt.subplots(figsize=(7, 3.4))
data, pos, cols = [], [], []
for i, s in enumerate(C.SOURCES):
    for j, lab in enumerate([0, 1]):
        v = corpus[(corpus.source == s) & (corpus.label == lab)].length_chars.clip(lower=1)
        if len(v):
            data.append(np.log10(v))
            pos.append(i * 3 + j)
            cols.append(viz.LEGIT if lab == 0 else viz.MALICIOUS)
bp = ax.boxplot(data, positions=pos, widths=0.75, patch_artist=True, showfliers=False, medianprops=dict(color=viz.INK, linewidth=1.2))
for p, c in zip(bp["boxes"], cols):
    p.set_facecolor(c)
    p.set_edgecolor(c)
    p.set_alpha(0.85)
ax.set_xticks([i * 3 + 0.5 for i in range(len(C.SOURCES))], C.SOURCES)
ax.set_ylabel("log10(characters)")
ax.legend(handles=[Patch(color=viz.LEGIT, label="Legitimate"), Patch(color=viz.MALICIOUS, label="Spam / phishing")], loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
ax.set_title("E-mail length by source and class (deduplicated corpus)", pad=26)
ax.grid(axis="x", visible=False)
fig.savefig(C.FIG / "fig2_length.png")
plt.close(fig)

# Fig 3 - missingness heatmap (metadata availability is decided by the source)
fig, ax = plt.subplots(figsize=(6, 2.8))
m = miss[["sender", "receiver", "date", "urls", "subject", "body"]]
im = ax.imshow(m.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(m.shape[1]), m.columns)
ax.set_yticks(range(m.shape[0]), m.index)
for i in range(m.shape[0]):
    for j in range(m.shape[1]):
        v = m.values[i, j]
        ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=7.5, color="white" if v > 55 else viz.INK)
ax.grid(False)
ax.set_title("Missing values (%) per field and source")
fig.colorbar(im, ax=ax, fraction=0.03, label="% missing")
fig.savefig(C.FIG / "fig3_missingness.png")
plt.close(fig)

# Fig 4 - year distribution per source (temporal shift)
fig, ax = plt.subplots(figsize=(7, 2.8))
dated = ["CEAS_08", "Nazario", "Nigerian_Fraud", "SpamAssasin"]  # Enron and Ling have no date field
for i, s in enumerate(dated):
    y = df[(df.source == s) & df.year.between(1995, 2023)].year
    ax.scatter(y + np.random.default_rng(0).uniform(-0.3, 0.3, len(y)), np.full(len(y), i) + np.random.default_rng(1).uniform(-0.25, 0.25, len(y)), s=2, alpha=0.15, color=viz.SERIES[i], rasterized=True)
ax.set_yticks(range(len(dated)), dated)
ax.set_xlabel("Year of the 'date' header (Enron and Ling have no date field)")
ax.set_title("Collection period per source: up to 20 years apart")
ax.grid(axis="y", visible=False)
fig.savefig(C.FIG / "fig4_years.png")
plt.close(fig)
print("done.")
