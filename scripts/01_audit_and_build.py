"""Audit of the raw data, EDA figures 1-4, and the clean corpus.

Writes data/processed/corpus.parquet (one mail per near-duplicate cluster) and
corpus_exact_dedup.parquet (only exact copies removed, used in 04 to measure
what near-duplicates do to the score).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from phish import config as cfg
from phish import viz
from phish.data import build_text, load_sources, near_duplicate_clusters

viz.setup()
df = load_sources()
print("raw rows:", len(df))
by_src = df.groupby("source")

# --- class mix per source
comp = by_src["label"].agg(n="size", n_malicious="sum").reindex(cfg.SOURCES)
comp["n_legit"] = comp.n - comp.n_malicious
comp["pct_malicious"] = (100 * comp.n_malicious / comp.n).round(1)
comp.loc["TOTAL"] = [comp.n.sum(), comp.n_malicious.sum(), comp.n_legit.sum(),
                     round(100 * comp.n_malicious.sum() / comp.n.sum(), 1)]
comp.to_csv(cfg.TAB / "audit_composition.csv")
print(comp)

# --- missing values (subject/body were filled with "", so missing = empty)
miss = pd.DataFrame({c: by_src[c].apply(lambda s: s.isna().mean() * 100) for c in ["sender", "receiver", "date", "urls"]})
for c in ["subject", "body"]:
    miss[c] = by_src[c].apply(lambda s: (s.str.strip() == "").mean() * 100)
miss = miss.reindex(cfg.SOURCES).round(1)
miss.to_csv(cfg.TAB / "audit_missingness_pct.csv")
print(miss)

# --- style differences between corpora
body = by_src["body"]
style = pd.DataFrame({
    "median_chars": body.apply(lambda s: s.str.len().median()),
    "max_chars": body.apply(lambda s: s.str.len().max()),
    "pct_truncated": body.apply(lambda s: (s.str.len() > cfg.MAX_CHARS).mean() * 100),
    "uppercase_char_ratio_pct": body.apply(lambda s: 100 * s.str.count(r"[A-Z]").sum() / max(1, s.str.len().sum())),
    "pct_with_http": body.apply(lambda s: s.str.contains("http", case=False).mean() * 100),
    "pct_spaced_punct": body.apply(lambda s: s.str.contains(r" [.,:] ").mean() * 100),
}).reindex(cfg.SOURCES).round(2)

# remove things like "(CET)" at the end of the date, pandas 2 crashes on them
dates = df["date"].str.replace(r"\s*\([^)]*\)\s*$", "", regex=True)
df["year"] = pd.to_datetime(dates, errors="coerce", utc=True, format="mixed").dt.year
years = df.groupby("source")["year"].agg(
    year_p05=lambda s: s.quantile(0.05), year_median="median",
    year_p95=lambda s: s.quantile(0.95), pct_unparsed=lambda s: s.isna().mean() * 100,
).reindex(cfg.SOURCES).round(1)
style = style.join(years)
style.to_csv(cfg.TAB / "audit_style_artifacts.csv")
print(style)

# --- exact duplicates
df = build_text(df)
df["exact_dup"] = df.duplicated("norm_hash", keep="first")
groups = df.groupby("norm_hash").agg(n_src=("source", "nunique"), n_lab=("label", "nunique"), n=("label", "size"))
groups = groups[groups.n > 1]
exact = {
    "rows_raw": len(df),
    "extra_copies_exact": int(df.exact_dup.sum()),
    "dup_groups": len(groups),
    "cross_source_groups": int((groups.n_src > 1).sum()),
    "label_conflict_groups": int((groups.n_lab > 1).sum()),
}
print("exact duplicates:", exact)

# --- near duplicates
uniq = df[~df.exact_dup].reset_index()
print("computing near-duplicate clusters on", len(uniq), "rows ...")
uniq["cluster"] = near_duplicate_clusters(uniq["raw_text"])
clusters = uniq.groupby("cluster").agg(n=("label", "size"), n_lab=("label", "nunique"), n_src=("source", "nunique"))
big = clusters[clusters.n > 1]
near = {
    "rows_after_exact": len(uniq),
    "clusters": int(len(clusters)),
    "rows_in_multi_clusters": int(big.n.sum()),
    "multi_clusters": int(len(big)),
    "cross_source_clusters": int((big.n_src > 1).sum()),
    "label_conflict_clusters": int((big.n_lab > 1).sum()),
    "largest_clusters": str(big.n.sort_values(ascending=False).head(5).tolist()),
}
print("near duplicates:", near)
pd.Series({**exact, **near}).to_csv(cfg.TAB / "audit_duplicates.csv", header=["value"])

top = []
for k in clusters.n.sort_values(ascending=False).head(6).index:
    c = uniq[uniq.cluster == k]
    top.append({"size": len(c), "source": c.source.mode()[0], "label": int(c.label.mode()[0]),
                "subject_example": c.subject.iloc[0][:60]})
pd.DataFrame(top).to_csv(cfg.TAB / "audit_top_clusters.csv", index=False)
print(pd.DataFrame(top))

dedup = pd.DataFrame({
    "raw": df.groupby("source").size(),
    "after_exact": uniq.groupby("source").size(),
    "after_near": uniq.drop_duplicates("cluster").groupby("source").size(),
}).reindex(cfg.SOURCES)
dedup["pct_removed"] = (100 * (1 - dedup.after_near / dedup.raw)).round(1)
dedup.to_csv(cfg.TAB / "audit_dedup_by_source.csv")
print(dedup)

# --- save the corpora: keep the first mail of each cluster, drop empty texts
cols = ["source", "row_in_source", "label", "subject", "raw_text", "text_min", "text_mask", "cluster", "length_chars", "year"]
uniq["length_chars"] = uniq.raw_text.str.len()
corpus = uniq.drop_duplicates("cluster", keep="first")
corpus = corpus[corpus.text_min.str.len() > 0].reset_index(drop=True)
corpus[cols].to_parquet(cfg.PROCESSED / "corpus.parquet")
uniq[uniq.text_min.str.len() > 0][cols].reset_index(drop=True).to_parquet(cfg.PROCESSED / "corpus_exact_dedup.parquet")

final = corpus.groupby("source").label.agg(["size", "sum"]).reindex(cfg.SOURCES)
final.columns = ["n", "n_malicious"]
final["pct_malicious"] = (100 * final.n_malicious / final.n).round(1)
final.loc["TOTAL"] = [final.n.sum(), final.n_malicious.sum(), round(100 * final.n_malicious.sum() / final.n.sum(), 1)]
final.to_csv(cfg.TAB / "final_corpus_composition.csv")
print("final corpus:\n", final)

# --- fig 1: class mix per source
fig, ax = plt.subplots(figsize=(7, 3.2))
ypos = np.arange(len(cfg.SOURCES))
legit = comp.loc[cfg.SOURCES, "n_legit"].values.astype(int)
mal = comp.loc[cfg.SOURCES, "n_malicious"].values.astype(int)
ax.barh(ypos, legit, color=viz.LEGIT, height=0.6, label="Legitimate (0)", edgecolor="white", linewidth=1)
ax.barh(ypos, mal, left=legit, color=viz.MALICIOUS, height=0.6, label="Spam / phishing (1)", edgecolor="white", linewidth=1)
for i, (l, m) in enumerate(zip(legit, mal)):
    ax.text(l + m + 400, i, f"{l + m:,}  ({100 * m / (l + m):.0f}% malicious)", va="center", fontsize=8, color=viz.INK2)
ax.set_yticks(ypos, cfg.SOURCES)
ax.invert_yaxis()
ax.set_xlabel("Number of e-mails (raw)")
ax.set_xlim(0, 52000)
ax.set_title("Class composition per source corpus")
ax.legend(loc="lower right")
ax.grid(axis="y", visible=False)
fig.savefig(cfg.FIG / "fig1_composition.png")
plt.close(fig)

# --- fig 2: length per source and class
fig, ax = plt.subplots(figsize=(7, 3.4))
data, pos, colors = [], [], []
for i, s in enumerate(cfg.SOURCES):
    for j in (0, 1):
        v = corpus[(corpus.source == s) & (corpus.label == j)].length_chars.clip(lower=1)
        if len(v):
            data.append(np.log10(v))
            pos.append(i * 3 + j)
            colors.append(viz.LEGIT if j == 0 else viz.MALICIOUS)
bp = ax.boxplot(data, positions=pos, widths=0.75, patch_artist=True, showfliers=False,
                medianprops=dict(color=viz.INK, linewidth=1.2))
for box, c in zip(bp["boxes"], colors):
    box.set_facecolor(c)
    box.set_edgecolor(c)
    box.set_alpha(0.85)
ax.set_xticks([i * 3 + 0.5 for i in range(len(cfg.SOURCES))], cfg.SOURCES)
ax.set_ylabel("log10(characters)")
ax.legend(handles=[Patch(color=viz.LEGIT, label="Legitimate"), Patch(color=viz.MALICIOUS, label="Spam / phishing")],
          loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
ax.set_title("E-mail length by source and class (deduplicated corpus)", pad=26)
ax.grid(axis="x", visible=False)
fig.savefig(cfg.FIG / "fig2_length.png")
plt.close(fig)

# --- fig 3: missing values
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
fig.savefig(cfg.FIG / "fig3_missingness.png")
plt.close(fig)

# --- fig 4: years per source (Enron and Ling have no date)
fig, ax = plt.subplots(figsize=(7, 2.8))
dated = ["CEAS_08", "Nazario", "Nigerian_Fraud", "SpamAssasin"]
for i, s in enumerate(dated):
    y = df[(df.source == s) & df.year.between(1995, 2023)].year
    jitter_x = np.random.default_rng(0).uniform(-0.3, 0.3, len(y))
    jitter_y = np.random.default_rng(1).uniform(-0.25, 0.25, len(y))
    ax.scatter(y + jitter_x, i + jitter_y, s=2, alpha=0.15, color=viz.COLORS[i], rasterized=True)
ax.set_yticks(range(len(dated)), dated)
ax.set_xlabel("Year of the 'date' header (Enron and Ling have no date field)")
ax.set_title("Collection period per source: up to 20 years apart")
ax.grid(axis="y", visible=False)
fig.savefig(cfg.FIG / "fig4_years.png")
plt.close(fig)
print("done.")
