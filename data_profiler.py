"""Quick profile of a labelled e-mail table: class counts, duplicates, empty texts,
and a few simple signals (length, URLs, capitals, urgency words, HTML) per class."""
import re

import matplotlib.pyplot as plt
import numpy as np

URGENT_WORDS = ["urgent", "immediately", "verify", "account", "suspended", "password", "security", "bank",
                "alert", "click", "access", "confirm", "update", "expire", "login", "action required"]
URL = re.compile(r"https?://|www\.", re.I)
HTML = re.compile(r"<[^>]+>")


def add_signals(df, text_col="text"):
    text = df[text_col].fillna("").astype(str)
    out = df.copy()
    out["words"] = text.str.split().str.len()
    out["urls"] = text.str.count(URL)
    out["caps_ratio"] = text.str.count(r"[A-Z]") / text.str.len().clip(lower=1)
    out["urgent"] = text.str.lower().apply(lambda s: sum(w in s for w in URGENT_WORDS))
    out["html_tags"] = text.str.count(HTML)
    return out


def profile(df, text_col="text", label_col="label"):
    d = add_signals(df, text_col)
    n = len(d)
    n_phish = int((d[label_col] == 1).sum())
    per_class = d.groupby(label_col).agg(
        avg_words=("words", "mean"),
        pct_with_url=("urls", lambda s: 100 * (s > 0).mean()),
        caps_pct=("caps_ratio", lambda s: 100 * s.mean()),
        avg_urgent=("urgent", "mean"),
        pct_with_html=("html_tags", lambda s: 100 * (s > 0).mean()),
    )
    stats = {
        "n": n,
        "n_legit": n - n_phish,
        "n_phish": n_phish,
        "pct_phish": 100 * n_phish / n,
        "missing_text": int(df[text_col].isna().sum()),
        "short_texts": int((d["words"] < 5).sum()),
        "exact_duplicates": int(d.duplicated(subset=[text_col]).sum()),
    }
    return stats, per_class


def print_profile(stats, per_class):
    print(f"e-mails: {stats['n']:,}  legit: {stats['n_legit']:,}  phishing: {stats['n_phish']:,} ({stats['pct_phish']:.1f}%)")
    print(f"missing texts: {stats['missing_text']}  texts < 5 words: {stats['short_texts']}  exact duplicates: {stats['exact_duplicates']}")
    print(per_class.rename(index={0: "legit", 1: "phishing"}).round(2).to_string())


def plot_profile(df, stats, per_class, text_col="text", label_col="label", save_path=None):
    d = add_signals(df, text_col)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ["#2b5c8f", "#d95f02"]

    axes[0, 0].bar(["Legit", "Phishing"], [stats["n_legit"], stats["n_phish"]], color=colors)
    axes[0, 0].set_title("Class counts")

    for label, c in zip([0, 1], colors):
        words = d.loc[d[label_col] == label, "words"].clip(lower=1)
        axes[0, 1].hist(np.log10(words), bins=50, alpha=0.6, color=c, label=["Legit", "Phishing"][label])
    axes[0, 1].set_xlabel("log10(number of words)")
    axes[0, 1].set_title("Length")
    axes[0, 1].legend()

    cols = ["pct_with_url", "caps_pct", "avg_urgent", "pct_with_html"]
    x = np.arange(len(cols))
    for i, (label, c) in enumerate(zip([0, 1], colors)):
        axes[1, 0].bar(x + (i - 0.5) * 0.4, per_class.loc[label, cols], 0.4, color=c, label=["Legit", "Phishing"][label])
    axes[1, 0].set_xticks(x, ["% with URL", "% capitals", "urgent words", "% with HTML"])
    axes[1, 0].set_title("Simple signals per class")
    axes[1, 0].legend()

    names = ["missing text", "< 5 words", "exact duplicates"]
    axes[1, 1].barh(names, [stats["missing_text"], stats["short_texts"], stats["exact_duplicates"]], color="#7f7f7f")
    axes[1, 1].set_title("Data quality")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print("saved", save_path)
    return fig
