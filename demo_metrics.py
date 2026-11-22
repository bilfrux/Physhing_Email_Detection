"""Demo: profile the clean corpus, train two simple models on the frozen split and
compare them with metrics_evaluation.py. Run `bash run_all.sh` first (needs 01 and 02).

The old version used a random split on the raw CSV, filled bad labels with 0, made
up data when the CSV was missing and faked an "attack" result (AI_AUDIT #11).
"""
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from data_profiler import plot_profile, print_profile, profile
from metrics_evaluation import evaluate, plot_comparison, plot_dashboard

root = Path(__file__).resolve().parent
corpus, split = root / "data/processed/corpus.parquet", root / "data/processed/split.csv"
if not (corpus.exists() and split.exists()):
    raise SystemExit("data/processed/ not found: run bash run_all.sh first (steps 01 and 02)")
df = pd.read_parquet(corpus, columns=["source", "label", "raw_text", "text_min"])
df["split"] = pd.read_csv(split)["split"].values
assert df["label"].isin([0, 1]).all(), "labels must be 0/1"
train, test = df[df.split == "train"], df[df.split == "test"]

stats, per_class = profile(df, text_col="raw_text")  # raw text: URLs, capitals and HTML are still there
print_profile(stats, per_class)
plot_profile(df, stats, per_class, text_col="raw_text", save_path="data_profiling_and_bias.png")

tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
X_train = tfidf.fit_transform(train.text_min)
X_test = tfidf.transform(test.text_min)

lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42).fit(X_train, train.label)
svm = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced", random_state=42), cv=3).fit(X_train, train.label)
p_lr = lr.predict_proba(X_test)[:, 1]
p_svm = svm.predict_proba(X_test)[:, 1]

plot_dashboard(test.label, p_svm, "Linear SVM (calibrated)", save_path="phishing_evaluation_dashboard.png")
results = [evaluate(test.label, p_lr, "Logistic regression"), evaluate(test.label, p_svm, "Linear SVM (calibrated)")]
plot_comparison(results, save_path="model_comparison_benchmark.png")

cols = ["model", "precision", "recall", "f1", "f0.5", "f2", "mcc", "pr_auc", "roc_auc", "fpr_at_95_recall"]
print(pd.DataFrame(results)[cols].round(4).to_string(index=False))
