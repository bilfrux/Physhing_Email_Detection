# PROMPT_LOG.md: critical AI interactions

Only high-stakes interactions are logged (leakage prevention, statistical checks, model and metric
choice). For each: the exact prompt, what the AI produced, and what **we** verified or changed.
Prompts were written in French and are translated here; the original wording is kept in brackets.

> TODO (team): the prompts below are the ones used by Imene. Every member adds the prompts *they*
> used for their own part in the same format (date, author, prompt, verification).

---

## P1 (week 1): dataset choice and honest critique
**Author:** Imene · **Date:** 2026-11-18

**Prompt:** *"Here are the project guidelines. Give me your honest opinion on the dataset
(Kaggle Phishing Email Dataset, 6 merged corpora)."* [« je veux ton avis sincère dessus pour le data set »]

**AI output:** three risks: (1) label 1 mixes generic spam and phishing; (2) the model may learn the
**source** rather than the scam (cited Bhuiyan & Bhuiyan 2026: source predictable at 97 %);
(3) duplicates → leakage. It also found that our planned leave-one-source-out angle had already been
published in 2026.

**What we verified:** opened the paper (BDCC 10(7):211, Tables 18–19) and copied its per-corpus
LOCO numbers into `scripts/06_summary_figures.py`; reproduced the source-predictability claim
ourselves (`03_shortcut_test.py`: 98.7 % on raw text). Decision: keep the dataset, cite the paper as
the benchmark, and reframe the label as "spam/phishing vs legitimate".

---

## P2: decomposed analysis prompt (leakage-safe preprocessing)
**Author:** Imene · **Date:** 2026-11-18

**Prompt (decomposed, as required by the guidelines):**
1. *Load the six per-source CSVs, show dtypes, missingness per column **per source**, class balance per source.*
2. *Is missingness related to other variables? (expect: metadata missing ⇔ source)*
3. *Find exact duplicates after normalisation **and** near-duplicates (MinHash, Jaccard ≥ 0.8); report cross-source duplicates and label conflicts.*
4. *De-duplicate **before** any split; split 80/20 stratified on source × label and **grouped by near-duplicate cluster**; assert no text crosses the split.*
5. *What assumptions are we making, and what could still leak?*

**AI output:** `01_audit_and_build.py`, `02_split.py` with two `assert` leakage checks.

**What we verified:** ran the asserts; checked the CNN-Alerts clusters by hand (they are real 2008
malware spam, label 1 is correct); caught the URL-masking bug (AI_AUDIT #1) via the per-source rates.

---

## P3: shortcut / confound test
**Author:** Imene · **Date:** 2026-11-18

**Prompt:** *"Before training the phishing classifier: train a classifier that predicts the SOURCE
corpus from (a) raw text, (b) lower-cased text, (c) masked text, (d) 5 style statistics only, with
grouped 5-fold CV on the training split. Also compute the label accuracy reachable from the source
identity alone. Then list the top source-identifying tokens."*

**AI output:** `03_shortcut_test.py`.

**What we verified:** results are coherent with the paper (97 % vs our 98.7 %). Top tokens
(`enron`, `vince`, `kaminski`, `monkey`, `jose`, `zzzzteana`…) inspected by hand. They are names,
mailing-list tags and anonymisation artefacts, i.e. real confounders. Caught AI_AUDIT #4.

---

## P4: baseline, metric choice and calibration
**Author:** Imene · **Date:** 2026-11-18

**Prompt:** *"Baseline = TF-IDF unigrams + logistic regression (same family as the SOTA paper so the
comparison is fair). Choose C by 5-fold grouped CV on train only, plot train AND validation F1 vs C,
check probability calibration (reliability diagram, Brier). Metrics: precision, recall, F1, FPR,
ROC-AUC, PR-AUC, and precision at a realistic 5 % prevalence. Never accuracy alone. Then LOSO with
recall-only on the positive-only corpora."*

**AI output:** `04_baseline.py`.

**What we verified:** the argmax-at-boundary problem (AI_AUDIT #3); LOSO numbers vs the paper
(within 0.05 F1 on all four corpora, FPR within ±1.1 pt).

---

## P5: AutoML v0.1 search space and objective
**Author:** Imene · **Date:** 2026-11-18

**Prompt:** *"AutoML v0.1 with Optuna: search jointly preprocessing (masking on/off), features (word
n-grams, char n-grams, hand-crafted stats scaled with MaxAbs, chi2 selection) and models (LR, linear
SVM, complement NB, SGD, ridge least-squares, decision tree) with their hyper-parameters. All fitted
transforms inside the CV folds. Objective = grouped CV F1 on a subsample of the TRAINING split only.
Then refit the best config of each family and report hold-out AND LOSO. State the limits of this
selection criterion."*

**AI output:** `src/phish/features.py`, `05_automl_v01.py`.

**What we verified:** that the exact same `build_features()` object is used in search and refit;
that the hand-crafted features exclude case and punctuation spacing (shown to be source artefacts);
the limitation that the selection objective is in-distribution (→ v0.2: nested LOSO).

---

## P6: recovering a stopped search without inventing results
**Author:** Imene · **Date:** 2026-11-18

**Prompt:** *"The AutoML process got stuck; check whether it still runs, kill it if so, otherwise read
how far it got in the log. Keep the best logged result (0.9827, ridge_lsq) and save it."*
[« Garde le meilleur résultat logué (0.9827, ridge_lsq) et sauvegarde-le »]

**AI output:** `05b_recover_best.py` (replay of the sampler + re-evaluation), `05c_eval_best.py`.

**What we verified:** the replay reproduces the model family of all 34 logged trials, and the
re-computed CV F1 of the top 3 configurations equals the logged values (0.9827). Only after this
check was the configuration saved and evaluated.
