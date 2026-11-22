# PROMPT_LOG.md: critical AI interactions

Only **high-stakes** interactions are logged (leakage prevention, statistical checks, metric and model
choice), **4 to 6 in total, at least one per student**. Each entry gives the exact prompt, what the AI
produced, and what **the author** verified or changed. Prompts were written in French and are
translated here; the original wording is kept in brackets. Smaller AI-assisted work is disclosed in
the table at the end, so that no AI-generated code is left undeclared.

| ID | Author | Module | Topic | Linked AI_AUDIT items |
|---|---|---|---|---|
| I-1 | Imene | M1 / M2 | From a bare prompt to a constrained, leakage-safe preprocessing plan | #1, #2, #5 |
| I-2 | Imene | M2 / M3 | Shortcut test, baseline, metric choice and calibration | #3, #4 |
| I-3 | Imene | M4 | AutoML v0.1: search space, recovery of a stopped search, reproducibility | #6, #7, #8 |
| F-1 | Florian | _to fill_ | _to fill_ | |
| A-1 | Ambroise | _to fill_ | _to fill_ | |
| D-1 | Dylan | _to fill_ | _to fill_ | |

---

## Imene

### I-1: from a bare prompt to a constrained, leakage-safe preprocessing plan
**Author:** Imene · **Dates:** 2026-11-18 · **Module:** M1 data audit & preprocessing, M2 split

**Iteration 1, bare prompt (week 1):** *"Here are the project guidelines. Give me your honest opinion
on the dataset (Kaggle Phishing Email Dataset, 6 merged corpora)."* [« je veux ton avis sincère dessus
pour le data set »]

**AI output:** three risks: (1) label 1 mixes generic spam and phishing; (2) the model may learn the
**source** rather than the scam (cited Bhuiyan & Bhuiyan 2026: source predictable at 97 %);
(3) duplicates → leakage. It also found that our planned leave-one-source-out angle had already been
published in 2026.

**What we verified:** opened the paper (BDCC 10(7):211, Tables 18–19) and copied its per-corpus LOCO
numbers into `scripts/06_summary_figures.py`. Decision: keep the dataset, cite the paper as the
benchmark, reframe the label as "spam/phishing vs legitimate". The bare prompt gave useful warnings
but no checkable plan, so the next prompt names every check explicitly.

**Iteration 2, constrained prompt (decomposed, as required by the guidelines):**
1. *Load the six per-source CSVs, show dtypes, missingness per column **per source**, class balance per source.*
2. *Is missingness related to other variables? (expect: metadata missing ⇔ source)*
3. *Find exact duplicates after normalisation **and** near-duplicates (MinHash, Jaccard ≥ 0.8); report cross-source duplicates and label conflicts.*
4. *De-duplicate **before** any split; split 80/20 stratified on source × label and **grouped by near-duplicate cluster**; assert no text crosses the split.*
5. *What assumptions are we making, and what could still leak?*

**AI output:** `01_audit_and_build.py`, `02_split.py` with two `assert` leakage checks.

**What we verified:** ran the asserts; checked the CNN-Alerts clusters by hand (real 2008 malware spam,
label 1 is correct); caught the URL-masking bug via per-source rates (AI_AUDIT #1); rejected the
AI's 6-fold LOSO plan because two corpora have no negatives (AI_AUDIT #2) and its use of the
pre-merged CSV (AI_AUDIT #5).

---

### I-2: shortcut test, baseline, metric choice and calibration
**Author:** Imene · **Date:** 2026-11-18 · **Module:** M2 leakage control, M3 baseline & metrics

**Prompt 1, confound test before any classifier:** *"Before training the phishing classifier: train a
classifier that predicts the SOURCE corpus from (a) raw text, (b) lower-cased text, (c) masked text,
(d) 5 style statistics only, with grouped 5-fold CV on the training split. Also compute the label
accuracy reachable from the source identity alone. Then list the top source-identifying tokens."*

**Prompt 2, baseline with named model, assumptions and metrics:** *"Baseline = TF-IDF unigrams +
logistic regression (same family as the SOTA paper so the comparison is fair). Choose C by 5-fold
grouped CV on train only, plot train AND validation F1 vs C, check probability calibration
(reliability diagram, Brier). Metrics: precision, recall, F1, FPR, ROC-AUC, PR-AUC, and precision at a
realistic 5 % prevalence. Never accuracy alone. Then LOSO with recall-only on the positive-only
corpora."*

**AI output:** `03_shortcut_test.py`, `04_baseline.py`.

**What we verified:** source predictability coherent with the paper (97 % vs our 98.7 %); top tokens
(`enron`, `vince`, `kaminski`, `monkey`, `jose`, `zzzzteana`…) inspected by hand: names, mailing-list
tags and anonymisation artefacts, i.e. real confounders, which revealed the underscore bug
(AI_AUDIT #4). On the validation curve, the AI's argmax choice of C sat at the grid boundary with
training F1 = 1.000 → one-standard-error rule (AI_AUDIT #3). LOSO numbers compared with the paper
corpus by corpus.

---

### I-3: AutoML v0.1, recovery of a stopped search, and reproducibility check
**Author:** Imene · **Dates:** 2026-11-18, 2026-11-22 · **Module:** M4 AutoML (+ reproducibility of the whole pipeline)

**Prompt 1, search space and objective:** *"AutoML v0.1 with Optuna: search jointly preprocessing
(masking on/off), features (word n-grams, char n-grams, hand-crafted stats scaled with MaxAbs, chi2
selection) and models (LR, linear SVM, complement NB, SGD, ridge least-squares, decision tree) with
their hyper-parameters. All fitted transforms inside the CV folds. Objective = grouped CV F1 on a
subsample of the TRAINING split only. Then refit the best config of each family and report hold-out
AND LOSO. State the limits of this selection criterion."*

**Prompt 2, recovering a stopped search without inventing results:** *"The AutoML process got stuck;
check whether it still runs, kill it if so, otherwise read how far it got in the log. Keep the best
logged result (0.9827, ridge_lsq) and save it."* [« Garde le meilleur résultat logué (0.9827,
ridge_lsq) et sauvegarde-le »]

**Prompt 3, reproducibility on a second machine:** *"When I run the .sh it does not work."*
[« quand je run le .sh ca ne marche pas »], then *"We have to do week 2."* [« on doit faire la
week 2 »] with the Project Brief & Governance Policy attached.

**AI output:** `src/phish/features.py`, `05_automl_v01.py`; `05b_recover_best.py` (replay of the
sampler + re-evaluation) and `05c_eval_best.py`; on 2026-11-22: diagnosis of three blocking problems
(raw CSVs not in `data/raw/`, date parsing crashing under pandas 2.2, `requirements.txt` with
merge-conflict markers), a `.venv` built from the pinned versions, a portable fallback in 05b,
`08_uncertainty.py` (cluster-bootstrap CIs, paired AutoML − baseline test, Wilson CIs on LOSO), and a
rewritten `make_submission_zip.sh`.

**What we verified:**
* the same `build_features()` object is used in search and refit; the hand-crafted features exclude
  case and punctuation spacing (shown to be source artefacts);
* the replay reproduces the model family of all 34 logged trials, and the recomputed CV F1 of the top
  3 configurations equals the logged value (0.9827) before anything was saved;
* SHA-256 of the raw files against `data/README.md`; the full `run_all.sh` re-run in the pinned
  environment, with `git diff` of the regenerated tables against the committed ones;
* `08_uncertainty.py` asserts that its recomputed point estimates equal the saved F1 / FPR / recall
  before computing any interval.

Problems caught: unbounded search with a diverging linear SVM (AI_AUDIT #6), version-dependent date
parsing and a wrong `pip` diagnosis (AI_AUDIT #7), a recovery step that was not reproducible on
another machine (AI_AUDIT #8). Limit stated: the selection objective is in-distribution → v0.2
nested LOSO.

---

## Florian

### F-1: _title_
> TODO (Florian): one high-stakes interaction **you** had with an AI for your module. Same format:
> date, module, **exact prompt** (original wording in brackets if in French), what the AI produced,
> what **you** checked or changed, and the AI_AUDIT item it is linked to. Do not copy another
> member's entry.

**Author:** Florian · **Date:** · **Module:**

**Prompt:**

**AI output:**

**What I verified:**

---

## Ambroise

### A-1: _title_
> TODO (Ambroise): same format as above. Your scripts `metrics_evaluation.py`, `data_profiler.py` and
> `demo_metrics.py` are a natural candidate if an AI helped write them.

**Author:** Ambroise · **Date:** · **Module:**

**Prompt:**

**AI output:**

**What I verified:**

---

## Dylan

### D-1: _title_
> TODO (Dylan): same format as above.

**Author:** Dylan · **Date:** · **Module:**

**Prompt:**

**AI output:**

**What I verified:**

---

## Disclosure of other AI-assisted work (not high-stakes, listed for completeness)

| File / change | AI contribution | Human check |
|---|---|---|
| `scripts/06_summary_figures.py`, `07_latex_tables.py`, `src/phish/viz.py` | figure and LaTeX-table boilerplate | figures read, axes and units checked; tables compared with the CSVs |
| `scripts/01_audit_and_build.py` (2026-11-22) | date-parsing fix; behaviour-preserving simplification (one `groupby`, dead code removed) | 01–03 re-run in an isolated copy: all 11 output tables byte-identical to the previous run |
| `scripts/03_shortcut_test.py`, `04_baseline.py`, `src/phish/features.py` (2026-11-22) | `random_state` added to liblinear models (good practice; no measurable effect: 03 tables byte-identical with and without it) | re-run of 03 in an isolated copy |
| `requirements.txt`, `.gitignore`, `run_all.sh` (2026-11-22) | conflict markers removed, ignores, pipeline order | `pip install -r` in a clean `.venv` |
| `ai-governance/CONTRIBUTIONS.md` (2026-11-22) | draft of the module split | module owners chosen by the team |
| `src/phish/text.py` masking prototype (AI_AUDIT #9, not applied) | proposed v0.2 masking | per-source parity measurements |
| `scripts/09_sensitivity_checks.py` (2026-11-22) | nested-C LOSO, non-ASCII and cross-source checks (AI_AUDIT #10 and null results) | numbers read and compared with `baseline_results.json` |
| `scripts/07_latex_tables.py` text macros, report and README rewording (2026-11-22) | every number of the prose generated from result files (AI_AUDIT #12) | static check: all 51 macros defined; sentences re-read against the tables |
| `reports/latex/main.tex`, `automl_discussion.tex` (2026-11-22) | report shortened (~20 %) and rewritten in plainer English; content, tables, figures and numbers unchanged | labels, references, citations and macros checked; every sentence re-read by the team before submission |
| `demo_metrics.py` (2026-11-22) | frozen split, no synthetic fallback, fake attack removed (AI_AUDIT #11) | old and new versions run side by side |
**What we verified:** the replay reproduces the model family of all 34 logged trials, and the
re-computed CV F1 of the top 3 configurations equals the logged values (0.9827). Only after this
check was the configuration saved and evaluated.

---

## visualisation data
**Auteur**: Ambroise
**promt:** fait moi un code qui permet de faire de la visualisation de données pour m'aider à choisir un modelèle. Pour cela j'ai choisis les metrique et fait moi la matrice de confusion, le f1 score, le nombre de doublons, le nombre d'email de phising et réel, le nombre de valeurs manquant
**output**  le code python qui m'affiche des tableaux:
📋 TABLEAU COMPARATIF :
                        Model Name  Precision  Recall (TPR)  F1-Score  F2-Score      MCC    PR-AUC   ROC-AUC  FPR @ 95% Recall
      Baseline (Al-Subaiey et al.)     0.8412        0.9350    0.8856    0.9146   0.8640    0.9577    0.9899            0.0459
   Cross-Corpus (Test sur CEAS_08)     0.5203        0.5944    0.5549    0.5780   0.4638    0.5804    0.8634            0.5086
Sous Attaque (Adversarial Evasion)     0.7605        0.5604    0.6453    0.5915   0.5982    0.7064    0.8779            0.3942

