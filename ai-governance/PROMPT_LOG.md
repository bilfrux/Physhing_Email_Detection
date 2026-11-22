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

### A-1: visualisation data
**Author**: Ambroise  **Date:** 2026-11-18

**Prompt:** fait moi un code qui permet de faire de la visualisation de données pour m'aider à choisir un modelèle. Pour cela j'ai choisis les metrique et fait moi la matrice de confusion, le f1 score, le nombre de doublons, le nombre d'email de phising et réel, le nombre de valeurs manquant
**output**  le code python qui m'affiche des tableaux:
📋 TABLEAU COMPARATIF :
                        Model Name  Precision  Recall (TPR)  F1-Score  F2-Score      MCC    PR-AUC   ROC-AUC  FPR @ 95% Recall
      Baseline (Al-Subaiey et al.)     0.8412        0.9350    0.8856    0.9146   0.8640    0.9577    0.9899            0.0459
   Cross-Corpus (Test sur CEAS_08)     0.5203        0.5944    0.5549    0.5780   0.4638    0.5804    0.8634            0.5086
Sous Attaque (Adversarial Evasion)     0.7605        0.5604    0.6453    0.5915   0.5982    0.7064    0.8779            0.3942

### A-2: metrics_evaluation.py

### A-3: data_profiler.py

### A-4: demo_metrics.py

---

## Dylan

### D-1: Data leakage verification 
**Author:** Dylan · **Date:** 2026-11-22

**Prompt:** *"You are an expert data analyst. Within the code, can you catch any data leakage?"*

**AI output:** "I found no direct train-to-test leakage in the main hold-out path (01, 02, 04, 05, 05c). There are several softer leaks and fragilities, mostly in the leave-one-source-out (LOSO) evaluation.

#### What is clean
- **Dedup and split:** Near-duplicate clusters are removed before splitting, and `02_split.py` asserts that no cluster or normalised text crosses the split.
- **Vectorizers:** They are fitted on the training fold only. This holds in 04 (CV, final fit, LOSO) and in 05 (`fold_features` fits on `folds[k][0]`).
- **Model selection:** C in 04 and the Optuna search in 05 use only the training split. Test is touched only after selection.
- **Shortcut test (03):** It uses training data only, with scalers and vectorizers inside the CV pipelines.

#### Flaws analysis

**1. LOSO uses hyperparameters selected on the held-out source (05, 05c, 04)**
- This is the most significant issue. C in 04 and the whole AutoML config in 05 were selected by CV on training data from all six sources.
- Then LOSO claims to measure generalisation to an unseen source, but the held-out source's training-split rows already influenced the choice of model family, n-grams, masking, C and so on.
- Your docstring acknowledges this. In practice it means LOSO numbers are optimistic, particularly for the AutoML model, which has far more selected degrees of freedom than the baseline's single C. So the AutoML-vs-baseline LOSO comparison is biased in favour of AutoML.
- The fix is nested LOSO selection, as planned for v0.2.

**2. The 04 near-duplicate inflation experiment reuses `best_C`**
- `best_C` was chosen using the main train split, which overlaps with the new random-split test rows (`m_te`). It is one scalar in a diagnostic, so the impact is small, but those rows are not truly unseen.

**3. Fragile positional alignment of `split.csv`**
- Every script does `df[split.split.values == "train"]`. That relies on `corpus.parquet` having exactly the same row order as when `02_split.py` ran.
- If 01 is re-run with any change in dedup order, the train/test labels silently misalign and you get real leakage with no error. The assertions in 02 only run at split time.
- Fix: merge on `(source, row_in_source)` and assert the cluster disjointness check in every downstream script.

**4. Dedup guarantees are only as good as the clustering threshold**
- The disjointness assertion covers only the clusters your algorithm found. Near-duplicates below the similarity cutoff (templated spam, forwarded or quoted Enron threads, Nigerian-fraud variants) can still straddle train/test and across sources in LOSO.
- A quick empirical check is to compute nearest-neighbour cosine similarity from each test email to the train set and look at the upper tail.

**5. Selection optimism and post-hoc choices in 05b**
- `cv_f1` is the best of 34 trials chosen on one 20k subsample with 3 folds, so it is biased upward. Don't report it as an estimate of performance.
- The tie-break "in favour of ridge_lsq as recorded" is a manual choice. Make sure it was fixed before looking at any test or LOSO result. Scores tied at 4 decimals are essentially noise-level differences. Also check that run 1's test results did not influence the decision to restart with the modified search space.
- The replay-consistency check compares only the model family, so it is a weak check. The recomputed CV F1 is the real verification.

**6. Repeated use of the test set**
- 05 prints hold-out results for all families, 05c re-evaluates, and 04 shows test diagnostics. Selection is by CV, so nothing is leaking in code. But if you ever pick or tweak anything after seeing those numbers, the test set stops being a clean estimate."

**What we verified:** that data-leakage hasn't escaped our vigilance and slipped into the code.

### D-2: Flaw verification 
**Author:** Dylan · **Date:** 2026-11-22

**Prompt:** *"You are an expert data analyst. Can you catch any major error or flaw with the code?"*

**AI output:** "
#### Major

**1. The AutoML-vs-baseline LOSO comparison is not like-for-like (05, 05c vs 04).**
- The `05c` docstring says "same protocol as the baseline, so directly comparable". The hold-out comparison is fair, because the test split is untouched during selection.
- The LOSO comparison is not fair. The AutoML search chose text variant, n-grams, char features, model family and about 8 other knobs by CV over all six sources. The baseline tuned only one hyper-parameter, C.
- The search includes `text_mask` vs `text_min`, which is exactly the lever against the source shortcut your `03` script measures.
- So the held-out source was seen during selection, and AutoML had far more degrees of freedom to exploit it. Its LOSO numbers will be optimistic relative to the baseline's.
- You document this as a "known limitation", but the `05c` docstring still claims comparability. Either do nested LOSO now, or label the LOSO numbers as non-comparable.

**2. Post-hoc, result-dependent selection in `05b`.**
- `sorted(ok, key=(-cv_f1, model != "ridge_lsq", trial))` hard-codes a preferred family into the tie-break. A tie within 4 decimals of a noisy 3-fold CV is not a real tie. Choosing the winner this way is selection by preference, then presenting it as "best".
- The baseline uses a one-standard-error rule, while AutoML uses argmax of a noisy CV. With 34 trials, 6 families and about 8 dimensions, the top configs are almost certainly within noise (`cv_f1_std` is computed but never used). The reported `cv_f1` is therefore an optimistic maximum. Apply the same 1-SE rule to both.

**3. The "exact" replay is weaker than claimed (`05b`).**
- The replay only checks that the model family matches at each step, not the full params.
- `t.report(value, 1)` hard-codes step 1 for every pruned trial. Pruning can also occur at step 2, and TPE ranks pruned trials by (step, value), so the history you feed it differs from the original.
- Values are the rounded 4-decimal log values, not the originals.
- Divergence could silently change later parameters. Only the top 3 by logged score are re-verified, so a better trial whose params were reconstructed wrongly would be missed.
- If none of the top 3 reproduce, `sorted(ok)[0]` raises an IndexError.

**4. Scores are not comparable across models (`05c`/`05`, check `metrics.py`).**
- The baseline passes `predict_proba`, with a 0.5 threshold, into `full_report`, which also reports Brier and calibration.
- `scores()` for ridge, SVM and Huber-SGD returns decision-function values. A Brier score or reliability diagram on those is meaningless, and a "0.5" cut-off on them is not the same operating point.
- Only AUC, F1 and recall are comparable here. Calibrate these models or drop the Brier comparison.

**5. Label noise from de-duplication (`01`).**
- You count `label_conflict_groups` and `label_conflict_clusters`, then keep the first row anyway. Which label survives depends on file-load order.
- Cross-source duplicates also get attributed to whichever source loads first, which biases per-source composition and LOSO.
- Drop conflicting groups, or vote on the label.

#### Moderate

**6. Single-class LOSO folds.** Nazario is 100% malicious, so AUC is undefined (check whether `full_report` raises or returns NaN) and precision is trivially 1. F1 reduces to recall, so averaging F1 across folds mixes incomparable quantities. Report recall only for such folds, and don't average them.

**7. Hyper-parameter transfer from a 20k subsample to the full training set.** `min_df`, `max_features`, `C` and `alpha` interact with sample size. The best settings at 20k are not necessarily the best at full scale. Re-tune at full size, at least for the winning family.

**8. Fragile alignment.** `split.csv` is joined to `corpus.parquet` purely by position (`split.split.values == "train"`). If `01` is re-run, row order can change and the split silently misaligns. `02`'s leakage asserts only run on its own in-memory frame. Join on `(source, row_in_source)` instead.

**9. Possible model/feature incompatibility (check `features.py`).** ComplementNB needs non-negative inputs, as does `chi2` selection. `use_stats=True` with StandardScaler produces negatives. If `build_head` doesn't handle this (for example with a MinMax scaler), those trials fail or are skipped, and TPE explores a smaller space than you think.

#### Minor
- `rng = np.random.default_rng(...)` in `05` is unused.
- `study.best_trial` inside `_log` raises if no trial has completed yet.
- `Parallel(n_jobs=2)` pickles the whole `df` and `tr` per worker, which is risky on a 7 GB box.
- `03` shows the source is predictable, but not that the classifier uses it. LOSO is the right test, which is another reason to fix point 1.

If you only fix one thing, fix #1 (nested LOSO selection), because it determines whether your generalization claim holds."

**What we verified:** list of potential flaws to eventually patch.

### D-3: Features integration flaw 
**Author:** Dylan · **Date:** 2026-11-22

**Prompt:** "A flaw I see with the model is the string parsing and feature selection. In theory, if a never-before-seen word appears in a future mail, our model won't know how to handle it. What solutions or alternatives can you suggest?"

**AI output:** "You're right to worry, but the behavior is more specific than "doesn't know how to handle it." `TfidfVectorizer` doesn't crash on unseen words, it silently drops them. The email is scored only on the words the model already knows. This is harmless for ordinary drift (a new product name in a legitimate email) but dangerous for adversarial cases, where a spammer obfuscates tokens ("v1agra", "p@ypal", zero-width characters) so the key signal falls out of vocabulary. Your `chi2` selection with `select_frac=0.2` and `min_df` make this worse, since they deliberately discard rare tokens.

**Cheap fixes within your current stack**

1. **Character n-grams.** You already have `use_char` in the search space. Char n-grams (3-5, `analyzer="char_wb"`) let "paypa1" still share most of its n-grams with "paypal". This is the most direct answer to your concern. Consider forcing it on rather than leaving it to TPE, since your in-distribution CV objective won't reward robustness to unseen words.
2. **HashingVectorizer.** There is no stored vocabulary, so every token maps to a bucket. Unseen words still collide into some bucket instead of vanishing, and it's stateless and memory-light (helpful on your 7 GB box). The cost is lost interpretability: you can't inspect top tokens as in `04`, and collisions add noise.
3. **More normalization.** Extend your masking beyond URL/email/digits: map leetspeak and homoglyphs to ASCII, strip zero-width and Unicode tricks, collapse repeated characters ("freeeee" to "free"), and add a rare-word placeholder (anything with document frequency below a threshold becomes `<RARE>`). The model then learns that a lot of rare tokens is itself a signal.
4. **Structural features.** Rate of out-of-vocabulary tokens, fraction of uppercase, URL count, and HTML/script presence. These don't depend on specific words, so they generalize across drift. Your `use_stats` option is a start.

**Heavier alternatives**

5. **fastText-style subword embeddings.** Words are represented as sums of their character n-gram vectors, so an unseen word still gets a meaningful vector. A good middle ground: small, fast, CPU-friendly.
6. **Pretrained transformer embeddings or fine-tuning** (DistilBERT, MiniLM, or a multilingual sentence encoder). WordPiece/BPE tokenization has essentially no OOV problem, and pretraining gives semantic knowledge of words never seen in your corpus. Tradeoffs are compute, and that long emails need truncation or chunking (you already cap at `MAX_CHARS`). A common compromise is frozen embeddings plus logistic regression, which keeps your pipeline and evaluation protocol intact.
7. **Ensemble.** Combine TF-IDF+LR (strong on known vocabulary, interpretable) with a char-level or embedding model (robust to novelty). Errors tend to be uncorrelated.

**Measure it before fixing it**

- Compute the **OOV rate** per held-out source in your LOSO loop (fraction of test tokens not in the training vocabulary) and plot it against the F1/recall drop. If they correlate, you have evidence that vocabulary is the bottleneck. If not, the shortcut/source problem from `03` matters more.
- Run an **obfuscation stress test**: perturb the test spam (leetspeak, inserted punctuation, synonym swaps) and compare recall across models. Word TF-IDF will degrade the most, char n-grams and embeddings the least.
- For deployment, monitor OOV rate and score distribution over time, and plan periodic retraining. No static vocabulary survives indefinitely, since spam is adversarial and non-stationary.

If I had to pick one path: force char n-grams on, add the normalization and structural features, and run the obfuscation test. If recall still collapses on unseen sources, move to frozen sentence embeddings plus a linear head."

**What we verified:** a flaw regarding model generalization and ideas to fix it. The most promising solution is using character n-grams.

### D-4: char n-grams implementation attempt 
**Author:** Dylan · **Date:** 2026-11-22

**Prompt:** "Let's implement the character n-gram strategy. Provide a modified version of the code with all the relevant changes."

**AI output:** `features_char.py`, `text_norm.py`, `obfuscate.py`, `data_io.py`, `05_automl_v02_char.py`, `05d_eval_char.py`

**What we verified:** an attempt to correct the flaw; we didn't keep the draft because of compatibility issues (the code was generated on partial context).


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