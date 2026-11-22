# AI_AUDIT.md: "Catch & Correct" report

Tool used: Claude (Anthropic), as an engineering accelerator for boilerplate code, EDA drafting and
literature search. Every item below is a case where the AI-generated logic was **wrong or unsafe**,
how we caught it (with evidence), and the version that fixed it. Every number here can be regenerated
with `run_all.sh`.

> TODO (team): each member re-reads one item and re-runs the check that proves it (see *Evidence*),
> so that anyone can defend it at the oral.

---

## 1. Masking regex silently missed every URL in Enron and Ling → a *new* source shortcut

| | |
|---|---|
| **AI output (v0.1.0)** | `clean_text()` replaced URLs with `urltoken` using `https?://\S+ \| www\.\S+`. |
| **Flaw** | Enron and Ling are distributed *pre-tokenised* (`http : / / www . pharmacy . com`). The regex matched **0 %** of their links, while it matched the raw corpora correctly. The model would have learned "contains `urltoken`" = "not Enron/Ling", so the masking step meant to *remove* a source artefact would have *created* one. |
| **How we caught it** | Per-source sanity check: share of e-mails containing `urltoken` vs share containing `http`/`www` in the raw text. |
| **Evidence** | Old detection rate: Enron 0.000, Ling 0.000 vs raw 0.204 / 0.398. New: 0.199 / 0.395 (CEAS, Nazario, Nigerian, SpamAssassin unchanged). |
| **Fix (v0.1.1)** | Added `_URL_SPACED` and `_EMAIL_SPACED` patterns in `src/phish/text.py`; the rates now match the raw `http` rate for every source. |
| **Lesson** | Any preprocessing step must be validated **per source**, not only globally. |

## 2. The week-1 evaluation plan proposed "train on 5 sources, test on the 6th" for all six sources

| | |
|---|---|
| **AI output (week-1 spec)** | Six leave-one-source-out folds, each reporting accuracy / F1. |
| **Flaw** | Nazario and Nigerian_Fraud contain **only malicious e-mails** (100 % label 1). On those folds precision is trivially 1, and F1 and FPR are undefined or meaningless (there are no negatives). Reporting F1 there would have inflated the average. |
| **How we caught it** | Class composition per source (Table `audit_composition.csv`, Fig. 1) and cross-check with Bhuiyan & Bhuiyan (2026), who exclude these two corpora from held-out binary testing. |
| **Fix** | `src/phish/split.py::loso_folds` + `metrics.full_report`: the 4 binary sources get full metrics; the 2 positive-only sources get **recall only** (detection rate). |

## 3. The regularisation grid stopped at the optimum → boundary solution, memorisation not flagged

| | |
|---|---|
| **AI output (v0.1.0 of `04_baseline.py`)** | Grid `C ∈ {0.01 … 100}`, choose `argmax(mean val F1)`. |
| **Flaw** | The argmax was **C = 100, the last grid point** (so the optimum may lie outside the grid), and training F1 was 1.000 (pure memorisation), while validation F1 was flat from C = 10 upward (differences smaller than the fold standard deviation). Picking the argmax means picking noise and the least-regularised model. |
| **How we caught it** | Validation curve with train and validation F1 and their std (Fig. 5). |
| **Fix (v0.1.1)** | Grid extended to 1000 and **one-standard-error rule**: pick the most regularised C whose mean validation F1 is within one SE of the best → C = 30 on the machine that produced the first results (argmax 300, validation F1 0.9888 vs 0.9885); C = 100 when re-run in the pinned environment (argmax 300, 0.9889 vs 0.9887), see #8. Either way the selected C is far more regularised than the argmax and statistically equivalent to it. |

## 4. Text cleaning kept underscores → "_____" separator lines became the top Nigerian_Fraud feature

| | |
|---|---|
| **AI output** | `[^a-z0-9_ ]` as the "non-word" class (underscore kept because it is part of `\w`). |
| **Flaw** | The shortcut test listed `___________…` as the most source-identifying token for Nigerian_Fraud and `_nextpart_` for SpamAssassin: formatting debris, not content. |
| **How we caught it** | `scripts/03_shortcut_test.py` prints the top tokens that identify each source. |
| **Fix** | Underscore removed with the other punctuation (`src/phish/text.py`). |

## 5. Using the pre-merged `phishing_email.csv` (the obvious, AI-suggested input)

| | |
|---|---|
| **AI output** | Load `phishing_email.csv` (82,486 rows, columns `text_combined`, `label`). |
| **Flaw** | No `source` column, so leave-one-source-out and the shortcut test are impossible. Its text was already transformed by the dataset authors (lower-casing, stop-word/punctuation removal: first row reads `hpl nom may 25 2001 see attached file hplno 525 xls…`), so our own preprocessing choices would be unauditable. |
| **Fix** | Rebuilt the corpus from the six per-source CSVs (`src/phish/data.py::load_sources`); row and label counts verified identical (82,486 = 42,891 + 39,595). |

## 6. AutoML search space: linear SVM with C up to 1000 → non-convergence and an unbounded search

| | |
|---|---|
| **AI output (run 1 of `05_automl_v01.py`)** | Same C range `[1e-2, 1e3]` for logistic regression and linear SVM, no time budget. |
| **Flaw** | liblinear's linear SVM does not converge for large C on 100k+ sparse features (`ConvergenceWarning`). Those trials took 200+ s each (vs ~10 s), and a non-converged model is not the model we think we evaluate. The search had no time budget, so the run had no predictable end. |
| **How we caught it** | Per-trial log (`log_05_run1_aborted.txt`): trial durations jumped from ~10 s to 245 s and 214 s on `linear_svm` trials, with liblinear warnings. |
| **Fix (run 2)** | Linear SVM C capped at 100, `timeout = 25 min` on the search, family refits in parallel. Run 1 was aborted after 35 trials (log kept: `log_05_run1_aborted.txt`). |
| **Follow-up** | Run 2 was also stopped manually after 34 trials, before it saved its trials (the script only wrote results at the end, a second flaw). We recovered the parameters by replaying the deterministic TPE sampler on the logged history (`05b_recover_best.py`) and **re-computed** the CV score of the 3 best configurations: all match the log to 4 decimals. Lesson: persist every trial as it completes (Optuna SQLite storage) — planned for v0.2. |

---

### Checks that the AI proposed and that turned out *not* to matter (reported for honesty)

* **Near-duplicate leakage.** 13,314 rows belong to near-duplicate clusters (e.g. 3,919 copies of
  fake "CNN Alerts" / "CNN.com Daily Top 10" spam in CEAS_08). We expected a random split to inflate
  scores. Measured effect on the exact-dedup corpus: F1 0.9907 (random split, 1,290 clusters crossing)
  vs 0.9907 (grouped split, 0 crossing). **No measurable inflation for this model.** We keep the
  grouped split anyway (it costs nothing and closes the leakage path by construction). Note: on the final de-duplicated corpus every cluster holds one e-mail, so the grouping is
  redundant there; it only matters when near-duplicates are kept.
* **Non-ASCII characters dropped by the cleaning.** We suspected this might explain part of the low
  Nazario recall. Measured (`check_nazario_nonascii.csv`): only 27 Nazario e-mails (1.9 %) are mostly
  non-ASCII; their recall is 0.41 vs 0.58, which moves the overall Nazario recall by 0.3 points.
  **Not the explanation.**
* **Cross-source near-duplicates.** The kept copy is the one of the first source in the list, so 402
  SpamAssassin e-mails (7.1 %) and 121 Nigerian_Fraud e-mails (3.7 %) are counted in Enron. No leakage
  (no e-mail on both sides of a split), small composition change, now stated in the report.

## 7. Reproducibility: version-dependent date parsing, a wrong `pip`, and a false alarm on a pinned version

| | |
|---|---|
| **AI output** | (a) `pd.to_datetime(df["date"], errors="coerce", utc=True, format="mixed")` in `01_audit_and_build.py`. (b) Advice to run `pip install datasketch optuna`. (c) While checking `requirements.txt`, the claim that `optuna==5.1.0` "does not exist on PyPI". |
| **Flaw** | (a) RFC-2822 dates ending in a zone comment such as `+0100 (CET)` were mishandled in two version-dependent ways. With pandas 2.2 (installed on a team laptop), 149 of them make `to_datetime` **crash** (`AttributeError ... total_seconds`) despite `errors="coerce"`, so `run_all.sh` stopped at step 1. With the pinned pandas 3 they did not crash but were **silently coerced to NaT**: 165 parseable dates were lost, which inflated the "unparsed" rate (Nazario 1.9 % instead of 0.2 %, SpamAssassin 3.5 % instead of 1.4 %). (b) On that laptop `pip` belongs to a Python 3.9 install, not to the Anaconda Python that runs the scripts, so the packages would have been installed in the wrong interpreter. (c) The "missing version" was an artefact of querying that same wrong `pip` (it only lists versions compatible with Python 3.9); all pins exist. |
| **How we caught it** | Running `run_all.sh` on a second machine (traceback); comparing the parsed years with and without the fix under pandas 3 (165 rows differ, all previously NaT); `which pip` vs `which python`; re-querying PyPI with `python -m pip index versions`. |
| **Fix** | (a) Strip the trailing parenthesised comment before parsing (the numeric offset already carries the zone). Works under pandas 2 and 3; the year quantiles (p05 / median / p95) of every source are unchanged, only the unparsed rates drop to their true values (Fig. 4 regenerated). (b)+(c) `requirements.txt` repaired (conflict markers removed, `seaborn`/`kagglehub` added) and documented with `python -m pip install -r requirements.txt`; a `.venv` built from the exact pins reproduces the pipeline. |
| **Lesson** | "It runs on my machine" is not reproducibility: re-run the whole pipeline in a clean environment built from `requirements.txt` before each submission, and verify the AI's environment diagnoses too. |

## 8. The AutoML recovery relied on a bit-exact replay of the Optuna sampler → `run_all.sh` failed on another machine

| | |
|---|---|
| **AI output** | `05b_recover_best.py` reconstructed the parameters of the stopped search by replaying the TPE sampler (seed 42) on the logged history, and **aborted** if the replay disagreed with the log. |
| **Flaw** | The replay is only bit-reproducible on the platform that ran the search. On a Mac with the same pinned versions it diverged at trial 19 (log: `decision_tree`, replay: `sgd_huber`), so `run_all.sh` stopped and no AutoML result could be regenerated. A reproducibility step that is itself not reproducible. |
| **How we caught it** | Full re-run of `run_all.sh` in a clean `.venv` built from `requirements.txt` on a second machine. |
| **Fix** | If the replay diverges, 05b falls back to the parameters recovered on the original platform (`reports/tables/automl_trials.csv`, committed), after checking that trial numbers, states, scores and model families match the log for all 34 trials. The proof that does not depend on the sampler is kept: the top-3 configurations are re-evaluated with the same grouped CV and reproduce their logged CV F1 (0.9827). Same best configuration (trial 25, ridge least-squares). |
| **Side finding** | The other regenerated numbers differ from the committed ones only in the 3rd–4th decimal (cross-platform floating point; corpus, clusters and split are identical). This is enough to move the one-standard-error choice of the baseline from C = 30 to C = 100, two values that are statistically indistinguishable (validation F1 0.9885 vs 0.9887, fold std 0.0008). Conclusions are unchanged; the report now quotes the numbers regenerated with the pinned environment. |

## 9. The URL/e-mail masking fix of #1 was incomplete: Enron addresses and URL fragments still identify the source

| | |
|---|---|
| **AI output (v0.1.1, `src/phish/text.py`)** | Four regexes (`_URL`, `_URL_SPACED`, `_EMAIL`, `_EMAIL_SPACED`) meant to replace links and addresses by `urltoken` / `emailtoken` in raw *and* pre-tokenised corpora. |
| **Flaw** | Still source-dependent on the pre-tokenised corpora. (a) Enron's internal Lotus Notes addresses (`vince j kaminski / hou / ect @ ect`, 19,877 occurrences) have no dot in the domain, so they are **not masked**: the "masked" text keeps employee names plus `hou`/`ect` (10.7 % of Enron mails, < 0.4 % elsewhere). (b) Paths and extensions after a spaced URL survive (`index . php` → `php`; `php`, `asp`, `href`, `htm` are among the top Enron/Ling source tokens in `shortcut_top_tokens_per_source.csv`). (c) `_URL_SPACED` swallows the word after a sentence-final period (`www . site . com . thanks` → `urltoken`), and `_EMAIL_SPACED` creates false addresses (`meeting @ 3 . 30` → `emailtoken`). |
| **How we caught it** | Per-source parity check (same idea as #1): share of mails with `emailtoken` vs share with `@` in the raw text. Enron 20.4 % vs 31.4 % (other sources within 2 points); then reading the unmasked contexts. |
| **Evidence** | Prototype fix measured on the 71,386-mail corpus: Enron `emailtoken` 20.4 % → 31.9 % (raw `@`: 31.4 %); Enron mails with `hou`/`ect` 10.7 % → 0.4 %; Ling mails with URL fragments 20.0 % → 13.8 %; raw corpora unchanged. |
| **Fix (planned v0.2, not applied in week 2)** | Undo the pre-tokenisation first, then mask with two standard regexes: join `/ : @` with their neighbours and ` . ` before a TLD or file extension (`http : / / www . x . com` → `http://www.x.com`), then `(https?://\|www\.)\S+` → `urltoken`, and `word@word(.word)*` or a Notes address `name/hou/ect(@ect)` → `emailtoken`. Simpler (2 steps instead of 4 overlapping regexes) and source-agnostic. **Not applied to the week-2 results** because AutoML v0.1 selected `text_mask`: changing it requires a new search, scheduled with the nested-LOSO search of v0.2. Expected effect to measure then: source-prediction accuracy on masked text (currently 95.0 %) and LOSO FPR on Enron. |
| **Lesson** | A fix must be re-validated with the same per-source check that found the bug, on *every* artefact type (URLs **and** addresses), not only on the example that triggered it. |

## 10. Hyper-parameters selected on all sources, then reused in leave-one-source-out

| | |
|---|---|
| **AI output (`04_baseline.py`, `05_automl_v01.py`)** | C of the baseline (and the whole AutoML configuration) chosen by CV on the training split, which contains **all six sources**, then reused unchanged in every LOSO fold. The report only flagged this for AutoML. |
| **Flaw** | Selection leakage: the hyper-parameter has "seen" the held-out source, so LOSO scores can be optimistic. |
| **How we caught it** | Reviewing which data each choice was made on, fold by fold. |
| **Evidence** | `scripts/09_sensitivity_checks.py`: C re-selected for each fold **without** the held-out source (3-fold grouped CV, same 1-SE rule). Selected C varies (30–1000), but LOSO F1 changes by at most 0.005 and FPR by at most 0.4 points (`check_loso_nested_C.csv`). |
| **Fix** | For the baseline: measured, negligible, now stated with the numbers in the report (*Risks*). For AutoML, whose search space is much larger: nested LOSO selection is item 1 of the v0.2 plan. |

## 11. `demo_metrics.py`: random split on raw data, silent synthetic data, and an invented "attack"

| | |
|---|---|
| **AI output** | `demo_metrics.py` (end-to-end metric demo). |
| **Flaw** | (a) `train_test_split` at random on one raw CSV, **without de-duplication**: near-copies of a campaign can sit in train and test. (b) If the CSV is missing, it **generates a simulated corpus** from 10 templates and carries on, so its metrics may describe fake data. (c) Invalid labels silently become 0 (`to_numeric(...).fillna(0)`). (d) The "SVM under evasion attack" row is not an attack: the probabilities of a random 35 % of phishing test mails are multiplied by 0.3 by hand (`y_proba_attacked[...] *= 0.30`), producing an invented F1 of 0.768. |
| **How we caught it** | Code review of every script that goes into the submission zip. |
| **Fix** | Loads the de-duplicated corpus and the project's frozen split (same test set as the baseline), stops with an error if they are missing, asserts labels ∈ {0, 1}; the fake attack was removed. New run: LR F1 0.980, calibrated linear SVM F1 0.983 on the frozen test (the old numbers were on SpamAssassin only, so they are not comparable). |
| **Lesson** | A demo script is part of the deliverable: a silent fallback or a simulated result is as serious as a leak, because nobody re-reads it. |

## 12. Report text with hand-typed numbers that drifted from the results (and one self-contradiction)

| | |
|---|---|
| **AI output** | Report prose and README drafted with numbers typed in (`0.9 %`, `5–23 %`, `59 %`, `FPR within 1.1 points`, `0.65 % vs 0.93 %`…), although `07_latex_tables.py` claims that "no number is typed by hand". |
| **Flaw** | After re-running the pipeline in the pinned environment (#8), several statements no longer matched the tables: Nazario recall 58 % (not 59 %), LOSO FPR 5.6–24.0 %, FPR gap to the paper 1.3 points (the claim "within 1.1 points" became **false**). The AutoML discussion also said "worse on **three** of the four binary sources" and then listed an F1 drop on **all four**. It also claimed that AutoML "trades recall for fewer false positives", but the paired bootstrap shows the FPR gain is not significant (p = 0.17). |
| **How we caught it** | `git diff` of the regenerated tables, then a sentence-by-sentence check of every number in `main.tex`, `automl_discussion.tex` and `README.md`. |
| **Fix** | `07_latex_tables.py` now writes every **model-result** number quoted in the prose as a LaTeX macro (`tables/val_text.tex`) and the text uses them (the data-audit counts, which only change if the data changes, stay typed and were re-checked against the audit tables); the contradiction and the FPR claim were rewritten from the macros and the confidence intervals. |
