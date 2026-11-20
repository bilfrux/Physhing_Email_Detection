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
| **Fix (v0.1.1)** | Grid extended to 1000 and **one-standard-error rule**: pick the most regularised C whose mean validation F1 is within one SE of the best → C = 30 (argmax was 300, F1 0.9888 vs 0.9885). |

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
  grouped split anyway (it costs nothing and closes the leakage path by construction).
