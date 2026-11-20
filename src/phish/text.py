"""Text normalisation.

Why each step exists (see report, Section 3):
- Enron and Ling are distributed fully lower-cased and tokenised with spaces
  around punctuation ("see attached file : hplno 530 . xls"), the other four
  corpora are raw. Without normalisation, casing and punctuation spacing are a
  near-perfect *source* identifier, i.e. a shortcut.
- Digits carry dates ("2008" = CEAS_08) and IDs, which identify the source
  rather than the intent of the email -> optional masking.
- URLs / e-mail addresses are replaced by placeholder tokens so that the model
  learns "contains a link" instead of memorising specific domains.
"""
import re

_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
# Enron/Ling are pre-tokenised: "http : / / www . pharmacy . com" -> must be caught too,
# otherwise masking itself becomes a source artefact (raw corpora masked, tokenised ones not).
_URL_SPACED = re.compile(r"\b(https?\s*:\s*/\s*/\s*|www\s*\.\s*)[\w-]+(\s*\.\s*[\w-]+)*(\s*/\s*[\w.%?=&-]+)*", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_EMAIL_SPACED = re.compile(r"\b[\w.+-]+\s*@\s*[\w-]+(\s*\.\s*[\w-]+)+")
_NUM = re.compile(r"\d+")
_NONWORD = re.compile(r"[^a-z0-9 ]+")  # underscores too: "_____" separator lines are a Nigerian_Fraud artefact
_WS = re.compile(r"\s+")


def basic_normalize(s: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace (used for dedup hashing)."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return _WS.sub(" ", s).strip()


def clean_text(s: str, mask: bool = True) -> str:
    """Model input text.

    mask=False : lower-case + punctuation removal only (minimal cleaning).
    mask=True  : additionally replace URLs, e-mails and digit runs by tokens.
    """
    s = s.lower()
    if mask:
        s = _URL.sub(" urltoken ", s)
        s = _URL_SPACED.sub(" urltoken ", s)
        s = _EMAIL.sub(" emailtoken ", s)
        s = _EMAIL_SPACED.sub(" emailtoken ", s)
        s = _NUM.sub(" numtoken ", s)
    s = _NONWORD.sub(" ", s)
    return _WS.sub(" ", s).strip()
