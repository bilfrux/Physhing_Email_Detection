"""Text cleaning.

Enron and Ling come lower-cased and tokenised ("see attached file : hplno 530 . xls"),
the other corpora are raw. If we keep case and punctuation, the model can tell the
corpora apart from that alone, so we remove both. Digits, URLs and addresses also
give the source away (years, domains), so they can be replaced by tokens.
"""
import re

_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
# tokenised version, e.g. "http : / / www . pharmacy . com" in Enron/Ling
_URL_SPACED = re.compile(r"\b(https?\s*:\s*/\s*/\s*|www\s*\.\s*)[\w-]+(\s*\.\s*[\w-]+)*(\s*/\s*[\w.%?=&-]+)*", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_EMAIL_SPACED = re.compile(r"\b[\w.+-]+\s*@\s*[\w-]+(\s*\.\s*[\w-]+)+")
_DIGITS = re.compile(r"\d+")
_NOT_WORD = re.compile(r"[^a-z0-9 ]+")  # also drops "_", the "_____" lines are typical of Nigerian_Fraud
_SPACES = re.compile(r"\s+")


def normalize(s):
    """Lower-case and keep only letters/digits. Used to find duplicates."""
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return _SPACES.sub(" ", s).strip()


def clean_text(s, mask=True):
    """Model input. mask=True also replaces URLs, e-mails and numbers by tokens."""
    s = s.lower()
    if mask:
        s = _URL.sub(" urltoken ", s)
        s = _URL_SPACED.sub(" urltoken ", s)
        s = _EMAIL.sub(" emailtoken ", s)
        s = _EMAIL_SPACED.sub(" emailtoken ", s)
        s = _DIGITS.sub(" numtoken ", s)
    s = _NOT_WORD.sub(" ", s)
    return _SPACES.sub(" ", s).strip()
