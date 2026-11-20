"""Project-wide constants: paths, seeds, source lists."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"

SEED = 42

# File names exactly as distributed in the Kaggle package (note the "SpamAssasin" typo).
SOURCES = ["CEAS_08", "Enron", "Ling", "Nazario", "Nigerian_Fraud", "SpamAssasin"]

# Sources containing both classes -> usable as held-out *binary* test corpora in LOSO.
BINARY_SOURCES = ["CEAS_08", "Enron", "Ling", "SpamAssasin"]
# Positive-only sources -> can only be evaluated with recall (detection rate).
POSITIVE_ONLY_SOURCES = ["Nazario", "Nigerian_Fraud"]

# Bodies longer than this are truncated (a few Nazario mails embed MB-sized base64 attachments).
MAX_CHARS = 20_000

# Near-duplicate detection (MinHash LSH on word 5-shingles).
NEARDUP_THRESHOLD = 0.8
NEARDUP_NUM_PERM = 64
NEARDUP_MAX_WORDS = 400

for p in (PROCESSED, FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)
