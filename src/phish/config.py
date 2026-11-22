from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"

SEED = 42

# file names from Kaggle (yes, "SpamAssasin" is misspelled there)
SOURCES = ["CEAS_08", "Enron", "Ling", "Nazario", "Nigerian_Fraud", "SpamAssasin"]
BINARY_SOURCES = ["CEAS_08", "Enron", "Ling", "SpamAssasin"]  # have both classes
POSITIVE_ONLY_SOURCES = ["Nazario", "Nigerian_Fraud"]  # only recall makes sense

MAX_CHARS = 20_000  # some Nazario mails have huge base64 attachments

# Bhuiyan & Bhuiyan (2026), Tables 18-19: TF-IDF + LR, leave-one-corpus-out
SOTA_LOCO = {"CEAS_08": {"f1": 0.8994, "fpr": 0.1137}, "Enron": {"f1": 0.8201, "fpr": 0.2266},
             "Ling": {"f1": 0.8368, "fpr": 0.0512}, "SpamAssasin": {"f1": 0.8433, "fpr": 0.1269}}

# near-duplicates: MinHash LSH on word 5-shingles
NEARDUP_THRESHOLD = 0.8
NEARDUP_NUM_PERM = 64
NEARDUP_MAX_WORDS = 400

for folder in (PROCESSED, FIG, TAB):
    folder.mkdir(parents=True, exist_ok=True)
