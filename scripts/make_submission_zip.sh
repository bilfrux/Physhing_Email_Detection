#!/usr/bin/env bash
# Builds the submission zip: code + report + governance + result tables/figures.
# Raw and processed data are excluded (licence + size); data/README.md explains how to get them.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=phish-or-not_week2
rm -f "../$NAME.zip"
cd ..
zip -qr "$NAME.zip" phish-or-not \
  -x "phish-or-not/data/raw/*.csv" \
  -x "phish-or-not/data/processed/*" \
  -x "*/__pycache__/*" \
  -x "phish-or-not/reports/latex/*.aux" "phish-or-not/reports/latex/*.log" "phish-or-not/reports/latex/*.fls" \
  -x "phish-or-not/reports/latex/*.fdb_latexmk" "phish-or-not/reports/latex/*.out"
echo "wrote $(pwd)/$NAME.zip"
