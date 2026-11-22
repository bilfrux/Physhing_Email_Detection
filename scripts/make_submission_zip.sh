#!/usr/bin/env bash
# Makes Group_XX_DS_Project.zip for Moodle (layout from the project brief).
# Usage: bash scripts/make_submission_zip.sh 07
set -euo pipefail
G="${1:?give the group number, e.g. 07}"
NAME="Group_${G}_DS_Project"
cd "$(dirname "$0")/.."
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
D="$TMP/$NAME"
mkdir -p "$D/report" "$D/ai_governance" "$D/data" "$D/reports"

cp reports/latex/rapport.pdf "$D/report/Report_Group_${G}.pdf"
cp ai-governance/*.md "$D/ai_governance/"
cp ai-governance/*.md "$D/"   # the brief also shows them at the root
cp requirements.txt README.md run_all.sh LICENSE *.py "$D/"
cp data/README.md "$D/data/"   # no data (licences), only how to download it
rsync -a --exclude '__pycache__' --exclude '.DS_Store' src scripts "$D/"
rsync -a --exclude '.DS_Store' --exclude '*.log' reports/tables reports/figures reports/latex "$D/reports/"

rm -f "$NAME.zip"
(cd "$TMP" && zip -qr "$OLDPWD/$NAME.zip" "$NAME")
echo "wrote $NAME.zip"
