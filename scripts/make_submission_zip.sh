#!/usr/bin/env bash
# Builds the Moodle archive with the hierarchy required by the Project Brief & Governance Policy:
#
#   Group_XX_DS_Project.zip
#   |- requirements.txt            (pinned versions)
#   |- src/  scripts/  run_all.sh  (code, relative paths)
#   |- report/Report_Group_XX.pdf
#   |- ai_governance/              CONTRIBUTIONS.md, PROMPT_LOG.md, AI_AUDIT.md
#   |- CONTRIBUTIONS.md  PROMPT_LOG.md  AI_AUDIT.md   (copies at the root, as drawn in the brief)
#   |- reports/                    result tables, figures, LaTeX source
#
# Raw and processed data are NOT included (licence + privacy); data/README.md explains how to get them.
# Usage:  bash scripts/make_submission_zip.sh 07      (07 = your group number)
set -euo pipefail
GROUP="${1:?usage: bash scripts/make_submission_zip.sh <group number, e.g. 07>}"
NAME="Group_${GROUP}_DS_Project"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PDF="$ROOT/reports/latex/main.pdf"
GOV="$ROOT/ai-governance"

for f in "$PDF" "$GOV/CONTRIBUTIONS.md" "$GOV/PROMPT_LOG.md" "$GOV/AI_AUDIT.md" "$ROOT/requirements.txt"; do
  [[ -s "$f" ]] || { echo "missing or empty: $f" >&2; exit 1; }
done
if grep -q '^<<<<<<<\|^>>>>>>>' "$ROOT/requirements.txt"; then echo "requirements.txt has merge-conflict markers" >&2; exit 1; fi
if grep -qi 'TODO' "$GOV"/*.md; then echo "warning: TODO left in ai-governance/*.md" >&2; fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
D="$STAGE/$NAME"
mkdir -p "$D/report" "$D/ai_governance" "$D/data" "$D/reports"

cp "$ROOT"/{requirements.txt,README.md,run_all.sh,LICENSE} "$D/"
cp "$ROOT"/{data_profiler.py,metrics_evaluation.py,demo_metrics.py,importdataset.py} "$D/" 2>/dev/null || true
cp "$ROOT/data/README.md" "$D/data/"
rsync -a --exclude '__pycache__' --exclude '.DS_Store' "$ROOT/src" "$ROOT/scripts" "$D/"
rsync -a --exclude '.DS_Store' --exclude '*.aux' --exclude '*.log' --exclude '*.fls' --exclude '*.fdb_latexmk' \
      --exclude '*.out' --exclude '*.synctex.gz' "$ROOT/reports/tables" "$ROOT/reports/figures" "$ROOT/reports/latex" "$D/reports/"
cp "$PDF" "$D/report/Report_Group_${GROUP}.pdf"
for f in CONTRIBUTIONS.md PROMPT_LOG.md AI_AUDIT.md; do
  cp "$GOV/$f" "$D/ai_governance/$f"
  cp "$GOV/$f" "$D/$f"
done

rm -f "$ROOT/$NAME.zip"
(cd "$STAGE" && zip -qr "$ROOT/$NAME.zip" "$NAME")
echo "wrote $ROOT/$NAME.zip ($(du -h "$ROOT/$NAME.zip" | cut -f1))"
unzip -l "$ROOT/$NAME.zip" | awk 'NR>3 {print $4}' | grep -E '^[^/]+/[^/]*/?$' | sort
