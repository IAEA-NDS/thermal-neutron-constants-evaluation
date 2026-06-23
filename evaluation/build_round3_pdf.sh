#!/usr/bin/env bash
# Build evaluation/report_capote_round3.pdf from the markdown source.
# Run from the evaluation/ directory or repo root; will cd to evaluation/.
set -euo pipefail

cd "$(dirname "$0")"

pandoc report_capote_round3.md \
  -o report_capote_round3.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=1.5cm \
  -V mainfont="Linux Libertine O" \
  -V monofont="DejaVu Sans Mono" \
  -V fontsize=10pt \
  -V colorlinks=true \
  -H report_capote_fit.header.tex

echo "Wrote report_capote_round3.pdf ($(stat -c%s report_capote_round3.pdf) bytes)"
