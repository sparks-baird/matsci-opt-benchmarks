#!/usr/bin/env bash
#
# make-tracked-changes.sh -- produce a "tracked changes" (markup) PDF that
# highlights every difference between a baseline manuscript and the current
# manuscript, using latexdiff.
#
# Usage:
#   bash make-tracked-changes.sh [BASELINE.tex] [CURRENT.tex]
#
# Defaults:
#   BASELINE = revision/manuscript_v0_submitted.tex   (frozen pre-revision copy)
#   CURRENT  = manuscript.tex                          (the live, edited file)
#
# Output:
#   revision/manuscript_diff.pdf                       (additions underlined,
#                                                       deletions struck through)
#
# Notes:
#   * Both inputs must use the same document class (sn-jnl here). To diff
#     against Xavier's *actual* submitted file, drop it in as the baseline
#     (see Q1 in author-questions.md) and pass it as the first argument.
#   * The diff is built inside the manuscript directory so that
#     \graphicspath{{figures/}} and \bibliography{references} resolve.
#   * Requires: latexdiff, latexmk, a TeX Live with the packages listed in
#     ../README.md (texlive-science provides algorithm.sty).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANU_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE="${1:-${SCRIPT_DIR}/manuscript_v0_submitted.tex}"
CUR="${2:-${MANU_DIR}/manuscript.tex}"
OUT_PDF="${SCRIPT_DIR}/manuscript_diff.pdf"
DIFF_TEX="${MANU_DIR}/manuscript_diff.tex"

for tool in latexdiff latexmk; do
  command -v "${tool}" >/dev/null 2>&1 || {
    echo "error: '${tool}' not found; install the toolchain (see ../README.md)" >&2
    exit 1
  }
done
for f in "${BASE}" "${CUR}"; do
  [ -f "${f}" ] || { echo "error: input not found: ${f}" >&2; exit 1; }
done

echo ">> latexdiff:"
echo "     baseline = ${BASE}"
echo "     current  = ${CUR}"
latexdiff "${BASE}" "${CUR}" > "${DIFF_TEX}"

echo ">> building markup PDF ..."
( cd "${MANU_DIR}" && latexmk -pdf -interaction=nonstopmode -halt-on-error \
    manuscript_diff.tex >/dev/null )

mv -f "${MANU_DIR}/manuscript_diff.pdf" "${OUT_PDF}"

# Tidy intermediates (the *.tex is regenerated on every run).
( cd "${MANU_DIR}" && latexmk -c manuscript_diff.tex >/dev/null 2>&1 || true )
rm -f "${DIFF_TEX}"

echo ">> wrote ${OUT_PDF}"
