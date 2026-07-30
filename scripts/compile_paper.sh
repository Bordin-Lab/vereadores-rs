#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAPER="$ROOT/paper"
cd "$PAPER"

BIBTEX="$(command -v bibtex || true)"
if [[ -z "$BIBTEX" || ! -x "$BIBTEX" ]]; then
    if [[ -x /usr/bin/bibtex.original ]]; then
        BIBTEX=/usr/bin/bibtex.original
    elif command -v bibtex8 >/dev/null 2>&1; then
        BIBTEX="$(command -v bibtex8)"
    else
        echo "BibTeX executable not found." >&2
        exit 1
    fi
fi

compile_one() {
    local stem="$1"
    pdflatex -interaction=nonstopmode -halt-on-error "$stem.tex" >/tmp/"$stem"_pdflatex_1.log
    "$BIBTEX" "$stem" >/tmp/"$stem"_bibtex.log
    pdflatex -interaction=nonstopmode -halt-on-error "$stem.tex" >/tmp/"$stem"_pdflatex_2.log
    pdflatex -interaction=nonstopmode -halt-on-error "$stem.tex" >/tmp/"$stem"_pdflatex_3.log
    if grep -Eq "undefined references|undefined citations|There were undefined" /tmp/"$stem"_pdflatex_3.log; then
        echo "Unresolved references remain in $stem.pdf" >&2
        exit 1
    fi
}

compile_one manuscript
compile_one supplementary_information
printf 'Built %s and %s\n' "$PAPER/manuscript.pdf" "$PAPER/supplementary_information.pdf"
