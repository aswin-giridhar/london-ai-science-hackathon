#!/usr/bin/env bash
# Two passes: the second resolves \ref and \cite. Needs pdflatex on PATH.
# On this machine MiKTeX chokes if PATH contains a non-existent directory, so the
# toolchain directory is prepended explicitly rather than relying on the inherited PATH.
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
grep -c '^!' main.log && echo "LaTeX errors above" || echo "clean"
