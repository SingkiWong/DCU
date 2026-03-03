#!/usr/bin/env bash
set -euo pipefail
UNITS=${1:-2}      # DCU units: 1/2/3 -> 2/4/6 cards
MATRIX_FILE=${2:-circuit_2.mtx}
make -f Makefile.multi multi
printf "%s\n" "$MATRIX_FILE" | ./spai_multi_dcu "$UNITS"
