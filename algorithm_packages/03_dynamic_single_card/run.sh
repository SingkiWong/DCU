#!/usr/bin/env bash
set -euo pipefail
MATRIX_FILE=${1:-circuit_2.mtx}
make -f Makefile.single spai_single_dcu
printf "%s\n" "$MATRIX_FILE" | ./spai_single_dcu
