#!/usr/bin/env bash
set -euo pipefail
RANKS=${1:-2}
UNITS=${2:-2}      # DCU units: 1/2/3 -> 2/4/6 cards
MATRIX_FILE=${3:-circuit_2.mtx}
make -f Makefile.multi mpi
printf "%s\n" "$MATRIX_FILE" | mpirun -np "$RANKS" ./spai_multi_dcu_mpi "$UNITS"
