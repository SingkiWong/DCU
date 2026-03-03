#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd || cd "$SCRIPT_DIR" && pwd)
cd "$REPO_ROOT"

RANKS=${1:-2}
UNITS=${2:-2}      # DCU units: 1/2/3 -> 2/4/6 cards
MATRIX_FILE=${3:-matrices/circuit_2.mtx}
make -f Makefile.multi mpi
printf "%s\n" "$MATRIX_FILE" | mpirun -np "$RANKS" ./spai_multi_dcu_mpi "$UNITS"
