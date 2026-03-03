#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd || cd "$SCRIPT_DIR" && pwd)
cd "$REPO_ROOT"

RANKS=${1:-2}
UNITS=${2:-2}      # DCU units: 1/2/3 -> 2/4/6 cards
MATRIX_FILE=${3:-matrices/circuit_2.mtx}
make -f Makefile.multi mpi
mpirun -np "$RANKS" ./spai_multi_dcu_mpi "$UNITS" --partition balanced_nnz --precondition-only --matrix "$MATRIX_FILE"
