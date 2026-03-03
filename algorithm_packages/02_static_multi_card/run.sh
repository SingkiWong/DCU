#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd || cd "$SCRIPT_DIR" && pwd)
cd "$REPO_ROOT"

UNITS=${1:-2}      # DCU units: 1/2/3 -> 2/4/6 cards
MATRIX_FILE=${2:-matrices/circuit_2.mtx}
make -f Makefile.multi multi
./spai_multi_dcu "$UNITS" --partition balanced_columns --matrix "$MATRIX_FILE"
