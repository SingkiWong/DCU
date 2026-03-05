#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd || cd "$SCRIPT_DIR" && pwd)
cd "$REPO_ROOT"

MATRIX_FILE=${1:-matrices/circuit_2.mtx}
make -f Makefile.single spai_single_dcu
./spai_single_dcu --strategy dynamic --matrix "$MATRIX_FILE"
