#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

MATRIX=${1:-matrices/circuit_2.mtx}

echo "[1/4] static_single_card"
algorithm_packages/01_static_single_card/run.sh "$MATRIX"

echo "[2/4] static_multi_card (2 cards)"
algorithm_packages/02_static_multi_card/run.sh 1 "$MATRIX"

echo "[2/4] static_multi_card (4 cards)"
algorithm_packages/02_static_multi_card/run.sh 2 "$MATRIX"

echo "[2/4] static_multi_card (6 cards)"
algorithm_packages/02_static_multi_card/run.sh 3 "$MATRIX"

echo "[3/4] dynamic_single_card"
algorithm_packages/03_dynamic_single_card/run.sh "$MATRIX"

echo "[4/4] dynamic_multi_card (MPI+HIP) 示例需要mpirun环境"
echo "例如：algorithm_packages/04_dynamic_multi_card/run.sh 3 3 $MATRIX"
