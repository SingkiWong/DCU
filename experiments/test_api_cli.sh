#!/usr/bin/env bash
set -euo pipefail

BIN=${1:-./spai_multi_dcu_mpi}
MATRIX=${2:-matrices/circuit_2.mtx}
DCU_COUNT=${3:-1}
MPI_NP=${MPI_NP:-1}

if [[ ! -x "$BIN" ]]; then
  echo "[FAIL] binary not found or not executable: $BIN" >&2
  exit 1
fi
if [[ ! -f "$MATRIX" ]]; then
  echo "[FAIL] matrix not found: $MATRIX" >&2
  exit 1
fi

run_case() {
  local name="$1"; shift
  local expected_mode="$1"; shift
  local expected_partition="$1"; shift

  echo "[CASE] $name"
  local out
  set +e
  out=$(mpirun -np "$MPI_NP" "$BIN" "$DCU_COUNT" --matrix "$MATRIX" --precondition-only "$@" 2>&1)
  local rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] case '$name' exited with $rc"
    echo "$out"
    exit 1
  fi

  local line
  line=$(echo "$out" | grep "\[ALGO_REPORT\]" | tail -n1 || true)
  if [[ -z "$line" ]]; then
    echo "[FAIL] case '$name' missing [ALGO_REPORT]"
    echo "$out"
    exit 1
  fi

  echo "$line" | grep -q "mode=${expected_mode}" || { echo "[FAIL] wrong mode in '$name': $line"; exit 1; }
  echo "$line" | grep -q "partition=${expected_partition}" || { echo "[FAIL] wrong partition in '$name': $line"; exit 1; }
  echo "$line" | grep -Eq "preconditioner_ms=[0-9]+(\.[0-9]+)?" || { echo "[FAIL] invalid preconditioner_ms in '$name': $line"; exit 1; }
  echo "$line" | grep -q "iteration_count=0" || { echo "[FAIL] precondition-only should produce iteration_count=0 in '$name': $line"; exit 1; }

  echo "[PASS] $name"
}

run_case "static default partition" static balanced_columns --algo static
run_case "dynamic default partition" dynamic balanced_nnz --algo dynamic
run_case "manual partition override" static balanced_nnz --algo static --partition balanced_nnz

echo "[PASS] API CLI smoke test finished"
