#!/usr/bin/env python3
"""Utilities for DCU SPAI benchmark database (4 algorithm packages).

Usage:
  python3 experiments/benchmark_db.py init --db experiments/benchmark.sqlite
  python3 experiments/benchmark_db.py sample --db experiments/benchmark.sqlite
  python3 experiments/benchmark_db.py report --db experiments/benchmark.sqlite
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCHEMA = ROOT / "schema.sql"


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA.read_text(encoding="utf-8"))


def insert_sample(db_path: Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA.read_text(encoding="utf-8"))

        algo_ids = {r["name"]: r["id"] for r in conn.execute("SELECT id, name FROM algorithms")}

        sample_runs = [
            # dcu_units, total_cards, pre_ms, solver_ms, iters
            ("demo-2cards", 1, 2, 3.10, 6.25, 51),
            ("demo-4cards", 2, 4, 1.92, 4.83, 48),
            ("demo-6cards", 3, 6, 1.48, 4.21, 46),
        ]

        for run_tag, dcu_units, total_cards, pre_ms, solver_ms, iters in sample_runs:
            cur = conn.execute(
                """
                INSERT INTO experiment_runs (
                    run_tag, matrix_name, matrix_rows, matrix_cols, matrix_nnz,
                    dcu_mode, dcu_units, cards_per_dcu, total_cards,
                    node_count, partition_strategy, converged, solver_tol,
                    max_iters, total_iters, preconditioner_ms, gpupbicgstab_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_tag,
                    "A16x16_demo",
                    16,
                    16,
                    64,
                    "single_node_multi_card",
                    dcu_units,
                    2,
                    total_cards,
                    1,
                    "balanced_columns",
                    1,
                    1e-8,
                    1000,
                    iters,
                    pre_ms,
                    solver_ms,
                ),
            )
            run_id = cur.lastrowid

            rows = [
                (run_id, algo_ids["static_single_card"], pre_ms + 0.30, solver_ms + 0.25, iters + 2, 1, 9.1e-9),
                (run_id, algo_ids["static_multi_card"], pre_ms, solver_ms, iters, 1, 8.8e-9),
                (run_id, algo_ids["dynamic_single_card"], pre_ms + 0.15, solver_ms + 0.10, iters - 1, 1, 8.6e-9),
                (run_id, algo_ids["dynamic_multi_card"], pre_ms - 0.10, solver_ms - 0.08, iters - 2, 1, 8.3e-9),
                (run_id, algo_ids["cusparse_csrilu0"], pre_ms + 0.55, solver_ms + 0.21, iters - 3, 1, 8.0e-9),
                (run_id, algo_ids["viennacl_sspai_vcl"], pre_ms + 0.72, solver_ms + 0.41, iters - 4, 1, 7.9e-9),
                (run_id, algo_ids["gspai_adaptive"], pre_ms + 0.61, solver_ms + 0.05, iters - 5, 1, 7.7e-9),
            ]

            conn.executemany(
                """
                INSERT INTO run_algorithm_metrics (
                    run_id, algorithm_id, preconditioner_ms,
                    gpupbicgstab_ms, iteration_count, converged, residual_norm
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )


def report(db_path: Path) -> None:
    query = """
    SELECT
        er.run_tag,
        er.total_cards,
        a.name AS algorithm,
        ram.preconditioner_ms,
        ram.gpupbicgstab_ms,
        ram.total_ms,
        ram.iteration_count,
        ram.converged
    FROM run_algorithm_metrics ram
    JOIN experiment_runs er ON er.id = ram.run_id
    JOIN algorithms a ON a.id = ram.algorithm_id
    ORDER BY er.total_cards ASC, ram.total_ms ASC;
    """
    with connect(db_path) as conn:
        rows = conn.execute(query).fetchall()

    if not rows:
        print("No benchmark records.")
        return

    for row in rows:
        print(
            f"{row['run_tag']} | cards={row['total_cards']} | {row['algorithm']} | "
            f"pre={row['preconditioner_ms']:.2f} ms | solver={row['gpupbicgstab_ms']:.2f} ms | "
            f"total={row['total_ms']:.2f} ms | iters={row['iteration_count']} | conv={row['converged']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["init", "sample", "report"])
    parser.add_argument("--db", required=True, type=Path)
    args = parser.parse_args()

    args.db.parent.mkdir(parents=True, exist_ok=True)

    if args.action == "init":
        init_db(args.db)
    elif args.action == "sample":
        insert_sample(args.db)
    else:
        report(args.db)


if __name__ == "__main__":
    main()
