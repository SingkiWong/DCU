#!/usr/bin/env python3
"""Utilities for DCU SPAI benchmark database.

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
    schema = SCHEMA.read_text(encoding="utf-8")
    with connect(db_path) as conn:
        conn.executescript(schema)


def insert_sample(db_path: Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA.read_text(encoding="utf-8"))

        cur = conn.execute(
            """
            INSERT INTO experiment_runs (
                run_tag, matrix_name, matrix_rows, matrix_cols, matrix_nnz,
                dcu_mode, dcu_count, node_count, partition_strategy,
                converged, solver_tol, max_iters, total_iters,
                preconditioner_ms, gpupbicgstab_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "demo-16x16-balanced-2dcu",
                "A16x16_demo",
                16,
                16,
                64,
                "single_node_multi_card",
                2,
                1,
                "balanced_columns",
                1,
                1e-8,
                1000,
                42,
                2.31,
                5.42,
            ),
        )
        run_id = cur.lastrowid

        algo_ids = {
            r["name"]: r["id"]
            for r in conn.execute("SELECT id, name FROM algorithms")
        }

        rows = [
            (run_id, algo_ids["single_dcu_static_spai"], 2.31, 5.42, 42, 1, 9.2e-9),
            (run_id, algo_ids["cusparse_csrilu0"], 3.85, 5.71, 39, 1, 8.7e-9),
            (run_id, algo_ids["viennacl_sspai_vcl"], 4.26, 6.03, 37, 1, 9.9e-9),
            (run_id, algo_ids["gspai_adaptive"], 4.95, 5.18, 35, 1, 7.8e-9),
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
        er.matrix_name,
        er.dcu_mode,
        er.dcu_count,
        a.name AS algorithm,
        ram.preconditioner_ms,
        ram.gpupbicgstab_ms,
        ram.total_ms,
        ram.iteration_count,
        ram.converged
    FROM run_algorithm_metrics ram
    JOIN experiment_runs er ON er.id = ram.run_id
    JOIN algorithms a ON a.id = ram.algorithm_id
    ORDER BY er.id DESC, ram.total_ms ASC;
    """
    with connect(db_path) as conn:
        rows = conn.execute(query).fetchall()

    if not rows:
        print("No benchmark records.")
        return

    for row in rows:
        print(
            f"{row['run_tag']} | {row['algorithm']} | DCU={row['dcu_count']} | "
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
