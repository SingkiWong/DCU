#!/usr/bin/env python3
"""Utilities for DCU SPAI benchmark database (2 algorithm interfaces).

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
            # run_tag, dcu_units, total_cards
            ("demo-2cards", 1, 2),
            ("demo-4cards", 2, 4),
            ("demo-6cards", 3, 6),
        ]

        for run_tag, dcu_units, total_cards in sample_runs:
            # static interface
            cur = conn.execute(
                """
                INSERT INTO experiment_runs (
                    run_tag, matrix_name, matrix_rows, matrix_cols, matrix_nnz,
                    dcu_mode, dcu_units, cards_per_dcu, total_cards,
                    node_count, converged, solver_tol,
                    max_iters, total_iters, preconditioner_ms, gpupbicgstab_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"{run_tag}-static",
                    "A16x16_demo",
                    16,
                    16,
                    64,
                    "multi_node_multi_card_mpi_hip",
                    dcu_units,
                    2,
                    total_cards,
                    max(1, dcu_units),
                    1,
                    1e-8,
                    1000,
                    48,
                    3.0 / dcu_units,
                    0.30,
                ),
            )
            run_id = cur.lastrowid
            conn.execute(
                """
                INSERT INTO run_algorithm_metrics (
                    run_id, algorithm_id, preconditioner_ms,
                    gpupbicgstab_ms, iteration_count, converged
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, algo_ids["static_spai_mpi"], 3.0 / dcu_units, 0.30, 48, 1),
            )

            # dynamic interface
            cur = conn.execute(
                """
                INSERT INTO experiment_runs (
                    run_tag, matrix_name, matrix_rows, matrix_cols, matrix_nnz,
                    dcu_mode, dcu_units, cards_per_dcu, total_cards,
                    node_count, converged, solver_tol,
                    max_iters, total_iters, preconditioner_ms, gpupbicgstab_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"{run_tag}-dynamic",
                    "A16x16_demo",
                    16,
                    16,
                    64,
                    "multi_node_multi_card_mpi_hip",
                    dcu_units,
                    2,
                    total_cards,
                    max(1, dcu_units),
                    1,
                    1e-8,
                    1000,
                    46,
                    2.7 / dcu_units,
                    0.30,
                ),
            )
            run_id = cur.lastrowid
            conn.execute(
                """
                INSERT INTO run_algorithm_metrics (
                    run_id, algorithm_id, preconditioner_ms,
                    gpupbicgstab_ms, iteration_count, converged
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, algo_ids["dynamic_spai_mpi"], 2.7 / dcu_units, 0.30, 46, 1),
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
    ORDER BY er.total_cards ASC, a.name ASC;
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
