-- Benchmark DB schema for DCU SPAI / GPUPBICGSTAB experiments (2 algorithm interfaces)

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS algorithms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    class TEXT NOT NULL CHECK(class IN ('project_algo')),
    backend TEXT NOT NULL,
    partition_strategy TEXT NOT NULL CHECK(partition_strategy IN ('balanced_columns', 'balanced_nnz')),
    notes TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_tag TEXT NOT NULL,
    matrix_name TEXT NOT NULL,
    matrix_rows INTEGER NOT NULL,
    matrix_cols INTEGER NOT NULL,
    matrix_nnz INTEGER NOT NULL,
    dcu_mode TEXT NOT NULL CHECK(dcu_mode IN ('single_node_multi_card', 'multi_node_multi_card_mpi_hip')),
    dcu_units INTEGER NOT NULL CHECK(dcu_units IN (1, 2, 3)),
    cards_per_dcu INTEGER NOT NULL DEFAULT 2 CHECK(cards_per_dcu = 2),
    total_cards INTEGER NOT NULL CHECK(total_cards IN (2, 4, 6)),
    node_count INTEGER NOT NULL,
    converged INTEGER NOT NULL CHECK(converged IN (0, 1)),
    solver_tol REAL NOT NULL,
    max_iters INTEGER NOT NULL,
    total_iters INTEGER NOT NULL,
    preconditioner_ms REAL NOT NULL,
    gpupbicgstab_ms REAL NOT NULL,
    total_ms REAL GENERATED ALWAYS AS (preconditioner_ms + gpupbicgstab_ms) STORED,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS run_algorithm_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    algorithm_id INTEGER NOT NULL,
    preconditioner_ms REAL NOT NULL,
    gpupbicgstab_ms REAL NOT NULL,
    iteration_count INTEGER NOT NULL,
    converged INTEGER NOT NULL CHECK(converged IN (0, 1)),
    total_ms REAL GENERATED ALWAYS AS (preconditioner_ms + gpupbicgstab_ms) STORED,
    FOREIGN KEY(run_id) REFERENCES experiment_runs(id) ON DELETE CASCADE,
    FOREIGN KEY(algorithm_id) REFERENCES algorithms(id) ON DELETE CASCADE,
    UNIQUE(run_id, algorithm_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_cards ON experiment_runs(dcu_mode, dcu_units, total_cards);
CREATE INDEX IF NOT EXISTS idx_runs_matrix ON experiment_runs(matrix_name);
CREATE INDEX IF NOT EXISTS idx_metrics_total ON run_algorithm_metrics(total_ms);

-- Two algorithm interfaces only.
INSERT OR IGNORE INTO algorithms (name, class, backend, partition_strategy, notes) VALUES
    ('static_spai_mpi', 'project_algo', 'HIP+MPI', 'balanced_columns', '接口: RunStaticSPAI_MultiDCU_MPI'),
    ('dynamic_spai_mpi', 'project_algo', 'HIP+MPI', 'balanced_nnz', '接口: RunDynamicSPAI_MultiDCU_MPI');
