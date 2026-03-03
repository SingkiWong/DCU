-- Benchmark DB schema for DCU SPAI / GPUPBICGSTAB experiments

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS algorithms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    mode TEXT NOT NULL CHECK(mode IN ('single_dcu_static', 'multi_dcu_static_dynamic', 'baseline')),
    backend TEXT NOT NULL,
    preconditioner TEXT NOT NULL,
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
    dcu_count INTEGER NOT NULL CHECK(dcu_count IN (1, 2, 4)),
    node_count INTEGER NOT NULL,
    partition_strategy TEXT NOT NULL CHECK(partition_strategy IN ('balanced_columns', 'balanced_nnz')),
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
    residual_norm REAL,
    total_ms REAL GENERATED ALWAYS AS (preconditioner_ms + gpupbicgstab_ms) STORED,
    FOREIGN KEY(run_id) REFERENCES experiment_runs(id) ON DELETE CASCADE,
    FOREIGN KEY(algorithm_id) REFERENCES algorithms(id) ON DELETE CASCADE,
    UNIQUE(run_id, algorithm_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_dcu ON experiment_runs(dcu_mode, dcu_count);
CREATE INDEX IF NOT EXISTS idx_runs_matrix ON experiment_runs(matrix_name);
CREATE INDEX IF NOT EXISTS idx_metrics_total ON run_algorithm_metrics(total_ms);

-- Seed algorithms required by the benchmark spec.
INSERT OR IGNORE INTO algorithms (name, mode, backend, preconditioner, notes) VALUES
    ('single_dcu_static_spai', 'single_dcu_static', 'HIP', 'Static SPAI', '单DCU静态算法'),
    ('multi_dcu_static_dynamic_spai', 'multi_dcu_static_dynamic', 'HIP+MPI', 'Static+Dynamic SPAI', '多DCU一静一动算法'),
    ('cusparse_csrilu0', 'baseline', 'cuSPARSE', 'CSRILU0', '不完全LU分解预条件'),
    ('viennacl_sspai_vcl', 'baseline', 'ViennaCL', 'SSPAI-VCL', '稀疏近似逆预条件'),
    ('gspai_adaptive', 'baseline', 'GSPAI', 'Adaptive SPAI', '自适应GSPAI算法');
