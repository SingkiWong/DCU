-- Benchmark DB schema for DCU SPAI / GPUPBICGSTAB experiments (4 algorithm packages)

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS algorithms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    class TEXT NOT NULL CHECK(class IN ('project_algo', 'baseline')),
    dimension TEXT NOT NULL CHECK(dimension IN ('static_single', 'static_multi', 'dynamic_single', 'dynamic_multi', 'baseline')),
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
    dcu_units INTEGER NOT NULL CHECK(dcu_units IN (1, 2, 3)),
    cards_per_dcu INTEGER NOT NULL DEFAULT 2 CHECK(cards_per_dcu = 2),
    total_cards INTEGER NOT NULL CHECK(total_cards IN (2, 4, 6)),
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

CREATE INDEX IF NOT EXISTS idx_runs_cards ON experiment_runs(dcu_mode, dcu_units, total_cards);
CREATE INDEX IF NOT EXISTS idx_runs_matrix ON experiment_runs(matrix_name);
CREATE INDEX IF NOT EXISTS idx_metrics_total ON run_algorithm_metrics(total_ms);

-- Seed required project algorithms + baseline algorithms.
INSERT OR IGNORE INTO algorithms (name, class, dimension, backend, preconditioner, notes) VALUES
    ('static_single_card', 'project_algo', 'static_single', 'HIP', 'Static SPAI', '包1：静态+单卡'),
    ('static_multi_card', 'project_algo', 'static_multi', 'HIP', 'Static SPAI', '包2：静态+多卡'),
    ('dynamic_single_card', 'project_algo', 'dynamic_single', 'HIP', 'Dynamic SPAI', '包3：动态+单卡'),
    ('dynamic_multi_card', 'project_algo', 'dynamic_multi', 'HIP+MPI', 'Dynamic SPAI', '包4：动态+多卡'),
    ('cusparse_csrilu0', 'baseline', 'baseline', 'cuSPARSE', 'CSRILU0', '不完全LU分解预条件'),
    ('viennacl_sspai_vcl', 'baseline', 'baseline', 'ViennaCL', 'SSPAI-VCL', '稀疏近似逆预条件'),
    ('gspai_adaptive', 'baseline', 'baseline', 'GSPAI', 'Adaptive SPAI', '自适应GSPAI算法');
