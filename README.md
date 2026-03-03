# DCU-SPAI（精简版）

本仓库已完成清理：
- 删除历史旧文档与过期报告
- 删除未使用示例/旧程序
- 保留核心源码、构建文件、实验脚本与结果模板

## 当前目录

```text
.
├── src/                       # 主程序源码（single/multi）
├── include/                   # 头文件
│   ├── common/
│   └── bicgstab/
├── matrices/                  # 测试矩阵
├── algorithm_packages/        # 四算法运行入口
├── experiments/               # 数据库schema、工具、结果模板
├── docs/
│   └── notes/                 # 实验规范说明
├── Makefile.single
├── Makefile.multi
└── run_all_packages.sh
```

## 构建

```bash
# 单卡
make -f Makefile.single

# 多卡
make -f Makefile.multi multi

# MPI 版本
make -f Makefile.multi mpi
```

## 运行

```bash
# 单卡静态
./spai_single_dcu --strategy static --matrix matrices/circuit_2.mtx

# 单卡动态（策略标签）
./spai_single_dcu --strategy dynamic --matrix matrices/circuit_2.mtx

# 多卡静态（列均衡）
./spai_multi_dcu 2 --partition balanced_columns --matrix matrices/circuit_2.mtx

# 多卡动态（NNZ均衡，预条件模式）
./spai_multi_dcu 2 --partition balanced_nnz --precondition-only --matrix matrices/circuit_2.mtx
```

## 一键运行分包

```bash
./run_all_packages.sh matrices/circuit_2.mtx
```

## 实验数据

```bash
python3 experiments/benchmark_db.py init --db experiments/benchmark.sqlite
python3 experiments/benchmark_db.py sample --db experiments/benchmark.sqlite
python3 experiments/benchmark_db.py report --db experiments/benchmark.sqlite
```

详细实验约束见：`docs/notes/EXPERIMENT_SPEC_MULTI_DCU.md`。
