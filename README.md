# DCU SPAI 项目（已按目录整理）

本仓库已完成目录归类，核心目标是：
- 静态/动态预条件策略
- 单卡/多卡（含 MPI+HIP）
- 统一实验记录（2/4/6 cards，对应 1/2/3 DCU）

## 目录结构

```text
.
├── src/                    # 主程序源码（single/multi/duospai）
├── include/
│   ├── common/            # 公共头文件
│   └── bicgstab/          # BiCGSTAB相关头文件
├── matrices/              # 测试矩阵
├── algorithm_packages/    # 四算法分包入口
├── experiments/           # 数据库schema、脚本、结果模板
├── docs/                  # 指南/报告/说明
├── scripts/               # 辅助脚本
├── examples/              # 示例程序
├── legacy/                # 历史代码
├── Makefile.single
├── Makefile.multi
└── run_all_packages.sh
```

## 四算法分包

- `algorithm_packages/01_static_single_card`
- `algorithm_packages/02_static_multi_card`
- `algorithm_packages/03_dynamic_single_card`
- `algorithm_packages/04_dynamic_multi_card`

> 多DCU口径：`1 DCU = 2 cards`，重点实验规模：2/4/6 cards（即 1/2/3 DCU）。

## 常用命令

```bash
# 单卡
make -f Makefile.single

# 多卡
make -f Makefile.multi multi

# MPI + HIP
make -f Makefile.multi mpi

# 一键运行分包流程（示例）
./run_all_packages.sh matrices/circuit_2.mtx
```

## 实验数据库

- schema：`experiments/schema.sql`
- 工具：`experiments/benchmark_db.py`
- 规范：`docs/notes/EXPERIMENT_SPEC_MULTI_DCU.md`

```bash
python3 experiments/benchmark_db.py init --db experiments/benchmark.sqlite
python3 experiments/benchmark_db.py sample --db experiments/benchmark.sqlite
python3 experiments/benchmark_db.py report --db experiments/benchmark.sqlite
```
