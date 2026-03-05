# Multi-DCU + MPI SPAI 求解器说明

本项目聚焦多 DCU + MPI 的 SPAI 预条件子流程，支持两种算法模式：

- `RunStaticSPAI_MultiDCU_MPI(...)`：静态模式（按列均衡，`balanced_columns`）
- `RunDynamicSPAI_MultiDCU_MPI(...)`：动态模式（按 NNZ 均衡，`balanced_nnz`）

---

## 1. 接口在哪里

- **接口声明**：`include/common/spai_multi_dcu_api.h`
- **接口实现**：`src/spai_multi_dcu.cpp`
- **算法入口函数**：
  - `RunStaticSPAI_MultiDCU_MPI(...)`
  - `RunDynamicSPAI_MultiDCU_MPI(...)`
- **统一核心实现**：`StaticSPAIv20_MultiDCU(..., bool useNnzBalance)`
- **动态/静态分歧点**：`distributeColumns(..., useNnzBalance)`

---

## 2. 命令行怎么调用

先编译 MPI 版本：

```bash
make -f Makefile.multi mpi
```

静态：

```bash
mpirun -np 2 ./spai_multi_dcu_mpi 2 \
  --algo static \
  --partition balanced_columns \
  --matrix matrices/circuit_2.mtx
```

动态：

```bash
mpirun -np 2 ./spai_multi_dcu_mpi 2 \
  --algo dynamic \
  --partition balanced_nnz \
  --matrix matrices/circuit_2.mtx
```

查看帮助：

```bash
./spai_multi_dcu_mpi --help
```

> 若不显式传 `--partition`，程序会按 `--algo` 自动映射：
> - static -> balanced_columns
> - dynamic -> balanced_nnz

---

## 3. 输出与指标

程序输出统一报告行：

```text
[ALGO_REPORT] mode=..., dcu_units=..., partition=..., preconditioner_ms=..., iteration_count=...
```

关键字段：
- `preconditioner_ms`：SPAI 预条件子构建时间
- `iteration_count`：BiCGSTAB 迭代次数（`--precondition-only` 下为 0）

---

## 4. 接口如何测试

推荐使用内置烟雾测试：

```bash
# 1) 编译
make -f Makefile.multi mpi

# 2) 跑脚本
./experiments/test_api_cli.sh ./spai_multi_dcu_mpi matrices/circuit_2.mtx 1

# 或者一条命令
make -f Makefile.multi test-api
```

`test_api_cli.sh` 会自动检查：
- static/dynamic 模式映射是否正确
- `--partition` 覆盖是否生效
- `[ALGO_REPORT]` 是否存在且字段格式正确

---

## 5. 目录速览

- 主程序：`src/spai_multi_dcu.cpp`
- 单卡程序：`src/spai_single_dcu.cpp`
- 求解器头：`include/bicgstab/bicgstab_solver.h`
- 多 DCU API 头：`include/common/spai_multi_dcu_api.h`
- 实验脚本：`experiments/test_api_cli.sh`
- 分包说明：`algorithm_packages/README.md`
