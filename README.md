# Multi-DCU + MPI SPAI 求解器说明

本项目当前聚焦 **多DCU + MPI** 运行流程，提供两个算法接口：

1. `RunStaticSPAI_MultiDCU_MPI(...)`：静态算法（列均衡分配）
2. `RunDynamicSPAI_MultiDCU_MPI(...)`：动态算法（按NNZ均衡分配）

两者核心原理一致：
- 都使用相同的 SPAI 预条件子构建流程
- 差异仅在任务划分策略（balanced_columns vs balanced_nnz）

---

## 0. 接口到底怎么调用？算法库在哪？

### 命令行调用（推荐）

编译后直接调用 `spai_multi_dcu_mpi`：

```bash
mpirun -np 2 ./spai_multi_dcu_mpi 2 --algo static --matrix matrices/circuit_2.mtx
mpirun -np 2 ./spai_multi_dcu_mpi 2 --algo dynamic --matrix matrices/circuit_2.mtx
```

也可以先看帮助：

```bash
./spai_multi_dcu_mpi --help
```

### 源码级接口（算法库位置）

当前项目不是单独 `.so/.a` 的外部库发布形态，而是**源码接口**：

- 头文件（接口声明）：`include/common/spai_multi_dcu_api.h`
- 实现文件（算法主体）：`src/spai_multi_dcu.cpp`

可直接调用两个接口：

- `RunStaticSPAI_MultiDCU_MPI(...)`
- `RunDynamicSPAI_MultiDCU_MPI(...)`

---

## 1. 运行前准备

- ROCm/HIP 环境可用
- MPI 环境可用（`mpirun`, `mpicc`）
- 矩阵文件在 `matrices/` 目录

编译：

```bash
make -f Makefile.multi mpi
```

---

## 2. 运行命令

### 静态算法（多DCU+MPI）

```bash
mpirun -np 2 ./spai_multi_dcu_mpi 2 \
  --algo static \
  --partition balanced_columns \
  --matrix matrices/circuit_2.mtx
```

### 动态算法（多DCU+MPI）

```bash
mpirun -np 2 ./spai_multi_dcu_mpi 2 \
  --algo dynamic \
  --partition balanced_nnz \
  --matrix matrices/circuit_2.mtx
```

> 若不显式指定 `--partition`，程序会按 `--algo` 自动选择：
> - static -> balanced_columns
> - dynamic -> balanced_nnz

---

## 3. 划分逻辑示例（16x16）

- 1 个 DCU：该 DCU 处理 16 列
- 2 个 DCU：每个 DCU 处理 8 列（均衡）
- 4 个 DCU：每个 DCU 处理 4 列（均衡）

动态模式下会尝试按列非零元数（NNZ）做更均衡分配。

---

## 4. 输出指标

程序会输出标准化行：

```text
[ALGO_REPORT] mode=..., dcu_units=..., partition=..., preconditioner_ms=..., iteration_count=...
```

含义：
- `preconditioner_ms`：预处理（SPAI）时间
- `iteration_count`：BiCGSTAB 迭代次数（已接入，不再用 -1 占位）

---

## 5. 算法流程

1. 读取矩阵 A（CSC）
2. 在每个 DCU 上复制 A
3. 按算法策略进行列分配（静态列均衡 / 动态NNZ均衡）
4. 并行构建局部预条件子
5. 聚合为全局预条件子 M
6. 在 rank0 上执行 BiCGSTAB 求解，返回迭代次数
7. 输出预处理时间和迭代次数

---

## 6. 相关文件

- 主程序：`src/spai_multi_dcu.cpp`
- 单卡程序：`src/spai_single_dcu.cpp`
- 求解器：`include/bicgstab/bicgstab_solver.h`
- 实验数据库：`experiments/schema.sql`, `experiments/benchmark_db.py`
