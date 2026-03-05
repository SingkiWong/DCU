# 多DCU实验与数据库规范（四算法分包版）

本版本按你的要求统一为：**静态/动态 × 单卡/多卡 = 4个算法包**，并明确多DCU口径：

- **1个DCU = 2张卡**
- **2个DCU = 4张卡**
- **3个DCU = 6张卡**

---

## 1. 算法库接口（两个算法）

算法库只保留两个接口（原理和流程保持一致）：

1. `RunStaticSPAI_MultiDCU_MPI(...)`：静态算法（balanced_columns）
2. `RunDynamicSPAI_MultiDCU_MPI(...)`：动态算法（balanced_nnz）

四个运行包仍可保留用于不同部署场景（单卡/多卡、MPI与否）。

---

## 2. 划分要求（均衡分块）

### 2.1 列均衡（推荐）

以 A 为 `16 x 16` 为例：

- `1 DCU (2 cards)`：每个DCU分配 `16` 列
- `2 DCU (4 cards)`：每个DCU分配 `8` 列
- `3 DCU (6 cards)`：建议按 `6/5/5` 列（或按NNZ近似均衡）

统一公式：

```text
base = ncol / dcu_units
rem  = ncol % dcu_units
cols_i = base + (i < rem ? 1 : 0)
```

### 2.2 多节点MPI映射

- `global_rank -> (node_id, local_dcu_id)`
- 每个rank负责一个子列块
- 动态多卡算法在求解阶段使用MPI同步边界向量

---

## 3. 实验规模

固定三组结果：

- `1 DCU / 2 cards`
- `2 DCU / 4 cards`
- `3 DCU / 6 cards`

每个算法都应给出以上三组结果（若硬件不足，需标注缺测原因）。

---

## 4. 指标

### 4.1 有效性分析
- `converged`（是否收敛）
- `iteration_count`（收敛迭代数）
- `total_ms`（总时间）

### 4.2 性能分析
- `preconditioner_ms`
- `gpupbicgstab_ms`
- `iteration_count`

并统一：

```text
total_ms = preconditioner_ms + gpupbicgstab_ms
```

---

## 5. 对比库算法（基线）

建议保留以下对比项：
- cuSPARSE `CSRILU0`
- ViennaCL `SSPAI-VCL`
- `GSPAI-Adaptive`

---

## 6. 数据库存储

- `experiments/schema.sql`
- `experiments/benchmark_db.py`
- `experiments/results/`（建议放CSV/JSON汇总）
