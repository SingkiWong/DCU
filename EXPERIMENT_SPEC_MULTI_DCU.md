# 多DCU实验与数据库规范（静态/动态 + MPI+HIP）

本规范用于统一以下实验场景：

1. **单点多卡（静态+动态代码）**：同一节点内多个DCU，HIP实现。
2. **多点多卡**：MPI + HIP实现。
3. **均衡划分策略**：对多DCU按列均衡划分（必要时可按NNZ均衡）。
4. **固定实验规模**：DCU数量固定为 `1 / 2 / 4`。

---

## 1. 划分要求

### 1.1 列均衡划分

以 A 为 `16 x 16` 为例：

- `1 DCU`：每个DCU分配 `16` 列。
- `2 DCU`：每个DCU分配 `8` 列。
- `4 DCU`：每个DCU分配 `4` 列。

建议实现形式：

```text
base = ncol / dcu_count
rem  = ncol % dcu_count
cols_i = base + (i < rem ? 1 : 0)
```

### 1.2 多节点MPI映射

- `global_rank -> (node_id, local_dcu_id)`。
- 每个rank负责一个列分块。
- 预条件子可局部构建，求解阶段按MPI通信同步向量边界。

---

## 2. 实验输出指标

### 2.1 有效性分析

- 收敛状态（是否收敛）
- 收敛迭代次数（iteration_count）
- 执行时间（总时间）

### 2.2 性能分析

每次实验至少输出：

- 预条件子计算时间（preconditioner_ms）
- GPUPBICGSTAB计算时间（gpupbicgstab_ms）
- 迭代次数（iteration_count）

总执行时间定义为：

```text
total_ms = preconditioner_ms + gpupbicgstab_ms
```

---

## 3. 算法集合（数据库中必须包含）

1. 单DCU静态算法：`single_dcu_static_spai`
2. 多DCU一静一动算法：`multi_dcu_static_dynamic_spai`
3. 对比算法（基线库）：
   - cuSPARSE: `CSRILU0`
   - ViennaCL: `SSPAI-VCL`
   - `GSPAI-Adaptive`

> 说明：对比集合可扩展，但库算法执行后必须保留上述性能字段。

---

## 4. 数据库存储

已提供 SQLite 方案：

- `experiments/schema.sql`
- `experiments/benchmark_db.py`

其中：

- `algorithms`：算法主数据。
- `experiment_runs`：实验配置与全局指标。
- `run_algorithm_metrics`：每个实验下各算法指标。

---

## 5. 推荐实验矩阵

- 小规模正确性：`16x16`、`32x32`
- 中规模性能：`circuit_2.mtx`
- 大规模压力：`fill_3elt.mtx`

每个矩阵在 `1/2/4 DCU` 下至少执行3次，建议报告平均值与标准差。
