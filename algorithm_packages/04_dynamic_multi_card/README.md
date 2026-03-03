# 包4：动态 + 多卡（MPI + HIP）

## 对应实现
- 可执行程序：`spai_multi_dcu_mpi`
- 源码：`src/spai_multi_dcu.cpp`（`USE_MPI=1`）
- 编译：`make -f Makefile.multi mpi`

## 卡数口径
- 1 DCU = 2 cards
- 2 DCU = 4 cards
- 3 DCU = 6 cards

## 运行示例
```bash
# 2卡（1 DCU）
mpirun -np 1 ./spai_multi_dcu_mpi 1

# 4卡（2 DCU）
mpirun -np 2 ./spai_multi_dcu_mpi 2

# 6卡（3 DCU）
mpirun -np 3 ./spai_multi_dcu_mpi 3
```

> 说明：实际节点数与rank数可按机器拓扑调整；建议保证每rank可见本地DCU。
