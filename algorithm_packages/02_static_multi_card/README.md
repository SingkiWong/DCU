# 包2：静态 + 多卡（Multi-DCU / MPI）

## 对应实现
- 可执行程序：`spai_multi_dcu` / `spai_multi_dcu_mpi`
- 源码：`src/spai_multi_dcu.cpp`
- 接口声明：`include/common/spai_multi_dcu_api.h`
- 关键接口：`RunStaticSPAI_MultiDCU_MPI(...)`

## 编译
```bash
make -f Makefile.multi multi
make -f Makefile.multi mpi
```

## 运行示例
```bash
# 非 MPI
./spai_multi_dcu 1 --algo static --partition balanced_columns --matrix matrices/circuit_2.mtx

# MPI
mpirun -np 2 ./spai_multi_dcu_mpi 2 --algo static --partition balanced_columns --matrix matrices/circuit_2.mtx
```

## 卡数口径
- 1 DCU = 2 cards
- 2 DCU = 4 cards
- 3 DCU = 6 cards
