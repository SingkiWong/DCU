# 包1：静态 + 单卡

## 对应实现
- 可执行程序：`spai_single_dcu`
- 源码：`src/spai_single_dcu.cpp`
- 编译：`make -f Makefile.single`

## 运行
```bash
./spai_single_dcu --strategy static --matrix matrices/circuit_2.mtx
```

## 采集指标
- 预条件子时间（preconditioner_ms）
- GPUPBICGSTAB时间（gpupbicgstab_ms）
- 迭代次数（iteration_count）
- 是否收敛（converged）
