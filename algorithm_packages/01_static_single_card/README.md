# 包1：静态 + 单卡

## 对应实现
- 可执行程序：`spai_single_dcu`
- 源码：`src/spai_single_dcu.cpp`
- 编译：`make -f Makefile.single`

## 运行
```bash
./spai_single_dcu --strategy static --matrix matrices/circuit_2.mtx
```

## 指标建议
- `preconditioner_ms`
- `gpupbicgstab_ms`
- `iteration_count`
- `converged`

## 备注
该包用于单 DCU 静态基线对比，多卡算法接口不在本包内。
