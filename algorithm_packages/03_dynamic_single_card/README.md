# 包3：动态 + 单卡

## 对应实现
- 可执行程序：`spai_single_dcu`
- 源码：`src/spai_single_dcu.cpp`
- 编译：`make -f Makefile.single`

## 运行
```bash
./spai_single_dcu --strategy dynamic --matrix matrices/circuit_2.mtx
```

## 指标建议
- `preconditioner_ms`
- `gpupbicgstab_ms`
- `iteration_count`
- `converged`

## 备注
该包用于单 DCU 动态策略标签实验；多卡动态接口见包4。
