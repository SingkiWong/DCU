# 多DCU静态SPAI快速参考

## 快速命令

```bash
# 编译多DCU版本
make -f Makefile.multiDCU

# 运行（使用2个DCU）
echo "circuit_2.mtx" | ./DtestStaticSPAINew30_multiDCU 2

# 性能测试
make -f Makefile.multiDCU test

# 正确性验证
make -f Makefile.multiDCU verify
```

## 文件说明

- `DtestStaticSPAINew30_multiDCU.cpp` - 多DCU主程序（框架）
- `MULTI_DCU_GUIDE.md` - 详细实现指南
- `Makefile.multiDCU` - 多DCU编译配置
