# 算法分包总览（已统一）

本目录提供 4 个实验入口包，用于区分静态/动态 + 单卡/多卡：

1. `01_static_single_card`
2. `02_static_multi_card`
3. `03_dynamic_single_card`
4. `04_dynamic_multi_card`

统一口径：
- 1 DCU = 2 cards
- 2 DCU = 4 cards
- 3 DCU = 6 cards

统一测试建议：

```bash
# 多卡接口烟雾测试
make -f Makefile.multi test-api

# 或直接脚本
./experiments/test_api_cli.sh ./spai_multi_dcu_mpi matrices/circuit_2.mtx 1
```

接口源码位置：
- `include/common/spai_multi_dcu_api.h`
- `src/spai_multi_dcu.cpp`

可配合 `run_all_packages.sh` 批量执行。
