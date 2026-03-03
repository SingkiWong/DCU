# 四算法分包总览

为便于区分与独立运行，已将实验入口拆为四个算法包：

1. `01_static_single_card`：静态 + 单卡
2. `02_static_multi_card`：静态 + 多卡
3. `03_dynamic_single_card`：动态 + 单卡
4. `04_dynamic_multi_card`：动态 + 多卡（支持MPI+HIP）

> 说明：本项目中按你的设备口径，`1个DCU=2张卡`。因此多DCU实验常用规模为：
>
> - 1 DCU = 2 cards
> - 2 DCU = 4 cards
> - 3 DCU = 6 cards
