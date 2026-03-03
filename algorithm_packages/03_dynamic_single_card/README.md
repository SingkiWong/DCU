# 包3：动态 + 单卡

## 说明
当前仓库主线已有静态SPAI实现；该包用于动态策略单卡实验入口统一。

## 推荐入口
- 通过 `--strategy dynamic` 启用动态策略标签（用于实验区分）

## 运行
```bash
./spai_single_dcu --strategy dynamic --matrix matrices/circuit_2.mtx
```
