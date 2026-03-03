# 包2：静态 + 多卡

## 对应实现
- 可执行程序：`spai_multi_dcu`
- 源码：`spai_multi_dcu.cpp`
- 编译：`make -f Makefile.multi multi`

## 卡数口径
- 1 DCU = 2 cards
- 2 DCU = 4 cards
- 3 DCU = 6 cards

## 运行示例
```bash
# 2卡（1 DCU）
echo "circuit_2.mtx" | ./spai_multi_dcu 1

# 4卡（2 DCU）
echo "circuit_2.mtx" | ./spai_multi_dcu 2

# 6卡（3 DCU）
echo "circuit_2.mtx" | ./spai_multi_dcu 3
```
