# StaticSPAIv20_ColumnRange 函数修改指南

## 核心修改思路

多DCU版本中，每个DCU只处理矩阵的一部分列。需要将单DCU版本中处理**全部列**的代码改为处理**部分列**。

## 关键修改点

### 1. 列数修改
```cpp
// 单DCU版本 ❌
devA->nCol          // 总列数 (例如: 10000)

// 多DCU版本 ✅
localCols           // 局部列数 (例如: DCU 0 处理 0-2500, localCols=2500)
```

### 2. 列指针偏移
```cpp
// 单DCU版本 ❌
devA->mPtr          // 从第0列开始

// 多DCU版本 ✅
devA->mPtr + colStart   // 从 colStart 列开始
```

### 3. 典型修改示例

#### 示例1: 计算 n2max
```cpp
// 单DCU版本 (spai_single_dcu.cpp 第1230行)
cuComputeN2MAXwithSparityofA<threadsPerBlock><<<blocksPerGrid, threadsPerBlock>>>(
    devA->mPtr,         // 从第0列开始
    devA->n,            // 处理所有列
    dev_n2
);

// 多DCU版本 (修改后)
cuComputeN2MAXwithSparityofA<threadsPerBlock><<<blocksPerGrid, threadsPerBlock>>>(
    devA->mPtr + colStart,   // ✅ 从 colStart 列开始
    localCols,               // ✅ 只处理 localCols 列
    dev_n2
);
```

#### 示例2: 分配内存
```cpp
// 单DCU版本
hipMalloc((void**)&dev_n1, sizeof(int) * devA->nCol);       // 为所有列分配

// 多DCU版本
hipMalloc((void**)&dev_n1, sizeof(int) * localCols);        // ✅ 只为局部列分配
```

#### 示例3: 计算 blocksPerGrid
```cpp
// 单DCU版本
blocksPerGrid = (devA->n - 1) / (threadsPerBlock / WarpSize) + 1;

// 多DCU版本
blocksPerGrid = (localCols - 1) / (threadsPerBlock / WarpSize) + 1;   // ✅
```

#### 示例4: 调用 kernel
```cpp
// 单DCU版本
cuComputeN1withSparityofA_SpMMv1<2, 2048><<<blocksPerGrid, threadsPerBlock>>>(
    devA->mPtr,         // ❌ 全部列
    devA->mIndex,
    devA->mPtr,         // ❌ 全部列
    devA->mIndex,
    dev_tI,
    ssize,
    devA->nCol,         // ❌ 总列数
    dev_n1,
    dev_atomic
);

// 多DCU版本
cuComputeN1withSparityofA_SpMMv1<2, 2048><<<blocksPerGrid, threadsPerBlock>>>(
    devA->mPtr + colStart,   // ✅ 从 colStart 开始
    devA->mIndex,
    devA->mPtr + colStart,   // ✅ 同样偏移
    devA->mIndex,
    dev_tI,
    ssize,
    localCols,               // ✅ 局部列数
    dev_n1,
    dev_atomic
);
```

## 完整修改流程

### 步骤1: 计算 n2max
- 传入 `devA->mPtr + colStart`
- 使用 `localCols`

### 步骤2: 计算 n1max
- 分配大小为 `localCols` 的数组
- 传入 `devA->mPtr + colStart` 和 `localCols`

### 步骤3: 计算 J (模式集)
```cpp
computeJ_SSPAIv10<<<...>>>(
    devA->mPtr + colStart,    // ✅
    devA->mIndex,
    localCols,                // ✅
    dev_J, dev_jPTR, n2max
);
```

### 步骤4: 计算 I (行索引集)
```cpp
computeI_Symbol_SpMMv1<<<...>>>(
    devA->mPtr,               // ⚠️ 注意：这里不需要偏移
    devA->mIndex,             //    因为要访问整个矩阵的行
    localCols,                // ✅ 但列数是局部的
    dev_I, dev_iPTR, n1max,
    dev_J, dev_jPTR, n2max,
    dev_atomic
);
```

### 步骤5: 组装 tildeA
```cpp
ComputeTildeACSR_SSPAIv10<<<...>>>(
    dev_tildeA,
    devA->mData,
    devA->mPtr,               // ⚠️ 这里也不需要偏移
    devA->mIndex,
    localCols,                // ✅ 局部列数
    dev_I, dev_iPTR,
    dev_J, dev_jPTR,
    n1max, n2max
);
```

### 步骤6: QR 分解
```cpp
QR_RShared_SSPAIv10<<<...>>>(
    dev_tildeA,
    dev_R,
    dev_iPTR,
    dev_jPTR,
    n1max,
    n2max,
    localCols                 // ✅ 局部列数
);
```

### 步骤7: 计算 E 和求解 X
```cpp
// 特殊注意：E 的计算需要考虑全局列索引
ComputeTildeE2_SSPAIv10<<<...>>>(
    dev_E,
    dev_I,
    dev_iPTR,
    n1max,
    localCols,                // ✅ 局部列数
    colStart                  // ✅ 传入起始列号，用于匹配全局列索引
);

Sol_SSPAIv10<<<...>>>(
    dev_tildeA,
    dev_R,
    dev_X,
    dev_E,
    dev_jPTR,
    n1max,
    n2max,
    localCols                 // ✅ 局部列数
);
```

### 步骤8: 存储结果
```cpp
// 将结果写入 devM（局部矩阵）
modifyIndexAndData_SSPAIv10<<<...>>>(
    devM->mData,
    devM->mIndex,
    devM->mPtr,
    dev_X,
    dev_J,
    dev_jPTR,
    n2max,
    localCols                 // ✅ 局部列数
);
```

## 重要注意事项

### ⚠️ 什么时候需要偏移？
| 操作 | 是否需要偏移 | 原因 |
|------|------------|------|
| 访问列指针 (mPtr) | ✅ 需要 | 每个DCU处理不同的列 |
| 列数 (nCol) | ✅ 改为 localCols | 只处理局部列 |
| 访问行索引 (mIndex) | ❌ 不需要 | 行索引是全局的 |
| 访问数据 (mData) | ❌ 不需要 | 通过全局指针访问 |
| 矩阵总维度 (n) | ❌ 不需要 | 保持不变 |

### ⚠️ 特殊处理：E 的计算
```cpp
// E 是单位向量的位置，需要匹配全局列号
// 例如: DCU 0 处理列 [0, 2500)
//       DCU 0 的局部列 100 对应全局列 100
// 例如: DCU 1 处理列 [2500, 5000)
//       DCU 1 的局部列 100 对应全局列 2600

// 在 ComputeTildeE2_SSPAIv10 中:
for (col = 0; col < localCols; col++) {
    int globalCol = col + colStart;    // ✅ 计算全局列号
    for (i = 0; i < iPTR[col]; i++) {
        if (I[col * n1max + i] == globalCol) {   // ✅ 与全局列号比较
            E[col] = i;
            break;
        }
    }
}
```

## 快速检查清单

修改完成后，检查以下各项：

- [ ] 所有 `devA->nCol` 改为 `localCols`
- [ ] 所有 `devA->n` 在表示列数时改为 `localCols`（表示矩阵维度时保持不变）
- [ ] Kernel 调用中传入列指针的地方加上 `+ colStart`
- [ ] 内存分配大小使用 `localCols` 而不是 `devA->nCol`
- [ ] blocksPerGrid 的计算使用 `localCols`
- [ ] E 的计算考虑了 `colStart` 偏移
- [ ] devM 的结构正确初始化（nCol = localCols）
- [ ] 所有循环 `for (i = 0; i < devA->nCol; i++)` 改为 `for (i = 0; i < localCols; i++)`

## 示例：完整的函数框架

```cpp
float StaticSPAIv20_ColumnRange(CSC_Matrix *devA, CSC_Matrix *devM,
                                 int colStart, int colEnd, int deviceId,
                                 hipStream_t stream) {
    // 1. 验证参数
    int localCols = colEnd - colStart;

    // 2. 计算 n2max (使用 devA->mPtr + colStart, localCols)

    // 3. 计算 n1max (使用 localCols)

    // 4. 分配工作内存 (大小为 localCols * ...)

    // 5. 计算 J (使用 devA->mPtr + colStart, localCols)

    // 6. 计算 I (使用 localCols)

    // 7. 组装 tildeA (使用 localCols)

    // 8. QR 分解 (使用 localCols)

    // 9. 计算 E 和求解 X (传入 colStart, 使用 localCols)

    // 10. 存储到 devM (使用 localCols)

    // 11. 清理内存

    return totalTime;
}
```

## 测试建议

1. 先用2个DCU测试，每个DCU处理一半的列
2. 验证结果与单DCU版本一致
3. 逐步增加DCU数量
4. 检查边界情况（最后一个DCU的列数可能不同）
