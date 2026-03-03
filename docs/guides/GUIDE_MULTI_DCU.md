# 多DCU静态SPAI实现指南

## 📋 目录
1. [并行化策略](#并行化策略)
2. [核心修改点](#核心修改点)
3. [完整实现步骤](#完整实现步骤)
4. [性能优化建议](#性能优化建议)
5. [测试验证](#测试验证)

---

## 🎯 并行化策略

### 基本原理

静态SPAI预条件子计算具有**列独立性**：
```
M的第i列 = f(A的所有行, A[:,pattern[i]])
```

每一列的计算只依赖于：
- 输入矩阵A（全局只读）
- 当前列的稀疏模式

因此可以：
- ✅ **矩阵A复制到所有DCU**（只读共享）
- ✅ **按列分割矩阵M**（每个DCU负责一部分列）
- ✅ **完全并行计算**（无通信开销）
- ✅ **最后拼接结果**

### 数据分布

```
总列数 N = 1000, 使用 4个DCU

DCU 0: 列 0-249    (250列)
DCU 1: 列 250-499  (250列)
DCU 2: 列 500-749  (250列)
DCU 3: 列 750-999  (250列)

矩阵A: 每个DCU都有完整副本（只读）
矩阵M: 每个DCU只计算自己负责的列
```

---

## 🔧 核心修改点

### 1. StaticSPAIv20函数签名修改

**原始单DCU版本**:
```cpp
float StaticSPAIv20(CSC_Matrix *devA, CSC_Matrix *devM);
```

**多DCU版本**:
```cpp
float StaticSPAIv20_ColumnRange(
    CSC_Matrix *devA,           // 输入矩阵A（完整）
    CSC_Matrix *devM,           // 输出矩阵M（局部）
    int colStart,               // 起始列索引
    int colEnd,                 // 结束列索引（不包含）
    int deviceId                // 当前DCU的ID
);
```

### 2. 列范围参数传递

所有kernel启动都需要传递列范围：

**修改前**:
```cpp
cuComputeN2MAXwithSparityofA<256><<<blocksPerGrid, threadsPerBlock>>>(
    devA->mPtr, devA->n, dev_n2);
```

**修改后**:
```cpp
int localCols = colEnd - colStart;
cuComputeN2MAXwithSparityofA<256><<<blocksPerGrid, threadsPerBlock>>>(
    devA->mPtr + colStart,  // 从colStart开始
    localCols,              // 只处理localCols列
    dev_n2);
```

### 3. 内存分配调整

**原始版本** (分配全部列):
```cpp
hipMalloc((void**)&dev_n1, sizeof(int) * devA->nCol);
```

**多DCU版本** (只分配负责的列):
```cpp
int localCols = colEnd - colStart;
hipMalloc((void**)&dev_n1, sizeof(int) * localCols);
```

---

## 📝 完整实现步骤

### 步骤1: 复制所有kernel函数

将 `DtestStaticSPAINew30_fixed.cpp` 中的所有kernel函数复制到多DCU版本：

```cpp
// 需要复制的kernel（完整列表）:
- cuComputeN2MAXwithSparityofA
- cuComputeN1MAX
- cuComputeN1withSparityofA_SpMMv1
- cuComputeN1withSparityofA2_SpMMv1
- computeJ_SSPAIv10
- computeJ2_SSPAIv10
- computeI_iter_Symbol_SpMMv1
- computeI_SSPAIv10
- computeIShared_Symbol_SpMMv1
- computeIShared_iter_Symbol_SpMMv1
- computeI_Symbol_SpMMv1
- OE_ParallelSort
- cuAssembleM
- cuAssembleM2
- cuAssembleM_iter
// ... 等所有kernel
```

### 步骤2: 修改StaticSPAIv20函数

创建支持列范围的版本：

```cpp
float StaticSPAIv20_ColumnRange(CSC_Matrix *devA, CSC_Matrix *devM,
                                 int colStart, int colEnd, int deviceId) {
    hipSetDevice(deviceId);  // 设置当前设备

    int localCols = colEnd - colStart;
    printf("DCU %d: 处理列 [%d, %d), 共 %d 列\n",
           deviceId, colStart, colEnd, localCols);

    // 计算n2max（只针对本地列）
    const int threadsPerBlock = 256;
    int blocksPerGrid = 15 * 8 * 4;

    int *dev_n2;
    hipMalloc((void**)&dev_n2, sizeof(int) * blocksPerGrid);

    // 注意：这里传入的是局部列数和偏移的指针
    cuComputeN2MAXwithSparityofA<256><<<blocksPerGrid, threadsPerBlock>>>(
        devA->mPtr + colStart,  // 从colStart开始的列指针
        localCols,              // 局部列数
        dev_n2
    );

    // ... 后续计算类似修改 ...

    // 重要：所有涉及nCol的地方都要替换为localCols
    // 所有涉及列索引的地方都要加上colStart偏移

    return elapsedTime;
}
```

### 步骤3: 实现聚合函数

将各DCU的结果合并成全局矩阵：

```cpp
void aggregateResults(MultiDCU_Context *ctx, CSC_Matrix *devCSC_M_global) {
    // 1. 计算总非零元和列指针
    int totalCols = ctx->devCSC_A[0]->nCol;
    int *globalPtr = (int*)malloc(sizeof(int) * (totalCols + 1));
    globalPtr[0] = 0;

    // 2. 从各DCU收集列指针信息
    for (int dcuId = 0; dcuId < ctx->numDCUs; dcuId++) {
        hipSetDevice(dcuId);
        CSC_Matrix *localM = ctx->devCSC_M_local[dcuId];
        int localCols = ctx->colEnd[dcuId] - ctx->colStart[dcuId];

        // 复制局部mPtr到主机
        int *localPtr_host = (int*)malloc(sizeof(int) * (localCols + 1));
        hipMemcpy(localPtr_host, localM->mPtr,
                  (localCols + 1) * sizeof(int),
                  hipMemcpyDeviceToHost);

        // 合并到全局mPtr
        int offset = globalPtr[ctx->colStart[dcuId]];
        for (int i = 0; i <= localCols; i++) {
            globalPtr[ctx->colStart[dcuId] + i] = localPtr_host[i] + offset;
        }

        free(localPtr_host);
    }

    // 3. 分配全局矩阵内存
    int totalNonzeros = globalPtr[totalCols];
    devCSC_M_global->nonzeroes = totalNonzeros;
    devCSC_M_global->nCol = totalCols;

    hipSetDevice(0);  // 在第一个DCU上创建全局矩阵
    hipMalloc((void**)&devCSC_M_global->mPtr, sizeof(int) * (totalCols + 1));
    hipMalloc((void**)&devCSC_M_global->mIndex, sizeof(int) * totalNonzeros);
    hipMalloc((void**)&devCSC_M_global->mData, sizeof(double) * totalNonzeros);

    // 4. 复制列指针
    hipMemcpy(devCSC_M_global->mPtr, globalPtr,
              (totalCols + 1) * sizeof(int),
              hipMemcpyHostToDevice);

    // 5. 从各DCU复制数据和索引
    for (int dcuId = 0; dcuId < ctx->numDCUs; dcuId++) {
        hipSetDevice(dcuId);
        CSC_Matrix *localM = ctx->devCSC_M_local[dcuId];

        int startNnz = globalPtr[ctx->colStart[dcuId]];
        int numNnz = localM->nonzeroes;

        // 使用peer-to-peer拷贝（如果支持）或通过主机中转
        if (dcuId == 0) {
            // 直接在同一设备上复制
            hipMemcpy(devCSC_M_global->mIndex + startNnz, localM->mIndex,
                      numNnz * sizeof(int), hipMemcpyDeviceToDevice);
            hipMemcpy(devCSC_M_global->mData + startNnz, localM->mData,
                      numNnz * sizeof(double), hipMemcpyDeviceToDevice);
        } else {
            // 通过主机中转
            int *tempIndex = (int*)malloc(numNnz * sizeof(int));
            double *tempData = (double*)malloc(numNnz * sizeof(double));

            hipMemcpy(tempIndex, localM->mIndex, numNnz * sizeof(int),
                      hipMemcpyDeviceToHost);
            hipMemcpy(tempData, localM->mData, numNnz * sizeof(double),
                      hipMemcpyDeviceToHost);

            hipSetDevice(0);
            hipMemcpy(devCSC_M_global->mIndex + startNnz, tempIndex,
                      numNnz * sizeof(int), hipMemcpyHostToDevice);
            hipMemcpy(devCSC_M_global->mData + startNnz, tempData,
                      numNnz * sizeof(double), hipMemcpyHostToDevice);

            free(tempIndex);
            free(tempData);
        }
    }

    free(globalPtr);
}
```

### 步骤4: 启用P2P内存访问（可选优化）

```cpp
void enableP2P(MultiDCU_Context *ctx) {
    for (int i = 0; i < ctx->numDCUs; i++) {
        for (int j = 0; j < ctx->numDCUs; j++) {
            if (i != j) {
                hipSetDevice(i);
                int canAccessPeer;
                hipDeviceCanAccessPeer(&canAccessPeer, i, j);
                if (canAccessPeer) {
                    hipDeviceEnablePeerAccess(j, 0);
                    printf("启用 DCU %d -> DCU %d 的P2P访问\n", i, j);
                }
            }
        }
    }
}
```

---

## 🚀 性能优化建议

### 1. 负载均衡

考虑列的计算复杂度不同：

```cpp
// 简单平均分配
void distributeColumns_Simple(MultiDCU_Context *ctx, int totalCols) {
    int colsPerDCU = totalCols / ctx->numDCUs;
    for (int i = 0; i < ctx->numDCUs; i++) {
        ctx->colStart[i] = i * colsPerDCU;
        ctx->colEnd[i] = (i + 1) * colsPerDCU;
    }
    ctx->colEnd[ctx->numDCUs - 1] = totalCols;  // 最后一个DCU处理剩余列
}

// 根据非零元数量加权分配
void distributeColumns_Weighted(MultiDCU_Context *ctx, CSC_Matrix *A) {
    // 计算每列的非零元数量
    int *colNnz = (int*)malloc(sizeof(int) * A->nCol);
    for (int i = 0; i < A->nCol; i++) {
        colNnz[i] = A->mPtr[i + 1] - A->mPtr[i];
    }

    // 根据非零元总数平均分配
    int totalNnz = A->nonzeroes;
    int nnzPerDCU = totalNnz / ctx->numDCUs;

    int currentDCU = 0;
    int currentNnz = 0;
    ctx->colStart[0] = 0;

    for (int col = 0; col < A->nCol; col++) {
        currentNnz += colNnz[col];
        if (currentNnz >= nnzPerDCU && currentDCU < ctx->numDCUs - 1) {
            ctx->colEnd[currentDCU] = col + 1;
            currentDCU++;
            ctx->colStart[currentDCU] = col + 1;
            currentNnz = 0;
        }
    }
    ctx->colEnd[ctx->numDCUs - 1] = A->nCol;

    free(colNnz);
}
```

### 2. 重叠计算和通信

```cpp
// 在一个DCU计算的同时，另一个DCU可以进行内存传输
for (int i = 0; i < ctx->numDCUs; i++) {
    hipSetDevice(i);
    // 使用不同的流实现计算和传输重叠
    hipMemcpyAsync(..., ctx->streams[i]);
    kernelLaunch<<<..., ctx->streams[i]>>>(...);
}
```

### 3. 减少内存拷贝

```cpp
// 尽量使用P2P直接访问，避免通过主机中转
if (p2pEnabled) {
    // 直接从DCU i 访问 DCU j 的内存
    hipMemcpyPeer(dst, dstDevice, src, srcDevice, size);
} else {
    // 必须通过主机中转
    hipMemcpy(host, src, size, hipMemcpyDeviceToHost);
    hipMemcpy(dst, host, size, hipMemcpyHostToDevice);
}
```

---

## 🧪 测试验证

### 编译命令

```bash
hipcc -O3 -o DtestStaticSPAINew30_multiDCU \
    DtestStaticSPAINew30_multiDCU.cpp \
    -I./common \
    --offload-arch=gfx906 \
    -D__HIP_PLATFORM_AMD__ \
    -lhipblas -lhipsparse -lm -fopenmp
```

### 运行测试

```bash
# 使用所有可用DCU
./DtestStaticSPAINew30_multiDCU
# 输入: circuit_2.mtx

# 指定DCU数量
./DtestStaticSPAINew30_multiDCU 2  # 使用2个DCU
./DtestStaticSPAINew30_multiDCU 4  # 使用4个DCU
```

### 验证正确性

```bash
# 1. 运行单DCU版本保存结果
echo "circuit_2.mtx" | ./DtestStaticSPAINew30_fixed > single_dcu_output.txt

# 2. 运行多DCU版本保存结果
echo "circuit_2.mtx" | ./DtestStaticSPAINew30_multiDCU 2 > multi_dcu_output.txt

# 3. 对比计算结果
# 检查预条件子矩阵M的非零元是否一致
# 检查BiCGSTAB迭代次数和残差是否相同
```

### 性能测试

```bash
# 测试不同DCU数量的性能
for n in 1 2 4 8; do
    echo "=== 使用 $n 个DCU ==="
    echo "circuit_2.mtx" | ./DtestStaticSPAINew30_multiDCU $n | grep "Time"
done
```

---

## 📊 预期性能提升

| DCU数量 | 理论加速比 | 实际加速比估计 | 说明 |
|---------|-----------|--------------|------|
| 1 | 1.0x | 1.0x | 基准 |
| 2 | 2.0x | 1.7-1.9x | 通信和同步开销 |
| 4 | 4.0x | 3.2-3.6x | 更多同步开销 |
| 8 | 8.0x | 5.5-7.0x | 负载不均和通信 |

实际加速比取决于：
- 矩阵规模（越大越好）
- 矩阵稀疏模式
- DCU间互连带宽
- P2P是否可用

---

## 🐛 常见问题

### 1. 编译错误：未定义的引用

**问题**：链接时提示kernel函数未定义

**解决**：确保所有kernel函数都在多DCU源文件中定义，或通过头文件包含

### 2. 运行时错误：设备同步失败

**问题**：`hipDeviceSynchronize()` 失败

**解决**：
```cpp
// 每个线程设置正确的设备
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    hipSetDevice(tid);
    // ... 后续操作
}
```

### 3. 计算结果不一致

**问题**：多DCU结果与单DCU不同

**解决**：
- 检查列范围偏移是否正确
- 验证数据拼接逻辑
- 确保所有`__syncthreads()`都存在

### 4. 性能反而下降

**问题**：使用多DCU比单DCU还慢

**可能原因**：
- 矩阵太小，通信开销大于计算收益
- 负载不均衡
- P2P未启用，通过主机中转效率低

---

## 📚 参考资料

- [HIP多GPU编程指南](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html)
- [OpenMP并行编程](https://www.openmp.org/)
- [稀疏矩阵并行计算](https://arxiv.org/abs/1907.05279)

---

## ✅ 实现检查清单

- [ ] 复制所有kernel函数
- [ ] 修改StaticSPAIv20支持列范围
- [ ] 实现多DCU初始化
- [ ] 实现列分配策略
- [ ] 实现矩阵A复制
- [ ] 实现OpenMP并行计算
- [ ] 实现结果聚合
- [ ] 启用P2P访问（可选）
- [ ] 编译测试
- [ ] 正确性验证
- [ ] 性能测试
- [ ] 文档更新

---

**版本**: 1.0
**更新日期**: 2026-02-05
**作者**: Claude Code Assistant
