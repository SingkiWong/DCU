// ============================================================================
// StaticSPAIv20_ColumnRange 函数 - 简化示例
// 对比单DCU版本和多DCU版本的修改
// ============================================================================

#include "hip/hip_runtime.h"

// ============================================================================
// 单DCU版本 (原始)
// ============================================================================
float StaticSPAIv20_SingleDCU(CSC_Matrix *devA, CSC_Matrix *devM) {
    printf("处理所有列: [0, %d)\n", devA->nCol);

    const int threadsPerBlock = 256;
    int blocksPerGrid = 15 * 8 * 4;

    // --- 步骤1: 计算 n2max ---
    int *dev_n2;
    hipMalloc((void**)&dev_n2, sizeof(int) * blocksPerGrid);

    // ❌ 处理所有列
    cuComputeN2MAXwithSparityofA<256><<<blocksPerGrid, threadsPerBlock>>>(
        devA->mPtr,          // 从第0列开始
        devA->nCol,          // 处理所有列 (例如 10000 列)
        dev_n2
    );

    // 获取结果...
    int n2max = /* 从 dev_n2 获取最大值 */;

    // --- 步骤2: 计算 n1max ---
    int WarpSize = (n2max <= 2) ? 2 : (n2max <= 4) ? 4 : 32;

    // ❌ 使用总列数计算
    blocksPerGrid = (devA->nCol - 1) / (threadsPerBlock / WarpSize) + 1;

    // ❌ 为所有列分配内存
    int *dev_n1;
    hipMalloc((void**)&dev_n1, sizeof(int) * devA->nCol);

    int *dev_tI;
    int ssize = n2max * 8;
    // ❌ 使用总列数
    hipMalloc((void**)&dev_tI, sizeof(int) * devA->nCol * ssize);

    int *dev_atomic;
    // ❌ 使用总列数
    hipMalloc((void**)&dev_atomic, sizeof(int) * devA->nCol);

    // ❌ kernel调用使用全部列
    cuComputeN1withSparityofA_SpMMv1<2, 2048><<<blocksPerGrid, threadsPerBlock>>>(
        devA->mPtr,          // 从第0列开始
        devA->mIndex,
        devA->mPtr,          // 从第0列开始
        devA->mIndex,
        dev_tI,
        ssize,
        devA->nCol,          // 处理所有列
        dev_n1,
        dev_atomic
    );

    // 获取 n1max...

    // --- 步骤3: 分配工作内存 ---
    int *dev_J, *dev_jPTR;
    // ❌ 使用总列数
    hipMalloc((void**)&dev_J, sizeof(int) * devA->nCol * n2max);
    hipMalloc((void**)&dev_jPTR, sizeof(int) * devA->nCol);

    // --- 步骤4: 计算 J ---
    computeJ_SSPAIv10<2><<<blocksPerGrid, threadsPerBlock>>>(
        devA->mPtr,          // 从第0列开始
        devA->mIndex,
        devA->nCol,          // 处理所有列
        dev_J, dev_jPTR, n2max
    );

    // 后续步骤...

    return 0.0f;
}

// ============================================================================
// 多DCU版本 (修改后) - 每个DCU只处理部分列
// ============================================================================
float StaticSPAIv20_ColumnRange(CSC_Matrix *devA, CSC_Matrix *devM,
                                 int colStart, int colEnd, int deviceId) {
    // ✅ 计算局部列数
    int localCols = colEnd - colStart;

    printf("DCU %d 处理列: [%d, %d), 共 %d 列\n",
           deviceId, colStart, colEnd, localCols);

    const int threadsPerBlock = 256;
    int blocksPerGrid = 15 * 8 * 4;

    // --- 步骤1: 计算 n2max ---
    int *dev_n2;
    hipMalloc((void**)&dev_n2, sizeof(int) * blocksPerGrid);

    // ✅ 只处理局部列
    cuComputeN2MAXwithSparityofA<256><<<blocksPerGrid, threadsPerBlock>>>(
        devA->mPtr + colStart,   // ✅ 从 colStart 列开始
        localCols,               // ✅ 只处理 localCols 列
        dev_n2
    );

    // 获取结果...
    int n2max = /* 从 dev_n2 获取最大值 */;

    // --- 步骤2: 计算 n1max ---
    int WarpSize = (n2max <= 2) ? 2 : (n2max <= 4) ? 4 : 32;

    // ✅ 使用局部列数计算
    blocksPerGrid = (localCols - 1) / (threadsPerBlock / WarpSize) + 1;

    // ✅ 只为局部列分配内存
    int *dev_n1;
    hipMalloc((void**)&dev_n1, sizeof(int) * localCols);

    int *dev_tI;
    int ssize = n2max * 8;
    // ✅ 使用局部列数
    hipMalloc((void**)&dev_tI, sizeof(int) * localCols * ssize);

    int *dev_atomic;
    // ✅ 使用局部列数
    hipMalloc((void**)&dev_atomic, sizeof(int) * localCols);

    // ✅ kernel调用使用局部列
    cuComputeN1withSparityofA_SpMMv1<2, 2048><<<blocksPerGrid, threadsPerBlock>>>(
        devA->mPtr + colStart,   // ✅ 从 colStart 列开始
        devA->mIndex,
        devA->mPtr + colStart,   // ✅ 同样偏移
        devA->mIndex,
        dev_tI,
        ssize,
        localCols,               // ✅ 只处理局部列
        dev_n1,
        dev_atomic
    );

    // 获取 n1max...

    // --- 步骤3: 分配工作内存 ---
    int *dev_J, *dev_jPTR;
    // ✅ 使用局部列数
    hipMalloc((void**)&dev_J, sizeof(int) * localCols * n2max);
    hipMalloc((void**)&dev_jPTR, sizeof(int) * localCols);

    // --- 步骤4: 计算 J ---
    computeJ_SSPAIv10<2><<<blocksPerGrid, threadsPerBlock>>>(
        devA->mPtr + colStart,   // ✅ 从 colStart 列开始
        devA->mIndex,
        localCols,               // ✅ 只处理局部列
        dev_J, dev_jPTR, n2max
    );

    // 后续步骤...

    return 0.0f;
}

// ============================================================================
// 实际使用示例
// ============================================================================
int main() {
    // 假设矩阵有 10000 列
    CSC_Matrix *devA = /* 已分配的矩阵 */;

    // === 单DCU版本 ===
    CSC_Matrix *devM_single = /* 分配 */;
    StaticSPAIv20_SingleDCU(devA, devM_single);
    // 处理所有 10000 列

    // === 多DCU版本 (4个DCU) ===
    // DCU 0
    CSC_Matrix *devM_0 = /* 分配 */;
    StaticSPAIv20_ColumnRange(devA, devM_0, 0, 2500, 0);
    // DCU 0 处理列 [0, 2500)

    // DCU 1
    CSC_Matrix *devM_1 = /* 分配 */;
    StaticSPAIv20_ColumnRange(devA, devM_1, 2500, 5000, 1);
    // DCU 1 处理列 [2500, 5000)

    // DCU 2
    CSC_Matrix *devM_2 = /* 分配 */;
    StaticSPAIv20_ColumnRange(devA, devM_2, 5000, 7500, 2);
    // DCU 2 处理列 [5000, 7500)

    // DCU 3
    CSC_Matrix *devM_3 = /* 分配 */;
    StaticSPAIv20_ColumnRange(devA, devM_3, 7500, 10000, 3);
    // DCU 3 处理列 [7500, 10000)

    // 最后聚合 devM_0, devM_1, devM_2, devM_3 成一个完整的矩阵

    return 0;
}

// ============================================================================
// 关键修改总结
// ============================================================================
/*
修改项                     单DCU版本           多DCU版本
-----------------------------------------------------------------------
列数参数                   devA->nCol          localCols
列指针                     devA->mPtr          devA->mPtr + colStart
内存分配大小               devA->nCol * ...    localCols * ...
blocksPerGrid计算          devA->nCol          localCols
循环范围                   for(i=0; i<devA->nCol; i++)   for(i=0; i<localCols; i++)

不需要修改的:
- devA->mIndex (行索引数组)
- devA->mData (数据数组)
- devA->n (矩阵总维度)
- devA->nRow (行数)
*/
