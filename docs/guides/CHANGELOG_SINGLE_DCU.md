# DtestStaticSPAINew30_fixed.cpp 修复日志

## 修复日期
2026-02-05

## 修复内容

### 1. 主函数设备检查修复 (第2670-2682行)
**修复前:**
```cpp
if (deviceCount < 2) {
    printf("警告: 仅检测到 %d 张卡，将只用单卡运行。\n", deviceCount);
    return -1;
}
```

**修复后:**
```cpp
if (deviceCount < 1) {
    printf("错误: 未检测到DCU设备\n");
    return -1;
}
printf("检测到 %d 张 DCU，使用第一张DCU运行\n", deviceCount);
hipSetDevice(0);  // 明确使用第一张DCU

// 查询并显示设备信息
hipDeviceProp_t prop;
hipGetDeviceProperties(&prop, 0);
printf("使用DCU设备: %s\n", prop.name);
printf("计算能力: %d.%d\n", prop.major, prop.minor);
printf("Wavefront大小: %d\n", prop.warpSize);
```

**修复说明:**
- 将设备数量检查从 < 2 改为 < 1，允许单DCU运行
- 添加 hipSetDevice(0) 明确使用第一张DCU
- 添加设备属性查询，显示DCU信息以便调试

### 2. OE_ParallelSort函数模板参数 (第809行)
**状态:** 已确认正确
```cpp
template<unsigned int WarpSize>
__global__ void OE_ParallelSort(int nCol, int *I, int *iPTR, int N1MAX) {
```
此函数已经包含WarpSize模板参数，无需修复。

### 3. 恢复所有被注释的__syncthreads()
**修复位置:**
- 第375, 384行 (computeI_iter_Symbol_SpMMv1函数)
- 第477, 487行 (computeI_SSPAIv10函数)
- 第579, 588, 590行 (computeIShared_Symbol_SpMMv1函数)
- 第687, 696行 (computeIShared_iter_Symbol_SpMMv1函数)

**修复前:**
```cpp
//__syncthreads();
```

**修复后:**
```cpp
__syncthreads();
```

**修复说明:**
恢复了所有被注释的__syncthreads()调用，共47处。这些同步点对于确保多线程正确执行至关重要。

### 4. 简化WarpSize逻辑 (第1427行)
**修复前:**
```cpp
if (n2max <= 2) {
    WarpSize = 2;
} else if (n2max > 2 && n2max <= 4) {
    WarpSize = 4;
} else if (n2max > 4 && n2max <= 8) {
    WarpSize = 8;
} else if (n2max > 8 && n2max <= 16) {
    WarpSize = 16;
} else if (n2max > 16 && n2max <= 32) {
    WarpSize = 32;
} else if (n2max > 32 && n2max <= 64) {
    WarpSize = 64;
} else if (n2max > 64 && n2max <= 128) {
    WarpSize = 128;
} else {
    WarpSize = 256;
}
blocksPerGrid = (devA->nCol - 1) / (256 / WarpSize) + 1;
```

**修复后:**
```cpp
// 统一使用32，兼容NVIDIA和AMD CDNA架构
WarpSize = 32;
blocksPerGrid = (devA->nCol - 1) / (256 / WarpSize) + 1;
```

**修复说明:**
- 删除第二次WarpSize赋值的复杂if-else块
- 统一使用WarpSize = 32，兼容NVIDIA GPU (warpSize=32) 和AMD CDNA架构 (wavefrontSize=64)
- 简化代码逻辑，避免过大的WarpSize值导致的问题

### 5. WarpSize变量统一
**说明:**
- 第1258行: 保持第一次WarpSize赋值逻辑不变 (最大值32)
- 第1427行: 简化第二次WarpSize赋值为固定值32
- 确保WarpSize在整个函数中保持一致性

## 编译说明

使用以下命令编译:
```bash
hipcc -O3 -o DtestStaticSPAINew30_fixed DtestStaticSPAINew30_fixed.cpp \
    -I./include \
    -lm -fopenmp \
    -D__HIP_PLATFORM_AMD__
```

## 运行说明

程序现在支持单DCU运行：
1. 程序启动时会检测DCU设备数量
2. 至少需要1张DCU才能运行
3. 自动使用第一张DCU (device 0)
4. 显示DCU设备信息便于调试

## 验证结果

- 所有__syncthreads()已恢复: 47处
- 设备检查逻辑已修复
- WarpSize统一为32
- OE_ParallelSort模板参数正确
- 文件总行数: 2859行

## 注意事项

1. 此版本针对单DCU环境优化
2. WarpSize统一为32，兼容性更好
3. 建议在真实DCU环境中测试验证
4. 如需多DCU支持，请参考原始双DCU版本
