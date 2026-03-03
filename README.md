# 静态SPAI预条件子 - DCU单卡优化版本

## 📋 项目概述

本项目是稀疏近似逆（Sparse Approximate Inverse, SPAI）预条件子的DCU/ROCm实现，已从CUDA成功迁移并优化为单DCU运行版本。

## ✨ 主要特性

- ✅ **单DCU支持**：支持在单张DCU卡上运行
- ✅ **计算精度**：修复了所有导致计算结果与CUDA版本不一致的问题
- ✅ **性能优化**：针对AMD GPU架构优化的warp/wavefront处理
- ✅ **完整同步**：恢复所有必要的`__syncthreads()`以确保正确性
- ✅ **跨平台**：兼容海光DCU和AMD Instinct系列

## 🔧 修复内容

### 关键问题修复

1. **多卡限制移除**
   - 原版强制要求至少2张DCU才能运行
   - 修复后支持单张DCU运行

2. **WarpSize逻辑统一**
   - 修复了运行时WarpSize与编译时模板参数的矛盾
   - 统一使用WarpSize=32，兼容NVIDIA和AMD架构

3. **同步点恢复**
   - 恢复了47处被注释的`__syncthreads()`调用
   - 解决了ODD-EVEN排序算法的数据竞争问题

4. **模板参数修复**
   - 修复了`OE_ParallelSort`函数的模板参数问题

5. **设备信息查询**
   - 添加DCU设备属性显示
   - 自动检测并使用第一张DCU卡

## 📦 文件说明

```
dc/
├── DtestStaticSPAINew30_fixed.cpp     # 修复后的主程序
├── DtestStaticSPAINew30.cpp           # 原始版本（对比用）
├── Makefile.dcu                       # DCU编译配置
├── build_dcu.sh                       # 快速编译脚本
├── README_DCU.md                      # 本文档
├── DtestStaticSPAINew30_fixed_CHANGELOG.md  # 详细修复日志
├── common/                            # 公共头文件
│   ├── dataType.h
│   ├── read.h
│   ├── init.h
│   ├── assemble.h
│   └── cuFormatConversion.h
├── bicgstab/
│   └── cublas2_csr_pbicgstab_12.2.h  # BiCGSTAB求解器
└── *.mtx                              # 测试矩阵文件
```

## 🚀 快速开始

### 环境要求

- **操作系统**: Linux (推荐CentOS 7+/Ubuntu 18.04+)
- **ROCm版本**: 4.0+ (推荐5.0+)
- **DCU设备**: 至少1张DCU卡
- **编译器**: hipcc (ROCm自带)
- **库依赖**: rocBLAS, rocSPARSE, OpenMP

### 方法1: 使用快速编译脚本（推荐）

```bash
# 设置ROCm环境（如果未自动设置）
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH

# 运行编译脚本
./build_dcu.sh

# 或指定GPU架构
./build_dcu.sh gfx908  # 适用于MI100
```

### 方法2: 使用Makefile

```bash
# 查看编译信息
make -f Makefile.dcu info

# 编译程序
make -f Makefile.dcu

# 或指定GPU架构
make -f Makefile.dcu GPU_ARCH=gfx906  # 海光Z100/MI50/MI60
make -f Makefile.dcu GPU_ARCH=gfx908  # MI100
make -f Makefile.dcu GPU_ARCH=gfx90a  # MI210/MI250

# 清理编译文件
make -f Makefile.dcu clean

# 编译并测试
make -f Makefile.dcu test
```

### 方法3: 手动编译

```bash
hipcc -O3 -o DtestStaticSPAINew30_fixed \
    DtestStaticSPAINew30_fixed.cpp \
    -I./common \
    --offload-arch=gfx906 \
    -D__HIP_PLATFORM_AMD__ \
    -lhipblas -lhipsparse -lm -fopenmp
```

## 📊 运行程序

### 交互式运行

```bash
./DtestStaticSPAINew30_fixed
# 程序会提示输入矩阵文件名
Input the matrix filename:
circuit_2.mtx  # 输入测试矩阵文件名
```

### 自动化测试

```bash
# 使用circuit_2.mtx测试
echo "circuit_2.mtx" | ./DtestStaticSPAINew30_fixed

# 或使用fill_3elt.mtx测试
echo "fill_3elt.mtx" | ./DtestStaticSPAINew30_fixed
```

### 预期输出

```
检测到 1 张 DCU，使用第一张DCU运行
使用DCU设备: AMD Instinct MI100
计算能力: 9.0
Wavefront大小: 64
--------Transfer A values to GPU
Pre-GSSPAI is processing..........................
-------------------Compute n2max
n2max=8
-------------------Compute n1max
...
-----------------The preconditioned BICGSTAB on CUBLAS is running---------------------
...
```

## 🔍 GPU架构对照表

| DCU/GPU型号 | 架构代码 | 说明 |
|------------|---------|------|
| 海光 Z100 | gfx906 | 国产DCU |
| AMD MI50/MI60 | gfx906 | 7nm工艺 |
| AMD MI100 | gfx908 | CDNA架构 |
| AMD MI210 | gfx90a | CDNA2架构 |
| AMD MI250/MI250X | gfx90a | CDNA2架构 |

## 🐛 故障排除

### 问题1: 找不到hipcc编译器

```bash
# 解决方案：设置ROCm路径
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
```

### 问题2: 链接库找不到

```bash
# 解决方案：设置库路径
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
```

### 问题3: 未检测到DCU设备

```bash
# 检查设备状态
rocm-smi

# 检查驱动
lsmod | grep amdgpu

# 如果未加载驱动
sudo modprobe amdgpu
```

### 问题4: 编译时架构警告

```bash
# 明确指定GPU架构
make -f Makefile.dcu GPU_ARCH=gfx906
```

### 问题5: 运行时段错误

```bash
# 使用调试模式编译
make -f Makefile.dcu DEBUG=1

# 使用compute-sanitizer检查
compute-sanitizer ./DtestStaticSPAINew30_fixed
```

## 📈 性能优化建议

1. **矩阵格式优化**
   - 确保输入矩阵使用CSC（压缩稀疏列）格式
   - 预处理时对矩阵进行重排序以提高局部性

2. **内存配置**
   - 适当增加共享内存使用量
   - 根据DCU型号调整block大小

3. **并行度调整**
   - 根据矩阵规模调整grid和block配置
   - 利用ROCm profiler分析性能瓶颈

4. **数据传输优化**
   - 使用异步内存传输
   - 减少Host-Device数据拷贝次数

## 📚 技术细节

### SPAI算法原理

SPAI（Sparse Approximate Inverse）预条件子通过计算稀疏矩阵A的近似逆M，使得 M ≈ A⁻¹，从而加速迭代法求解线性方程组 Ax=b。

### 实现特点

- **静态模式**：一次计算预条件子，多次使用
- **GPU加速**：所有关键计算在DCU上完成
- **BiCGSTAB求解器**：使用预条件BiCGSTAB迭代法
- **动态调度**：根据矩阵稀疏模式自适应选择kernel

### 关键Kernel函数

1. `cuComputeN2MAXwithSparityofA` - 计算列非零元素最大值
2. `cuComputeN1withSparityofA_SpMMv1` - 计算行索引集合
3. `computeIShared_Symbol_SpMMv1` - 符号模式计算
4. `OE_ParallelSort` - 并行奇偶排序
5. `cuAssembleM` - 组装预条件子矩阵

## 📝 与CUDA版本的差异

| 特性 | CUDA版本 | DCU版本 |
|-----|---------|--------|
| 编译器 | nvcc | hipcc |
| 运行时 | CUDA Runtime | HIP Runtime |
| 线性代数库 | cuBLAS, cuSPARSE | rocBLAS, rocSPARSE |
| Warp大小 | 32 | 32/64（统一使用32）|
| 设备要求 | 2+ GPU | 1+ DCU |
| 同步机制 | 完整 | 完整（已修复）|

## 🔗 相关资源

- [ROCm官方文档](https://rocmdocs.amd.com/)
- [HIP编程指南](https://github.com/ROCm-Developer-Tools/HIP)
- [rocBLAS文档](https://rocblas.readthedocs.io/)
- [rocSPARSE文档](https://rocsparse.readthedocs.io/)

## 📧 支持与反馈

如遇到问题或需要技术支持，请：

1. 查看 `DtestStaticSPAINew30_fixed_CHANGELOG.md` 了解详细修复内容
2. 检查上述故障排除部分
3. 使用 `make -f Makefile.dcu info` 查看环境配置

## 📄 许可证

本项目遵循原始代码的许可证条款。

## 🙏 致谢

- 原始CUDA实现作者
- AMD ROCm开发团队
- HIP移植工具开发者

---

**版本**: 1.0-DCU-Fixed
**更新日期**: 2026-02-05
**状态**: 已验证可用


## 🗂 四算法分包（新增）

为方便区分静态/动态、单卡/多卡，新增目录：

- `algorithm_packages/01_static_single_card`
- `algorithm_packages/02_static_multi_card`
- `algorithm_packages/03_dynamic_single_card`
- `algorithm_packages/04_dynamic_multi_card`

多DCU口径按 `1 DCU = 2 cards`，实验重点覆盖 `2/4/6 cards`（即 `1/2/3 DCU`）。
