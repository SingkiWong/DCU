# 文件整理和重命名方案

## 📊 当前文件分析

### 文件总数统计
- 源代码文件: 7个
- 头文件: 17个 (common/)
- 编译配置: 4个
- 文档文件: 6个
- 测试数据: 2个
- 编译产物: 1个
- 其他: 1个

---

## 🗂️ 文件分类

### ✅ 核心代码（保留）

#### 主程序代码
| 当前文件名 | 建议文件名 | 说明 | 操作 |
|-----------|----------|------|------|
| `DtestStaticSPAINew30_fixed.cpp` | `spai_single_dcu.cpp` | 单DCU主程序 | 重命名 |
| `DtestStaticSPAINew30_multiDCU.cpp` | `spai_multi_dcu.cpp` | 多DCU主程序 | 重命名 |

#### 求解器
| 当前文件名 | 建议文件名 | 说明 | 操作 |
|-----------|----------|------|------|
| `bicgstab/cublas2_csr_pbicgstab_12.2.h` | `bicgstab/bicgstab_solver.h` | BiCGSTAB求解器 | 重命名 |

#### 公共头文件 (common/)
| 文件 | 状态 | 说明 |
|------|------|------|
| `dataType.h` | ✅ 保留 | 数据类型定义 |
| `read.h` | ✅ 保留 | 矩阵读取 |
| `init.h` | ✅ 保留 | 初始化 |
| `assemble.h` | ✅ 保留 | 矩阵组装 |
| `cuFormatConversion.h` | ✅ 保留 | 格式转换 |
| 其他.h文件 | ✅ 保留 | 工具函数 |

---

### 📝 文档文件（保留，重命名）

| 当前文件名 | 建议文件名 | 说明 | 操作 |
|-----------|----------|------|------|
| `README_DCU.md` | `README.md` | 主文档 | 重命名 |
| `DtestStaticSPAINew30_fixed_CHANGELOG.md` | `CHANGELOG_SINGLE_DCU.md` | 单DCU修复日志 | 重命名 |
| `MULTI_DCU_GUIDE.md` | `GUIDE_MULTI_DCU.md` | 多DCU指南 | 重命名 |
| `MULTI_DCU_SUMMARY.md` | `SUMMARY_MULTI_DCU.md` | 多DCU总结 | 重命名 |
| `MULTI_DCU_QUICKREF.md` | `QUICKREF_MULTI_DCU.md` | 快速参考 | 重命名 |
| `CODE_ISSUES_AND_FIXES.md` | ✅ 保留原名 | 代码问题和修复 | 保留 |

---

### ⚙️ 编译配置（保留，重命名）

| 当前文件名 | 建议文件名 | 说明 | 操作 |
|-----------|----------|------|------|
| `Makefile.dcu` | `Makefile.single` | 单DCU编译配置 | 重命名 |
| `Makefile.multiDCU` | `Makefile.multi` | 多DCU编译配置 | 重命名 |
| `build_dcu.sh` | `build_single.sh` | 单DCU编译脚本 | 重命名 |

---

### 📦 测试数据（保留）

| 文件 | 状态 | 说明 |
|------|------|------|
| `circuit_2.mtx` | ✅ 保留 | 测试矩阵1 |
| `fill_3elt.mtx` | ✅ 保留 | 测试矩阵2 |

---

### ❌ 待删除文件

#### 编译产物
| 文件 | 大小 | 原因 | 操作 |
|------|------|------|------|
| `DtestStaticSPAINew30` | 1.7MB | 编译产物（ELF可执行文件） | **删除** |

#### 旧版本/测试代码
| 文件 | 大小 | 原因 | 操作 |
|------|------|------|------|
| `DtestStaticSPAINew30.cpp` | 125KB | 旧版HIP代码（有bug） | **移至archive/** |
| `DtestStaticSPAINew30.cu` | 126KB | 原始CUDA代码 | **移至archive/** |
| `duospai.cpp` | 51KB | 旧的多DCU实现 | **移至archive/** |
| `chaxun.cpp` | 1.1KB | DCU查询小工具 | **移至tools/** |
| `demo.cpp` | 2.1KB | 向量加法演示 | **移至examples/** |

#### 无用Makefile
| 文件 | 原因 | 操作 |
|------|------|------|
| `common/Makefile` | CUDA版本Makefile | **删除** |
| `common/Makefile1070` | 旧版本Makefile | **删除** |

---

## 🏗️ 新目录结构

```
C:\Users\HP\Desktop\dc\
│
├── README.md                          # 主文档
├── CODE_ISSUES_AND_FIXES.md          # 问题和修复
│
├── src/                               # 源代码目录（新建）
│   ├── spai_single_dcu.cpp           # 单DCU主程序
│   └── spai_multi_dcu.cpp            # 多DCU主程序
│
├── include/                           # 头文件目录（新建）
│   ├── bicgstab/
│   │   └── bicgstab_solver.h         # BiCGSTAB求解器
│   └── common/
│       ├── dataType.h
│       ├── read.h
│       ├── init.h
│       ├── assemble.h
│       ├── cuFormatConversion.h
│       └── ... (其他头文件)
│
├── build/                             # 编译配置（新建）
│   ├── Makefile.single               # 单DCU编译
│   ├── Makefile.multi                # 多DCU编译
│   └── build_single.sh               # 编译脚本
│
├── data/                              # 测试数据（新建）
│   ├── circuit_2.mtx
│   └── fill_3elt.mtx
│
├── docs/                              # 文档目录（新建）
│   ├── CHANGELOG_SINGLE_DCU.md
│   ├── GUIDE_MULTI_DCU.md
│   ├── SUMMARY_MULTI_DCU.md
│   └── QUICKREF_MULTI_DCU.md
│
├── tools/                             # 工具脚本（新建）
│   └── check_dcu.cpp                 # DCU查询工具
│
├── examples/                          # 示例代码（新建）
│   └── vector_add_demo.cpp           # 向量加法示例
│
└── archive/                           # 归档旧代码（新建）
    ├── DtestStaticSPAINew30.cpp      # 旧版HIP代码
    ├── DtestStaticSPAINew30.cu       # 原始CUDA代码
    └── duospai.cpp                   # 旧的多DCU实现
```

---

## 🚀 执行脚本

### 方案A: 保持扁平结构（简单重命名）

```bash
#!/bin/bash
# rename_simple.sh - 简单重命名，保持扁平结构

cd /c/Users/HP/Desktop/dc

echo "=== 开始文件重命名 ==="

# 1. 主程序重命名
mv DtestStaticSPAINew30_fixed.cpp spai_single_dcu.cpp
mv DtestStaticSPAINew30_multiDCU.cpp spai_multi_dcu.cpp
echo "✓ 主程序重命名完成"

# 2. 文档重命名
mv README_DCU.md README.md
mv DtestStaticSPAINew30_fixed_CHANGELOG.md CHANGELOG_SINGLE_DCU.md
mv MULTI_DCU_GUIDE.md GUIDE_MULTI_DCU.md
mv MULTI_DCU_SUMMARY.md SUMMARY_MULTI_DCU.md
mv MULTI_DCU_QUICKREF.md QUICKREF_MULTI_DCU.md
echo "✓ 文档重命名完成"

# 3. 编译配置重命名
mv Makefile.dcu Makefile.single
mv Makefile.multiDCU Makefile.multi
mv build_dcu.sh build_single.sh
chmod +x build_single.sh
echo "✓ 编译配置重命名完成"

# 4. BiCGSTAB求解器重命名
mv bicgstab/cublas2_csr_pbicgstab_12.2.h bicgstab/bicgstab_solver.h
echo "✓ 求解器重命名完成"

# 5. 创建归档目录并移动旧文件
mkdir -p archive
mv DtestStaticSPAINew30.cpp archive/
mv DtestStaticSPAINew30.cu archive/
mv duospai.cpp archive/
echo "✓ 旧代码已归档"

# 6. 创建工具目录
mkdir -p tools
mv chaxun.cpp tools/check_dcu.cpp
echo "✓ 工具已整理"

# 7. 创建示例目录
mkdir -p examples
mv demo.cpp examples/vector_add_demo.cpp
echo "✓ 示例已整理"

# 8. 删除编译产物
rm -f DtestStaticSPAINew30
echo "✓ 编译产物已删除"

# 9. 删除无用Makefile
rm -f common/Makefile common/Makefile1070
echo "✓ 无用Makefile已删除"

echo ""
echo "=== 文件整理完成 ==="
echo ""
echo "新的主文件："
echo "  - spai_single_dcu.cpp     (单DCU主程序)"
echo "  - spai_multi_dcu.cpp      (多DCU主程序)"
echo "  - README.md               (主文档)"
echo ""
echo "归档文件位置: archive/"
echo "工具位置: tools/"
echo "示例位置: examples/"
```

### 方案B: 创建标准目录结构（推荐）

```bash
#!/bin/bash
# reorganize_full.sh - 完整重组织，创建标准目录结构

cd /c/Users/HP/Desktop/dc

echo "=== 开始文件重组织 ==="

# 1. 创建目录结构
mkdir -p src include/bicgstab include/common build data docs tools examples archive
echo "✓ 目录结构创建完成"

# 2. 移动源代码
mv DtestStaticSPAINew30_fixed.cpp src/spai_single_dcu.cpp
mv DtestStaticSPAINew30_multiDCU.cpp src/spai_multi_dcu.cpp
echo "✓ 源代码已移动"

# 3. 移动头文件
mv bicgstab/cublas2_csr_pbicgstab_12.2.h include/bicgstab/bicgstab_solver.h
mv common/*.h include/common/
rmdir bicgstab common 2>/dev/null
echo "✓ 头文件已移动"

# 4. 移动编译配置
mv Makefile.dcu build/Makefile.single
mv Makefile.multiDCU build/Makefile.multi
mv build_dcu.sh build/build_single.sh
chmod +x build/build_single.sh
echo "✓ 编译配置已移动"

# 5. 移动测试数据
mv circuit_2.mtx data/
mv fill_3elt.mtx data/
echo "✓ 测试数据已移动"

# 6. 移动文档
mv README_DCU.md docs/README.md
mv DtestStaticSPAINew30_fixed_CHANGELOG.md docs/CHANGELOG_SINGLE_DCU.md
mv MULTI_DCU_GUIDE.md docs/GUIDE_MULTI_DCU.md
mv MULTI_DCU_SUMMARY.md docs/SUMMARY_MULTI_DCU.md
mv MULTI_DCU_QUICKREF.md docs/QUICKREF_MULTI_DCU.md
# 主README放在根目录
cp docs/README.md README.md
echo "✓ 文档已移动"

# 7. 归档旧代码
mv DtestStaticSPAINew30.cpp archive/
mv DtestStaticSPAINew30.cu archive/
mv duospai.cpp archive/
echo "✓ 旧代码已归档"

# 8. 整理工具和示例
mv chaxun.cpp tools/check_dcu.cpp
mv demo.cpp examples/vector_add_demo.cpp
echo "✓ 工具和示例已整理"

# 9. 删除编译产物和无用文件
rm -f DtestStaticSPAINew30
rm -f include/common/Makefile include/common/Makefile1070
echo "✓ 无用文件已删除"

echo ""
echo "=== 文件重组织完成 ==="
echo ""
echo "新目录结构："
tree -L 2 -I '.claude'
```

---

## 📋 执行建议

### 推荐步骤

1. **备份当前目录**
   ```bash
   cd /c/Users/HP/Desktop
   cp -r dc dc_backup_$(date +%Y%m%d)
   ```

2. **选择方案**
   - 方案A: 适合快速重命名，保持简单
   - 方案B: 适合长期维护，结构清晰（推荐）

3. **执行整理**
   ```bash
   cd /c/Users/HP/Desktop/dc
   chmod +x rename_simple.sh    # 或 reorganize_full.sh
   ./rename_simple.sh           # 或 ./reorganize_full.sh
   ```

4. **更新代码中的路径**
   如果选择方案B，需要更新：
   - Makefile中的include路径
   - 源代码中的#include路径

---

## ✅ 整理后的优势

### 方案A优势
- ✅ 简单快速
- ✅ 不破坏现有include路径
- ✅ 文件名更清晰
- ✅ 移除无用文件

### 方案B优势
- ✅ 标准项目结构
- ✅ 便于版本控制
- ✅ 清晰的文件分类
- ✅ 易于扩展维护

---

## 🔍 文件对比表

### 重命名对照

| 旧文件名 | 新文件名 | 变化 |
|---------|---------|------|
| `DtestStaticSPAINew30_fixed.cpp` | `spai_single_dcu.cpp` | 更简洁 |
| `DtestStaticSPAINew30_multiDCU.cpp` | `spai_multi_dcu.cpp` | 更简洁 |
| `README_DCU.md` | `README.md` | 标准名称 |
| `Makefile.dcu` | `Makefile.single` | 更清晰 |
| `build_dcu.sh` | `build_single.sh` | 更清晰 |
| `bicgstab/cublas2_csr_pbicgstab_12.2.h` | `bicgstab/bicgstab_solver.h` | 更简洁 |

### 文件归档/删除

| 文件 | 操作 | 原因 |
|------|------|------|
| `DtestStaticSPAINew30` | 删除 | 编译产物 |
| `DtestStaticSPAINew30.cpp` | 归档 | 旧版本 |
| `DtestStaticSPAINew30.cu` | 归档 | CUDA版本 |
| `duospai.cpp` | 归档 | 旧实现 |
| `chaxun.cpp` | 移至tools/ | 工具 |
| `demo.cpp` | 移至examples/ | 示例 |
| `common/Makefile*` | 删除 | 无用 |

---

**建议**: 使用**方案A（简单重命名）**开始，文件名更清晰且不影响现有代码。
