# 多DCU SPAI 预条件子求解器 - MPI 使用说明

## 编译

### 单节点多DCU版本（默认）
```bash
make -f Makefile.multi
```

### 多节点MPI版本
```bash
make -f Makefile.multi mpi
```

这将生成 `spai_multi_dcu_mpi` 可执行文件。

### 编译选项
```bash
# 指定GPU架构
make -f Makefile.multi mpi GPU_ARCH=gfx908

# 启用调试模式
make -f Makefile.multi debug-mpi

# 同时指定多个选项
make -f Makefile.multi mpi GPU_ARCH=gfx90a DEBUG=1
```

## 运行

### 单节点多DCU模式
```bash
# 使用所有可用DCU
./spai_multi_dcu

# 指定使用2个DCU
./spai_multi_dcu 2
```

### 多节点MPI模式

#### 基本用法
```bash
# 2个MPI进程，每个进程使用所有可用DCU
mpirun -np 2 ./spai_multi_dcu_mpi

# 4个MPI进程，每个进程使用2个DCU
mpirun -np 4 ./spai_multi_dcu_mpi 2
```

#### 跨节点运行
```bash
# 使用hostfile指定节点
mpirun -np 4 -hostfile hostfile.txt ./spai_multi_dcu_mpi

# 指定节点列表
mpirun -np 4 -H node1,node2,node3,node4 ./spai_multi_dcu_mpi
```

#### hostfile 示例
```
# hostfile.txt
node1 slots=2
node2 slots=2
node3 slots=2
node4 slots=2
```

## 环境变量

### OpenMP 线程数（每节点的DCU数量）
```bash
export OMP_NUM_THREADS=4
```

### 指定可见的DCU设备
```bash
export HIP_VISIBLE_DEVICES=0,1,2,3
```

### MPI相关环境变量
```bash
# 设置MPI通信缓冲区大小（可选）
export OMPI_MCA_btl_tcp_eager_limit=32768
export OMPI_MCA_btl_tcp_max_send_size=131072
```

## 工作原理

### 单节点多DCU模式
- 使用 OpenMP 在一个节点内并行化，每个线程管理一个DCU
- 矩阵按列划分给各个DCU
- 每个DCU独立计算其负责的列
- 计算完成后在主DCU（DCU 0）上聚合结果

### 多节点MPI模式
- 首先在 MPI 层面将矩阵列按 rank 划分
- 每个 MPI 进程（节点）内部使用 OpenMP + 多DCU 并行
- 各节点独立计算其负责的列范围
- 计算完成后通过 MPI 将结果聚合到 rank 0

### 两层并行结构
```
MPI 层（节点间）
  └─ Rank 0: 列 [0, n/4)
       └─ OpenMP/DCU 层（节点内）
            ├─ DCU 0: 列 [0, n/16)
            ├─ DCU 1: 列 [n/16, n/8)
            ├─ DCU 2: 列 [n/8, 3n/16)
            └─ DCU 3: 列 [3n/16, n/4)
  └─ Rank 1: 列 [n/4, n/2)
       └─ ...
  └─ Rank 2: 列 [n/2, 3n/4)
       └─ ...
  └─ Rank 3: 列 [3n/4, n)
       └─ ...
```

## 性能优化建议

1. **DCU数量设置**
   - 每个节点的DCU数应该能被总列数整除
   - 避免负载不均衡

2. **MPI进程数设置**
   - MPI进程数应该等于节点数
   - 每个节点运行一个MPI进程

3. **内存管理**
   - 确保每个节点有足够的主机内存
   - 确保每个DCU有足够的显存

4. **网络优化**
   - 使用高速互联网络（InfiniBand等）
   - 调整MPI缓冲区大小

## 验证和调试

### 验证正确性
```bash
# 对比单DCU和多DCU结果
make -f Makefile.multi verify
```

### 调试模式
```bash
# 编译调试版本
make -f Makefile.multi debug-mpi

# 运行时查看详细输出
mpirun -np 2 ./spai_multi_dcu_mpi
```

### 性能分析
```bash
# 使用 rocprof 分析性能
mpirun -np 2 rocprof --stats ./spai_multi_dcu_mpi
```

## 常见问题

### Q: MPI版本和非MPI版本有什么区别？
A: MPI版本支持跨节点并行，可以利用多个计算节点的DCU资源。非MPI版本只能在单个节点内使用多个DCU。

### Q: 如何确定最优的MPI进程数和每进程DCU数？
A: 通常设置 MPI进程数 = 节点数，每进程DCU数 = 每个节点的DCU总数。然后根据实际性能测试调整。

### Q: 如果某些节点的DCU数量不同怎么办？
A: 可以为每个节点单独设置 OMP_NUM_THREADS 和 HIP_VISIBLE_DEVICES，但可能导致负载不均衡。

### Q: 结果聚合需要多少时间？
A: 聚合时间取决于矩阵规模和网络速度。对于大规模矩阵，建议使用高速互联网络。

## 示例输出

```
MPI Rank 0: 列范围 [0, 512)
MPI Rank 1: 列范围 [512, 1024)
使用 4 个OpenMP线程进行并行计算

DCU 0: 线程 0 开始计算列 [0, 128)
DCU 1: 线程 1 开始计算列 [128, 256)
...

MPI 进程 0: 开始跨进程结果聚合...
MPI Rank 1: 数据发送完成
MPI Rank 0: 全局矩阵聚合完成

程序执行完成
```
