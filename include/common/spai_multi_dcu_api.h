#ifndef SPAI_MULTI_DCU_API_H
#define SPAI_MULTI_DCU_API_H

#include "common/dataType.h"

struct MultiDCU_Context;

// 初始化多 DCU 运行上下文。requestedDCUs <= 0 时自动使用全部可用设备。
MultiDCU_Context* initMultiDCU(int requestedDCUs = -1);

// 释放多 DCU 运行上下文及关联资源。
void cleanupMultiDCU(MultiDCU_Context *ctx);

// 静态划分（按列均衡）SPAI 预条件子构建。
float RunStaticSPAI_MultiDCU_MPI(MultiDCU_Context *ctx, const CSC_Matrix *CSC_A,
                                 CSC_Matrix *devCSC_M_global,
                                 int globalColStart, int globalColEnd);

// 动态划分（按 NNZ 均衡）SPAI 预条件子构建。
float RunDynamicSPAI_MultiDCU_MPI(MultiDCU_Context *ctx, const CSC_Matrix *CSC_A,
                                  CSC_Matrix *devCSC_M_global,
                                  int globalColStart, int globalColEnd);

#endif
