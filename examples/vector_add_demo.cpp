#include <iostream>
#include <hip/hip_runtime.h>

__global__ void vecAdd(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

int main() {
    int N = 1 << 20;  // 100万数据
    size_t size = N * sizeof(float);

    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    std::cout << "检测到 " << deviceCount << " 张卡" << std::endl;

    // --- 假设有2张卡，拆分数据 ---
    int halfN = N / 2;

    // === 卡0处理前一半 ===
    hipSetDevice(0);
    float *dA0, *dB0, *dC0;
    hipMalloc(&dA0, size/2);
    hipMalloc(&dB0, size/2);
    hipMalloc(&dC0, size/2);

    // host 数据
    float *hA = new float[N];
    float *hB = new float[N];
    float *hC = new float[N];
    for (int i=0; i<N; i++) { hA[i]=i; hB[i]=2*i; }

    hipMemcpy(dA0, hA, size/2, hipMemcpyHostToDevice);
    hipMemcpy(dB0, hB, size/2, hipMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (halfN + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(vecAdd, dim3(gridSize), dim3(blockSize), 0, 0,
                       dA0, dB0, dC0, halfN);

    // === 卡1处理后一半 ===
    hipSetDevice(1);
    float *dA1, *dB1, *dC1;
    hipMalloc(&dA1, size/2);
    hipMalloc(&dB1, size/2);
    hipMalloc(&dC1, size/2);

    hipMemcpy(dA1, hA+halfN, size/2, hipMemcpyHostToDevice);
    hipMemcpy(dB1, hB+halfN, size/2, hipMemcpyHostToDevice);

    gridSize = (halfN + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(vecAdd, dim3(gridSize), dim3(blockSize), 0, 0,
                       dA1, dB1, dC1, halfN);

    // === 把结果拷回主机并拼起来 ===
    hipMemcpy(hC, dC0, size/2, hipMemcpyDeviceToHost);
    hipMemcpy(hC+halfN, dC1, size/2, hipMemcpyDeviceToHost);

    std::cout << "C[100] = " << hC[100]
              << ", C[N-1] = " << hC[N-1] << std::endl;

    // === 清理 ===
    delete [] hA; delete [] hB; delete [] hC;
    hipFree(dA0); hipFree(dB0); hipFree(dC0);
    hipFree(dA1); hipFree(dB1); hipFree(dC1);
    return 0;
}
