#include <iostream>
#include <hip/hip_runtime.h>   // 曙光 DCU SDK 如果不支持 HIP，请换成 <dcu_runtime.h>

int main() {
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err != hipSuccess) {
        std::cerr << "hipGetDeviceCount failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "本机可见 DCU 数量: " << deviceCount << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, dev);

        std::cout << "-------------------------------" << std::endl;
        std::cout << "设备 " << dev << " 名称: " << prop.name << std::endl;
        std::cout << "  计算单元数 (CUs): " << prop.multiProcessorCount << std::endl;
        std::cout << "  全局显存大小 (bytes): " << prop.totalGlobalMem << std::endl;
        std::cout << "  时钟频率 (kHz): " << prop.clockRate << std::endl;
        std::cout << "  设备ID: " << prop.pciBusID << ":" << prop.pciDeviceID << std::endl;
    }

    return 0;
}
