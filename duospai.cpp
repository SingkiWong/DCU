#include "hip/hip_runtime.h"
#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <memory.h>
#include <string>
#include <vector>
#include "common/dataType.h"
#include "common/read.h"
#include "common/init.h"
#include "common/assemble.h"
#include "common/cuFormatConversion.h"
#include "bicgstab/cublas2_csr_pbicgstab_12.2.h"

using namespace std;

#define CHECK_KERNEL_ERROR(stage) do { \
    hipError_t err = hipGetLastError(); \
    if (err != hipSuccess) { \
        printf("Kernel '%s' failed at iter %d: %s\n", stage, iter, hipGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

template<unsigned int N2SIZE>
__global__ void cuComputeN2MAXwithSparityofA(int *aPtr, int nCol, int *n2max) {

    __shared__ int n2_s[N2SIZE];
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x * gridDim.x;
    int tid = threadIdx.x;

    int col, col_s, col_e, tV;
    int i;

    int value = 0;
    for (col = gid; col < nCol; col += offset) {
        col_s = aPtr[col];
        col_e = aPtr[col + 1];
        tV = col_e - col_s;
        if (value < tV) value = tV;

    }//for col

    n2_s[tid] = value;

    __syncthreads();

    i = N2SIZE / 2;
    while (tid < i) {
        if (n2_s[tid] < n2_s[tid + i]) n2_s[tid] = n2_s[tid + i];
        i = i / 2;
        __syncthreads();
    }

    if (tid == 0) n2max[blockIdx.x] = n2_s[0];

}

template<unsigned int N1SIZE>
__global__ void cuComputeN1MAX(int *n1, int nCol, int *n1max) {

    __shared__ int n1_s[N1SIZE];
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x * gridDim.x;
    int tid = threadIdx.x;

    int col, value;
    int i;

    value = 0;

    for (col = gid; col < nCol; col += offset) {
        if (value < n1[col]) value = n1[col];

    }//for col

    n1_s[tid] = value;

    __syncthreads();

    i = N1SIZE / 2;
    while (tid < i) {
        if (n1_s[tid] < n1_s[tid + i]) n1_s[tid] = n1_s[tid + i];
        i = i / 2;
        __syncthreads();
    }

    if (tid == 0) n1max[blockIdx.x] = n1_s[0];

}


template<unsigned int WarpSize, unsigned int ISIZE>
__global__ void cuComputeN1withSparityofA_SpMMv1(int *aPtr, int *aIndex, int *mPtr, int *mIndex,
                                                 int *I, int n1max, int nCol, int *n1, int *atomic) {
    __shared__ int I_s[ISIZE];
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp
    int tid = threadIdx.x / WarpSize;

    int col, col_s, col_e;
    int fcol, fcol_s, fcol_e;
    int othercol, ocol, ocol_s, ocol_e;
    int bV, bV1, bV2;
    int j, j1, j2, tN, flag, idx;

    int seg = ISIZE / (blockDim.x / WarpSize);
    int TSEG = tid * seg;

    for (col = warp_id; col < nCol; col += offset) {
        //read the columns
        col_s = mPtr[col];
        col_e = mPtr[col + 1];
        bV = col_e - col_s;

        //for the first column, all data are written into I
        fcol = mIndex[col_s];
        fcol_s = aPtr[fcol];
        fcol_e = aPtr[fcol + 1];
        bV1 = fcol_e - fcol_s;

        atomic[col] = bV1;

        for (j = lane; j < bV1; j += WarpSize) {
            I_s[TSEG + j] = aIndex[j + fcol_s];
        }//for j

        __syncthreads();


        //for other columns
        for (othercol = 1; othercol < bV; othercol++) {
            ocol = mIndex[col_s + othercol];
            ocol_s = aPtr[ocol];
            ocol_e = aPtr[ocol + 1];

            bV2 = ocol_e - ocol_s;
            //if the element in other column does not exist in I, append it to I
            tN = atomic[col];
            for (j = lane; j < bV2; j += WarpSize) {
                idx = aIndex[j + ocol_s];
                flag = -1;
                for (j1 = 0; j1 < tN; j1++) {
                    if (idx == I_s[TSEG + j1]) {
                        flag = 1;
                        break;
                    }
                }
                if (flag == -1) {
                    j2 = atomicAdd(&atomic[col], 1);
                    if (j2 >= seg) {
                        printf("exceed the maximum shared size for col = %d\n", col);
                        break;
                    }
                    I_s[TSEG + j2] = idx;
                }

            }//for j
            __syncthreads();


        }//for othercol

        n1[col] = atomic[col];

    }//end for col
}

template<unsigned int WarpSize>
__global__ void cuComputeN1withSparityofA2_SpMMv1(int *aPtr, int *aIndex, int *mPtr, int *mIndex,
                                                  int *I, int n1max, int nCol, int *n1, int *atomic) {

    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp


    int col, col_s, col_e;
    int fcol, fcol_s, fcol_e;
    int othercol, ocol, ocol_s, ocol_e;
    int bV, bV1, bV2, tN, flag;
    int j, j1, j2, idx;

    for (col = warp_id; col < nCol; col += offset) {
        //read the columns
        col_s = mPtr[col];
        col_e = mPtr[col + 1];
        bV = col_e - col_s;

        //for the first column, all data are written into I
        fcol = mIndex[col_s];
        fcol_s = aPtr[fcol];
        fcol_e = aPtr[fcol + 1];
        bV1 = fcol_e - fcol_s;

        atomic[col] = bV1;

        for (j = lane; j < bV1; j += WarpSize) {
            I[col * n1max + j] = aIndex[j + fcol_s];
        }//for j

        __syncthreads();

        //for other columns
        for (othercol = 1; othercol < bV; othercol++) {
            ocol = mIndex[col_s + othercol];
            ocol_s = aPtr[ocol];
            ocol_e = aPtr[ocol + 1];

            bV2 = ocol_e - ocol_s;
            //if the element in other column does not exist in I, append it to I
            tN = atomic[col];

            for (j = lane; j < bV2; j += WarpSize) {
                idx = aIndex[j + ocol_s];
                flag = -1;
                for (j1 = 0; j1 < tN; j1++) {
                    if (idx == I[col * n1max + j1]) {
                        flag = 1;
                        break;
                    }
                }
                if (flag == -1) {
                    j2 = atomicAdd(&atomic[col], 1);
                    if (j2 >= n1max) {
                        printf("exceed the maximum shared size for col = %d\n", col);
                        break;
                    }
                    I[col * n1max + j2] = idx;
                }

            }//for j 0:bv2

            __syncthreads();

        }//for othercol

        n1[col] = atomic[col];

    }//end for col
}

template<unsigned int WarpSize>
__global__ void computeJ_SSPAIv10(int *MPtr, int *MIndex,
                                  int nCol, int *J, int *jPTR, int N2MAX) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);//index of threads in warp

    int col, col_s, col_e, j, bV;

    for (col = warp_id; col < nCol; col += offset) {
        col_s = MPtr[col];
        col_e = MPtr[col + 1];
        bV = col_e - col_s;
        jPTR[col] = bV;
        for (j = lane; j < bV; j += WarpSize) {
            J[col * N2MAX + j] = MIndex[j + col_s];
        }
    }//for col

}

template<unsigned int WarpSize>
__global__ void computeJ2_SSPAIv10(int *MPtr, int *MIndex,
                                   int nCol, int *J, int *jPTR, int N2MAX, int sK) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);//index of threads in warp

    int col, col_s, col_e, j, bV;
    int colK;

    for (col = warp_id; col < nCol; col += offset) {
        colK = col + sK;
        col_s = MPtr[colK];
        col_e = MPtr[colK + 1];
        bV = col_e - col_s;
        jPTR[col] = bV;
        for (j = lane; j < bV; j += WarpSize) {
            J[col * N2MAX + j] = MIndex[j + col_s];
        }
    }//for col

}

template<unsigned int WarpSize>
__global__ void computeI_iter_Symbol_SpMMv1(int *APtr, int *AIndex, int nCol,
                                            int *I, int *iPTR, int N1MAX, int *J, int *jPTR, int N2MAX, int *atomic) {

    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp
    //int tid = threadIdx.x / WarpSize;

    int col, fcol, fcol_s, fcol_e, bV, bV1, jN, tN;
    int ocol, othercol, othercol_s, othercol_e, j, j1, j2;

    int JBnd;
    int IBnd;

    //int seg = SIZE_I_SHARED/CounterSize;

    for (col = warp_id; col < nCol; col += offset) {
        JBnd = col * N2MAX;
        IBnd = col * N1MAX;

        //read the columns
        fcol = J[JBnd];
        fcol_s = APtr[fcol];
        fcol_e = APtr[fcol + 1];
        bV = fcol_e - fcol_s;

        atomic[col] = bV;

        //for the first column, all data are written into I
        for (j = lane; j < bV; j += WarpSize) {
            I[IBnd + j] = AIndex[j + fcol_s];
        }//for j

        __syncthreads();

        //for other columns
        jN = jPTR[col];
        for (othercol = 1; othercol < jN; othercol++) {
            ocol = J[JBnd + othercol];
            othercol_s = APtr[ocol];
            othercol_e = APtr[ocol + 1];

            bV1 = othercol_e - othercol_s;
            //if the element in other column does not exist in I, append it to I
            tN = atomic[col];
            for (j = lane; j < bV1; j += WarpSize) {
                int idx = AIndex[j + othercol_s];
                int flag = 1;
                for (j1 = 0; j1 < tN; j1++) {
                    if (idx == I[IBnd + j1]) {
                        flag = -1;
                        break;
                    }
                }

                if (flag == 1) {
                    j2 = atomicAdd(&atomic[col], 1);
                    I[IBnd + j2] = idx;
                }
            }//for j


            __syncthreads();

        }//for othercol

        iPTR[col] = atomic[col];

        tN = iPTR[col];
        //ODD_EVEN
        for (int i = 1; i <= tN; i++) {
            if (i % 2 == 1) {
                for (int j1 = lane; j1 < tN; j1 += WarpSize) {
                    if ((j1 * 2 + 1) < tN && (I[IBnd + j1 * 2] > I[IBnd + j1 * 2 + 1])) {
                        bV1 = I[IBnd + j1 * 2];
                        I[IBnd + j1 * 2] = I[IBnd + j1 * 2 + 1];
                        I[IBnd + j1 * 2 + 1] = bV1;
                    }
                }
                //__syncthreads() ;
            } else {
                for (int j2 = lane; j2 < tN; j2 += WarpSize) {
                    if ((j2 * 2 + 2) < tN && (I[IBnd + j2 * 2 + 1] > I[IBnd + j2 * 2 + 2])) {
                        bV1 = I[IBnd + j2 * 2 + 2];
                        I[IBnd + j2 * 2 + 2] = I[IBnd + j2 * 2 + 1];
                        I[IBnd + j2 * 2 + 1] = bV1;
                    }
                }
                //__syncthreads() ;
            }
            if (WarpSize > 32) {
                __syncthreads();
            }
        }

    }//for col
}

template<unsigned int WarpSize, unsigned int CounterSize, unsigned int SIZE_I_SHARED>
__global__ void computeI_SSPAIv10(int *APtr, int *AIndex, int nCol,
                                  int *I, int *iPTR, int N1MAX, int *J, int *jPTR, int N2MAX) {

    __shared__ int I_s[SIZE_I_SHARED];
    __shared__ int counter_s[CounterSize];
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp
    int tid = threadIdx.x / WarpSize;

    int col, fcol, fcol_s, fcol_e, bV, bV1;
    int ocol, othercol, othercol_s, othercol_e, j, j1;

    int seg = SIZE_I_SHARED / CounterSize;

    for (col = warp_id; col < nCol; col += offset) {
        //read the columns
        fcol = J[col * N2MAX];
        fcol_s = APtr[fcol];
        fcol_e = APtr[fcol + 1];
        bV = fcol_e - fcol_s;

        counter_s[tid] = bV;

        __syncthreads();

        //for the first column, all data are written into I
        for (j = lane; j < bV; j += WarpSize) {
            I[col * N1MAX + j] = AIndex[j + fcol_s];
        }//for j

        __syncthreads();

        //for other columns
        for (othercol = 1; othercol < jPTR[col]; othercol++) {
            ocol = J[col * N2MAX + othercol];
            othercol_s = APtr[ocol];
            othercol_e = APtr[ocol + 1];

            bV1 = othercol_e - othercol_s;
            //if the element in other column does not exist in I, append it to I
            for (j = lane; j < bV1; j += WarpSize) {
                I_s[tid * seg + j] = AIndex[j + othercol_s];
                for (j1 = 0; j1 < counter_s[tid]; j1++) {
                    if (I_s[tid * seg + j] == I[col * N1MAX + j1]) {
                        I_s[tid * seg + j] = -1;
                        break;
                    }
                }
            }//for j

            __syncthreads();

            if (lane == 0) {
                for (j = 0; j < bV1; j++) {
                    if (I_s[tid * seg + j] != -1) {
                        I[col * N1MAX + counter_s[tid]] = I_s[tid * seg + j];
                        //counter++;
                        counter_s[tid] += 1;
                    }
                }
            }

            __syncthreads();
        }//for othercol

        iPTR[col] = counter_s[tid];


        //ODD_EVEN
        //if(lane==0){
        for (int i = 1; i <= counter_s[tid]; i++) {
            if (i % 2 == 1) {
                for (int j1 = lane; j1 < counter_s[tid]; j1 += WarpSize) {
                    if (((j1 + 1) % 2 == 1) && (I[col * N1MAX + j1] > I[col * N1MAX + j1 + 1]) &&
                        (j1 + 1) < counter_s[tid]) {
                        bV1 = I[col * N1MAX + j1];
                        I[col * N1MAX + j1] = I[col * N1MAX + j1 + 1];
                        I[col * N1MAX + j1 + 1] = bV1;
                    }
                }
                //__syncthreads() ;
            } else {
                for (int j2 = lane; j2 < counter_s[tid]; j2 += WarpSize) {
                    if (((j2 + 1) % 2 == 0) && (I[col * N1MAX + j2] > I[col * N1MAX + j2 + 1]) &&
                        (j2 + 1) < counter_s[tid]) {
                        bV1 = I[col * N1MAX + j2];
                        I[col * N1MAX + j2] = I[col * N1MAX + j2 + 1];
                        I[col * N1MAX + j2 + 1] = bV1;
                    }
                }
                //__syncthreads() ;
            }
        }

        // }
    }//for col
}

template<unsigned int WarpSize, unsigned int SIZE_I_SHARED>
__global__ void computeIShared_Symbol_SpMMv1(int *APtr, int *AIndex, int nCol,
                                             int *I, int *iPTR, int N1MAX, int *J, int *jPTR, int N2MAX, int *atomic) {

    __shared__ int I_s[SIZE_I_SHARED];
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp
    int tid = threadIdx.x / WarpSize;

    int col, fcol, fcol_s, fcol_e, bV, bV1, jN, tN;
    int ocol, othercol, othercol_s, othercol_e, j, j1, j2;

    int seg = SIZE_I_SHARED / (blockDim.x / WarpSize);
    int TSEG = seg * tid;
    int JBnd;
    int IBnd;

    for (col = warp_id; col < nCol; col += offset) {

        JBnd = col * N2MAX;
        IBnd = col * N1MAX;

        //read the columns
        fcol = J[JBnd];
        fcol_s = APtr[fcol];
        fcol_e = APtr[fcol + 1];
        bV = fcol_e - fcol_s;

        atomic[col] = bV;

        //for the first column, all data are written into I
        for (j = lane; j < bV; j += WarpSize) {
            I_s[TSEG + j] = AIndex[j + fcol_s];
        }//for j

        __syncthreads();


        jN = jPTR[col];
        //for other columns
        for (othercol = 1; othercol < jN; othercol++) {
            ocol = J[JBnd + othercol];
            othercol_s = APtr[ocol];
            othercol_e = APtr[ocol + 1];

            bV1 = othercol_e - othercol_s;
            //if the element in other column does not exist in I, append it to I
            tN = atomic[col];
            for (j = lane; j < bV1; j += WarpSize) {
                int idx = AIndex[j + othercol_s];
                int flag = 1;
                for (j1 = 0; j1 < tN; j1++) {
                    if (idx == I_s[TSEG + j1]) {
                        flag = -1;
                        break;
                    }
                }

                if (flag == 1) {
                    j2 = atomicAdd(&atomic[col], 1);
                    I_s[TSEG + j2] = idx;
                }
            }//for j

            __syncthreads();


        }//for othercol

        iPTR[col] = atomic[col];

        tN = iPTR[col];
        //ODD_EVEN
        for (int i = 1; i <= tN; i++) {
            if (i % 2 == 1) {
                for (int j1 = lane; j1 < tN; j1 += WarpSize) {
                    if ((j1 * 2 + 1) < tN && (I_s[TSEG + j1 * 2] > I_s[TSEG + j1 * 2 + 1])) {
                        bV1 = I_s[TSEG + j1 * 2];
                        I_s[TSEG + j1 * 2] = I_s[TSEG + j1 * 2 + 1];
                        I_s[TSEG + j1 * 2 + 1] = bV1;
                    }
                }
                //__syncthreads() ;
            } else {
                for (int j2 = lane; j2 < tN; j2 += WarpSize) {
                    if ((j2 * 2 + 2) < tN && (I_s[TSEG + j2 * 2 + 1] > I_s[TSEG + j2 * 2 + 2])) {
                        bV1 = I_s[TSEG + j2 * 2 + 2];
                        I_s[TSEG + j2 * 2 + 2] = I_s[TSEG + j2 * 2 + 1];
                        I_s[TSEG + j2 * 2 + 1] = bV1;
                    }
                }
                //__syncthreads() ;
            }
            //__syncthreads() ;
        }


        //copy shared memory to I
        for (j = lane; j < tN; j += WarpSize) {
            I[IBnd + j] = I_s[TSEG + j];
        }

    }//for col
}

template<unsigned int WarpSize, unsigned int SIZE_I_SHARED>
__global__ void computeIShared_iter_Symbol_SpMMv1(int *APtr, int *AIndex, int nCol,
                                                  int *I, int *iPTR, int N1MAX, int *J, int *jPTR, int N2MAX,
                                                  int *atomic) {

    __shared__ int I_s[SIZE_I_SHARED];
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp
    int tid = threadIdx.x / WarpSize;

    int col, fcol, fcol_s, fcol_e, bV, bV1, jN, tN;
    int ocol, othercol, othercol_s, othercol_e, j, j1, j2;

    int seg = SIZE_I_SHARED / (blockDim.x / WarpSize);
    int TSEG = seg * tid;
    int JBnd;
    int IBnd;

    for (col = warp_id; col < nCol; col += offset) {

        JBnd = col * N2MAX;
        IBnd = col * N1MAX;

        //read the columns
        fcol = J[JBnd];
        fcol_s = APtr[fcol];
        fcol_e = APtr[fcol + 1];
        bV = fcol_e - fcol_s;

        atomic[col] = bV;

        //for the first column, all data are written into I
        for (j = lane; j < bV; j += WarpSize) {
            I_s[TSEG + j] = AIndex[j + fcol_s];
        }//for j

        __syncthreads();


        jN = jPTR[col];
        //for other columns
        for (othercol = 1; othercol < jN; othercol++) {
            ocol = J[JBnd + othercol];
            othercol_s = APtr[ocol];
            othercol_e = APtr[ocol + 1];

            bV1 = othercol_e - othercol_s;
            //if the element in other column does not exist in I, append it to I
            tN = atomic[col];
            for (j = lane; j < bV1; j += WarpSize) {
                int idx = AIndex[j + othercol_s];
                int flag = 1;
                for (j1 = 0; j1 < tN; j1++) {
                    if (idx == I_s[TSEG + j1]) {
                        flag = -1;
                        break;
                    }
                }

                if (flag == 1) {
                    j2 = atomicAdd(&atomic[col], 1);
                    I_s[TSEG + j2] = idx;
                }
            }//for j

            __syncthreads();


        }//for othercol

        iPTR[col] = atomic[col];

        tN = iPTR[col];
        //ODD_EVEN
        for (int i = 1; i <= tN; i++) {
            if (i % 2 == 1) {
                for (int j1 = lane; j1 < tN; j1 += WarpSize) {
                    if ((j1 * 2 + 1) < tN && (I_s[TSEG + j1 * 2] > I_s[TSEG + j1 * 2 + 1])) {
                        bV1 = I_s[TSEG + j1 * 2];
                        I_s[TSEG + j1 * 2] = I_s[TSEG + j1 * 2 + 1];
                        I_s[TSEG + j1 * 2 + 1] = bV1;
                    }
                }
                //__syncthreads() ;
            } else {
                for (int j2 = lane; j2 < tN; j2 += WarpSize) {
                    if ((j2 * 2 + 2) < tN && (I_s[TSEG + j2 * 2 + 1] > I_s[TSEG + j2 * 2 + 2])) {
                        bV1 = I_s[TSEG + j2 * 2 + 2];
                        I_s[TSEG + j2 * 2 + 2] = I_s[TSEG + j2 * 2 + 1];
                        I_s[TSEG + j2 * 2 + 1] = bV1;
                    }
                }
                //__syncthreads() ;
            }
            if (WarpSize > 32) {
                __syncthreads();
            }
        }

        //copy shared memory to I
        for (j = lane; j < tN; j += WarpSize) {
            I[IBnd + j] = I_s[TSEG + j];
        }

    }//for col
}

template<unsigned int WarpSize>
__global__ void computeI_Symbol_SpMMv1(int *APtr, int *AIndex, int nCol,
                                       int *I, int *iPTR, int N1MAX, int *J, int *jPTR, int N2MAX, int *atomic) {

    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp
    //int tid = threadIdx.x / WarpSize;

    int col, fcol, fcol_s, fcol_e, bV, bV1, jN, tN;
    int ocol, othercol, othercol_s, othercol_e, j, j1, j2;

    int JBnd;
    int IBnd;

    //int seg = SIZE_I_SHARED/CounterSize;

    for (col = warp_id; col < nCol; col += offset) {
        JBnd = col * N2MAX;
        IBnd = col * N1MAX;

        //read the columns
        fcol = J[JBnd];
        fcol_s = APtr[fcol];
        fcol_e = APtr[fcol + 1];
        bV = fcol_e - fcol_s;

        atomic[col] = bV;

        //for the first column, all data are written into I
        for (j = lane; j < bV; j += WarpSize) {
            I[IBnd + j] = AIndex[j + fcol_s];
        }//for j

        __syncthreads();

        //for other columns
        jN = jPTR[col];
        for (othercol = 1; othercol < jN; othercol++) {
            ocol = J[JBnd + othercol];
            othercol_s = APtr[ocol];
            othercol_e = APtr[ocol + 1];

            bV1 = othercol_e - othercol_s;
            //if the element in other column does not exist in I, append it to I
            tN = atomic[col];
            for (j = lane; j < bV1; j += WarpSize) {
                int idx = AIndex[j + othercol_s];
                int flag = 1;
                for (j1 = 0; j1 < tN; j1++) {
                    if (idx == I[IBnd + j1]) {
                        flag = -1;
                        break;
                    }
                }

                if (flag == 1) {
                    j2 = atomicAdd(&atomic[col], 1);
                    I[IBnd + j2] = idx;
                }
            }//for j


            __syncthreads();

        }//for othercol

        iPTR[col] = atomic[col];

        tN = iPTR[col];
        //ODD_EVEN
        for (int i = 1; i <= tN; i++) {
            if (i % 2 == 1) {
                for (int j1 = lane; j1 < tN; j1 += WarpSize) {
                    if ((j1 * 2 + 1) < tN && (I[IBnd + j1 * 2] > I[IBnd + j1 * 2 + 1])) {
                        bV1 = I[IBnd + j1 * 2];
                        I[IBnd + j1 * 2] = I[IBnd + j1 * 2 + 1];
                        I[IBnd + j1 * 2 + 1] = bV1;
                    }
                }
                //__syncthreads() ;
            } else {
                for (int j2 = lane; j2 < tN; j2 += WarpSize) {
                    if ((j2 * 2 + 2) < tN && (I[IBnd + j2 * 2 + 1] > I[IBnd + j2 * 2 + 2])) {
                        bV1 = I[IBnd + j2 * 2 + 2];
                        I[IBnd + j2 * 2 + 2] = I[IBnd + j2 * 2 + 1];
                        I[IBnd + j2 * 2 + 1] = bV1;
                    }
                }
                //__syncthreads() ;
            }
            //__syncthreads() ;
        }

    }//for col
}

template<unsigned int WarpSize>
__global__ void OE_ParallelSort(int nCol, int *I, int *iPTR, int N1MAX) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp

    int col, i, j1, j2, bV1, SZ;


    for (col = warp_id; col < nCol; col += offset) {
        SZ = iPTR[col];
        //ODD_EVEN
        for (i = 1; i <= SZ; i++) {
            if (i % 2 == 1) {
                for (j1 = lane; j1 < SZ; j1 += WarpSize) {
                    if (((j1 + 1) % 2 == 1) && (I[col * N1MAX + j1] > I[col * N1MAX + j1 + 1]) && (j1 + 1) < SZ) {
                        bV1 = I[col * N1MAX + j1];
                        I[col * N1MAX + j1] = I[col * N1MAX + j1 + 1];
                        I[col * N1MAX + j1 + 1] = bV1;
                    }
                }
                __syncthreads();
            } else {
                for (j2 = lane; j2 < SZ; j2 += WarpSize) {
                    if (((j2 + 1) % 2 == 0) && (I[col * N1MAX + j2] > I[col * N1MAX + j2 + 1]) && (j2 + 1) < SZ) {
                        bV1 = I[col * N1MAX + j2];
                        I[col * N1MAX + j2] = I[col * N1MAX + j2 + 1];
                        I[col * N1MAX + j2 + 1] = bV1;
                    }
                }
                __syncthreads();
            }
        }
    }//for col
}

template<unsigned int WarpSize, unsigned int CounterSize>
__global__ void computeI1_SSPAIv10(int *APtr, int *AIndex, int nCol,
                                   int *I, int *iPTR, int N1MAX, int *J, int *jPTR, int N2MAX) {

    __shared__ int counter_s[CounterSize];
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp
    int tid = threadIdx.x / WarpSize;

    int col, fcol, fcol_s, fcol_e, bV, bV1;
    int ocol, othercol, othercol_s, othercol_e, j, j1;

    //int seg = SIZE_I_SHARED/CounterSize;

    for (col = warp_id; col < nCol; col += offset) {
        //read the columns
        fcol = J[col * N2MAX];
        fcol_s = APtr[fcol];
        fcol_e = APtr[fcol + 1];
        bV = fcol_e - fcol_s;

        counter_s[tid] = bV;

        __syncthreads();

        //for the first column, all data are written into I
        for (j = lane; j < bV; j += WarpSize) {
            I[col * N1MAX + j] = AIndex[j + fcol_s];
        }//for j

        __syncthreads();


        //for other columns
        for (othercol = 1; othercol < jPTR[col]; othercol++) {
            ocol = J[col * N2MAX + othercol];
            othercol_s = APtr[ocol];
            othercol_e = APtr[ocol + 1];

            bV1 = othercol_e - othercol_s;
            //if the element in other column does not exist in I, append it to I
            int counter = 0;
            for (j = 0; j < bV1; j += 1) {
                int idx = AIndex[j + othercol_s];
                int flag = 1;
                for (j1 = 0; j1 < counter_s[tid]; j1++) {
                    if (idx == I[col * N1MAX + j1]) {
                        flag = -1;
                        break;
                    }
                }

                if (flag == 1) {
                    I[col * N1MAX + counter_s[tid] + counter] = idx;
                    counter++;
                }
            }//for j

            counter_s[tid] += counter;
            __syncthreads();

        }//for othercol

        iPTR[col] = counter_s[tid];


        //}
    }
}

template<unsigned int WarpSize>
__global__ void ComputeTildeACSR_SSPAIv10(double *A1, double *AData, int *APtr, int *AIndex, int nCol,
                                          int *I, int *iPTR, int *J, int *jPTR, int N1MAX, int N2MAX) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);//index of threads in warp

    int col, i, irow, j, jcol, jcol_b, jcol_e, jcol1;
    int AM, AN;
    double idata;

    for (col = warp_id; col < nCol; col += offset) {
        AM = iPTR[col];
        AN = jPTR[col];
        for (i = 0; i < AM; i++) {
            irow = I[col * N1MAX + i];
            for (j = lane; j < AN; j += WarpSize) {
                jcol = J[col * N2MAX + j];
                jcol_b = APtr[jcol];
                jcol_e = APtr[jcol + 1];

                idata = 0.0;
                for (jcol1 = jcol_b; jcol1 < jcol_e; jcol1++) {
                    if (AIndex[jcol1] == irow) {
                        idata = AData[jcol1];
                        break;
                    }
                }

                A1[col * N1MAX * N2MAX + i * N2MAX + j] = idata;

            }//for j

            __syncthreads();
        }//for i

    }//for col
}

template<unsigned int WarpSize>
__global__ void ComputeTildeE_SSPAIv10(int *E, int *I, int *iPTR, int N1MAX, int nCol) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);//index of threads in warp

    int col, i, AM;

    for (col = warp_id; col < nCol; col += offset) {
        E[col] = -1;
        AM = iPTR[col];
        for (i = lane; i < AM; i += WarpSize) {
            if (I[col * N1MAX + i] == col) {
                E[col] = i;
                break;
            }
        }//for i
    }//for col
}

template<unsigned int WarpSize>
__global__ void ComputeTildeE2_SSPAIv10(int *E, int *I, int *iPTR, int N1MAX, int nCol, int sK) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);//index of threads in warp

    int col, i, AM, colK;

    for (col = warp_id; col < nCol; col += offset) {
        colK = sK + col;
        E[col] = -1;
        AM = iPTR[col];
        for (i = lane; i < AM; i += WarpSize) {
            if (I[col * N1MAX + i] == colK) {
                E[col] = i;
                break;
            }
        }//for i
    }//for col
}

template<unsigned int WarpSize>
__global__ void Sol_SSPAIv10(double *Q, double *R, double *X, int *E, int *jPTR, int N1MAX, int N2MAX, int nCol) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);//index of threads in warp

    int col, AN, i, j;

    //compute Q^TE
    for (col = warp_id; col < nCol; col += offset) {
        AN = jPTR[col];
        if (E[col] == -1) {
            for (i = lane; i < AN; i += WarpSize) {
                X[col * N2MAX + i] = 0.0;
            }//for i
        } else {
            for (i = lane; i < AN; i += WarpSize) {
                X[col * N2MAX + i] = Q[col * N1MAX * N2MAX + E[col] * N2MAX + i];
            }//for i
        }

        __syncthreads();

        //solving the upper triangular system
        for (i = AN - 1; i >= 0; i--) {
            if (lane == 0) { X[col * N2MAX + i] /= R[col * N2MAX * N2MAX + i * N2MAX + i]; }

            __syncthreads();

            for (j = lane; j < i; j += WarpSize) {
                X[col * N2MAX + j] -= R[col * N2MAX * N2MAX + j * N2MAX + i] * X[col * N2MAX + i];//csr
                //X[col*N2MAX+j] -= R[col*N2MAX*N2MAX+i*N2MAX+j] * X[col*N2MAX+i]; //csc
            }// for j
            __syncthreads();
        }//for i

    }//for col

}

template<unsigned int WarpSize, unsigned int SIZE_R_SHARED>
__global__ void QR_RShared_SSPAIv10(double *Q, double *R, int *iPTR, int *jPTR, int N1MAX, int N2MAX, int nCol) {
    __shared__ double R_s[SIZE_R_SHARED];
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);//index of threads in warp

    int tid = threadIdx.x / WarpSize;

    int col, AM, AN;
    int i, j, k;
    double rii, tR;

    int segR = SIZE_R_SHARED * WarpSize / blockDim.x;

    for (col = warp_id; col < nCol; col += offset) {
        //for each col corresponding to a tilde of A
        AM = iPTR[col];
        AN = jPTR[col];

        for (i = 0; i < AN; i++) {

            /*Compute R in parallel an put them into shared memory*/
            for (j = lane + i; j < AN; j += WarpSize) {
                tR = 0.0;
                for (k = 0; k < AM; k++) {
                    tR += Q[col * N1MAX * N2MAX + i + k * N2MAX] * Q[col * N1MAX * N2MAX + j + k * N2MAX];
                }//for k

                R_s[tid * segR + j - i] = tR;
            }//for j

            __syncthreads();

            rii = sqrt(R_s[tid * segR]);
            //normalize column i of Q
            for (j = lane; j < AM; j += WarpSize) {
                Q[col * N1MAX * N2MAX + j * N2MAX + i] /= rii;
            }//for j

            __syncthreads();

            //compute projection factors
            for (j = lane + i; j < AN; j += WarpSize) {
                R_s[tid * segR + j - i] /= rii;
                R[col * N2MAX * N2MAX + i * N2MAX + j] = R_s[tid * segR + j - i];
            }//for j

            __syncthreads();

            for (j = lane + i + 1; j < AN; j += WarpSize) {
                for (k = 0; k < AM; k++) {
                    Q[col * N1MAX * N2MAX + k * N2MAX + j] -=
                            R_s[tid * segR + j - i] * Q[col * N1MAX * N2MAX + k * N2MAX + i];
                }
            }

            __syncthreads();

        }//for i

    }//for col

}

template<unsigned int WarpSize>
__global__ void modifyData4_SSPAIv10(double *mData, int *mPtr,
                                     double *X, int *jPTR, int n2max, int nCol) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp

    int col, j;
    int col_s;

    for (col = warp_id; col < nCol; col += offset) {
        col_s = mPtr[col];
        for (j = lane; j < jPTR[col]; j += WarpSize) {
            mData[col_s + j] = X[col * n2max + j];
        }
    }
}

template<unsigned int WarpSize>
__global__ void modifyIndexAndData_SSPAIv10(double *mData, int *mIndex, int *mPtr,
                                            double *X, int *J, int *jPTR, int n2max, int nCol) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp

    int col, j;
    int col_s;

    for (col = warp_id; col < nCol; col += offset) {
        col_s = mPtr[col];
        for (j = lane; j < jPTR[col]; j += WarpSize) {
            mData[col_s + j] = X[col * n2max + j];
            mIndex[col_s + j] = J[col * n2max + j];
        }
    }
}

template<unsigned int WarpSize>
__global__ void modifyData24_SSPAIv10(double *mData, int *mPtr,
                                      double *X, int *jPTR, int n2max, int nCol, int sK) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp

    int col, j;
    int col_s, colK;

    for (col = warp_id; col < nCol; col += offset) {
        colK = col + sK;
        col_s = mPtr[colK];
        for (j = lane; j < jPTR[col]; j += WarpSize) {
            mData[col_s + j] = X[col * n2max + j];
        }
    }
}

template<unsigned int WarpSize>
__global__ void modifyIndexAndData2_SSPAIv10(double *mData, int *mIndex, int *mPtr,
                                             double *X, int *J, int *jPTR, int n2max, int nCol, int sK) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp

    int col, j;
    int col_s, colK;

    for (col = warp_id; col < nCol; col += offset) {
        colK = col + sK;
        col_s = mPtr[colK];
        for (j = lane; j < jPTR[col]; j += WarpSize) {
            mData[col_s + j] = X[col * n2max + j];
            mIndex[col_s + j] = J[col * n2max + j];
        }
    }
}

template<unsigned int WarpSize>
__global__ void modifyIndexAndData3_SSPAIv10(double *mData, int *mIndex, int *mPtr,
                                             double *mTmpData, int *mTmpIndex, int *mTmpPtr, int *aPtr, int nCol) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp

    int col, j;
    int col_s, col_s1;

    for (col = warp_id; col < nCol; col += offset) {
        col_s = mPtr[col];
        col_s1 = aPtr[col];
        for (j = lane; j < mTmpPtr[col]; j += WarpSize) {
            mData[col_s + j] = mTmpData[col_s1 + j];
            mIndex[col_s + j] = mTmpIndex[col_s1 + j];
        }
    }
}

float StaticSPAIv20(CSC_Matrix *devA, CSC_Matrix *devM,
    int colStart, int colCount) 
{
printf("StaticSPAIv20: colStart=%d, colCount=%d\n", colStart, colCount);

hipEvent_t start, stop;
float elapsedTime, preTime=0, computeTime=0, postTime=0;
const int threadsPerBlock = 256;

// -------------------- Compute n2max --------------------
int blocksPerGrid = 15*8*4;
int *n2 = (int*)malloc(sizeof(int)*blocksPerGrid);
int *dev_n2; hipMalloc((void**)&dev_n2,sizeof(int)*blocksPerGrid);

cuComputeN2MAXwithSparityofA<threadsPerBlock>
<<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, colCount, dev_n2);
hipMemcpy(n2,dev_n2,sizeof(int)*blocksPerGrid,hipMemcpyDeviceToHost);
int n2max=0; for(int i=0;i<blocksPerGrid;i++) n2max=max(n2max,n2[i]);
hipFree(dev_n2); free(n2);

printf("n2max=%d\n", n2max);

// -------------------- Compute n1max --------------------
int WarpSize = (n2max<=2?2:(n2max<=4?4:(n2max<=8?8:(n2max<=16?16:32))));
blocksPerGrid = (colCount-1)/(threadsPerBlock/WarpSize)+1;

int *dev_n1; hipMalloc((void**)&dev_n1,sizeof(int)*colCount);
int *dev_tI; hipMalloc((void**)&dev_tI,sizeof(int)*colCount*min(n2max*8,colCount));
int *dev_atomic; hipMalloc((void**)&dev_atomic,sizeof(int)*colCount);

cuComputeN1withSparityofA_SpMMv1<32,2048>
<<<blocksPerGrid,threadsPerBlock>>>(devA->mPtr,devA->mIndex,
                            devA->mPtr,devA->mIndex,
                            dev_tI, min(n2max*8,colCount),
                            colCount, dev_n1, dev_atomic);

hipFree(dev_tI); hipFree(dev_atomic); hipFree(dev_n1);

// -------------------- Allocate temporary buffers --------------------
double *dev_tildeA, *dev_R, *dev_X;
int *dev_J, *dev_jPTR, *dev_I, *dev_iPTR, *dev_E;

hipMalloc((void**)&dev_J,    sizeof(int)*colCount*n2max);
hipMalloc((void**)&dev_jPTR, sizeof(int)*colCount);
hipMalloc((void**)&dev_I,    sizeof(int)*colCount*WarpSize);
hipMalloc((void**)&dev_iPTR, sizeof(int)*colCount);
hipMalloc((void**)&dev_tildeA,sizeof(double)*colCount*WarpSize*n2max);
hipMalloc((void**)&dev_R,    sizeof(double)*colCount*n2max*n2max);
hipMalloc((void**)&dev_X,    sizeof(double)*colCount*n2max);
hipMalloc((void**)&dev_E,    sizeof(int)*colCount);

// -------------------- Step 1: compute J (use colStart) --------------------
computeJ2_SSPAIv10<32><<<blocksPerGrid,threadsPerBlock>>>(
devA->mPtr, devA->mIndex,
colCount, dev_J, dev_jPTR, n2max, colStart);

// -------------------- Step 2: compute I (use colStart) --------------------
computeI_iter_Symbol_SpMMv1<32><<<blocksPerGrid,threadsPerBlock>>>(
devA->mPtr, devA->mIndex,
colCount, dev_I, dev_iPTR, WarpSize,
dev_J, dev_jPTR, n2max, dev_atomic);

// -------------------- Step 3: Compute tilde(A) --------------------
ComputeTildeACSR_SSPAIv10<32><<<blocksPerGrid,threadsPerBlock>>>(
dev_tildeA, devA->mData, devA->mPtr, devA->mIndex,
colCount, dev_I, dev_iPTR, dev_J, dev_jPTR, WarpSize, n2max);

// -------------------- Step 4: QR factorization -----------------·---
QR_RShared_SSPAIv10<32,256><<<blocksPerGrid,threadsPerBlock>>>(
dev_tildeA, dev_R, dev_iPTR, dev_jPTR,
WarpSize, n2max, colCount);

// -------------------- Step 5: Compute tilde(E) --------------------
ComputeTildeE2_SSPAIv10<32><<<blocksPerGrid,threadsPerBlock>>>(
dev_E, dev_I, dev_iPTR, WarpSize, colCount, colStart);

// -------------------- Step 6: Solve for X --------------------
Sol_SSPAIv10<32><<<blocksPerGrid,threadsPerBlock>>>(
dev_tildeA, dev_R, dev_X,
dev_E, dev_jPTR, WarpSize, n2max, colCount);

// -------------------- Step 7: Write results back --------------------
devM->n        = devA->n;
devM->nRow     = devA->nRow;
devM->nCol     = colCount;
devM->nonzeroes= devA->nonzeroes; // 或者分片非零元数（需预处理）

hipMalloc((void**)&devM->mPtr,   sizeof(int)*(colCount+1));
hipMalloc((void**)&devM->mIndex, sizeof(int)*devM->nonzeroes);
hipMalloc((void**)&devM->mData,  sizeof(double)*devM->nonzeroes);

// 注意这里加 colStart 偏移
hipMemcpy(devM->mPtr, devA->mPtr+colStart, sizeof(int)*(colCount+1), hipMemcpyDeviceToDevice);
hipMemcpy(devM->mIndex, devA->mIndex, sizeof(int)*devM->nonzeroes, hipMemcpyDeviceToDevice);
hipMemcpy(devM->mData, dev_X, sizeof(double)*devM->nonzeroes, hipMemcpyDeviceToDevice);

// -------------------- Free temporaries --------------------
hipFree(dev_J); hipFree(dev_jPTR);
hipFree(dev_I); hipFree(dev_iPTR);
hipFree(dev_tildeA); hipFree(dev_R); hipFree(dev_X); hipFree(dev_E);

return preTime + computeTime + postTime;
}


/*
 *Test for Reading the matrix from the file that comes from the SuiteSparse Matrix Collection
 */
int main(int argc, char **argv) {
    int deviceCount = 0;
    hipGetDeviceCount(&deviceCount);
    printf("检测到 %d 张 DCU\n", deviceCount);
    if (deviceCount < 1) {
        printf("没有检测到 DCU，退出\n");
        return -1;
    }

    char filename[50];
    cout << "Input the matrix filename:" << endl;
    cin >> filename;

    // Host 上完整矩阵 A
    CSC_Matrix *CSC_A = (CSC_Matrix*)malloc(sizeof(CSC_Matrix));
    readMatrixToCSC(filename, CSC_A);

    // Host 上存放结果 M
    CSC_Matrix *CSC_M = (CSC_Matrix*)malloc(sizeof(CSC_Matrix));
    CSC_M->n = CSC_A->n;
    CSC_M->nCol = CSC_A->nCol;
    CSC_M->nRow = CSC_A->nRow;
    CSC_M->nonzeroes = CSC_A->nonzeroes;
    CSC_M->mData  = (double*)malloc(sizeof(double)*CSC_A->nonzeroes);
    CSC_M->mIndex = (int*)malloc(sizeof(int)*CSC_A->nonzeroes);
    CSC_M->mPtr   = (int*)malloc(sizeof(int)*(CSC_A->nCol+1));

    float totalPrecondTime = 0.0f;
    int totalCols = CSC_A->nCol;
    int perCard   = (totalCols + deviceCount - 1) / deviceCount;

    std::vector<float> deviceTimes(deviceCount, 0.0f);

    // === 并行执行 SPAI ===
    #pragma omp parallel for num_threads(deviceCount)
    for (int dev = 0; dev < deviceCount; dev++) {
        hipSetDevice(dev);

        int colStart = dev * perCard;
        int colCount = std::min(perCard, totalCols - colStart);
        if (colCount <= 0) continue;

        int nnzPart = CSC_A->mPtr[colStart+colCount] - CSC_A->mPtr[colStart];

        // 构造子矩阵
        CSC_Matrix *devCSC_A = (CSC_Matrix*)malloc(sizeof(CSC_Matrix));
        devCSC_A->n = CSC_A->n;
        devCSC_A->nRow = CSC_A->nRow;
        devCSC_A->nCol = colCount;
        devCSC_A->nonzeroes = nnzPart;

        CSC_Matrix *devCSC_M = (CSC_Matrix*)malloc(sizeof(CSC_Matrix));

        hipMalloc((void**)&devCSC_A->mData,  sizeof(double)*nnzPart);
        hipMalloc((void**)&devCSC_A->mIndex, sizeof(int)*nnzPart);
        hipMalloc((void**)&devCSC_A->mPtr,   sizeof(int)*(colCount+1));

        hipMemcpy(devCSC_A->mData,  CSC_A->mData  + CSC_A->mPtr[colStart],
                  nnzPart*sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(devCSC_A->mIndex, CSC_A->mIndex + CSC_A->mPtr[colStart],
                  nnzPart*sizeof(int), hipMemcpyHostToDevice);

        std::vector<int> hostPtrPart(colCount+1);
        for (int j=0; j<=colCount; j++) {
            hostPtrPart[j] = CSC_A->mPtr[colStart+j] - CSC_A->mPtr[colStart];
        }
        hipMemcpy(devCSC_A->mPtr, hostPtrPart.data(),
                  (colCount+1)*sizeof(int), hipMemcpyHostToDevice);

        // === 每线程计时 ===
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);
        hipEventRecord(start, 0);

        // ⭐ 修改：传入 colStart, colCount
        StaticSPAIv20(devCSC_A, devCSC_M, colStart, colCount);

        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        float elapsed = 0.0f;
        hipEventElapsedTime(&elapsed, start, stop);
        hipEventDestroy(start);
        hipEventDestroy(stop);

        deviceTimes[dev] = elapsed;

        // === 拷回结果 ===
        #pragma omp critical
        {
            hipMemcpy(CSC_M->mData  + CSC_A->mPtr[colStart],
                      devCSC_M->mData, sizeof(double)*nnzPart, hipMemcpyDeviceToHost);
            hipMemcpy(CSC_M->mIndex + CSC_A->mPtr[colStart],
                      devCSC_M->mIndex, sizeof(int)*nnzPart, hipMemcpyDeviceToHost);
            for (int j=0; j<=colCount; j++) {
                CSC_M->mPtr[colStart+j] = CSC_A->mPtr[colStart] + hostPtrPart[j];
            }
        }

        // 清理
        hipFree(devCSC_A->mData);
        hipFree(devCSC_A->mIndex);
        hipFree(devCSC_A->mPtr);
        hipFree(devCSC_M->mData);
        hipFree(devCSC_M->mIndex);
        hipFree(devCSC_M->mPtr);
        free(devCSC_A);
        free(devCSC_M);
    }

    for (int d=0; d<deviceCount; d++) {
        printf("设备 %d 预处理耗时: %.4f ms\n", d, deviceTimes[d]);
        totalPrecondTime += deviceTimes[d];
    }
    printf("预处理总耗时 (累加): %.4f ms\n", totalPrecondTime);

    // === CSC -> CSR 转换 ===
    CSR_Matrix *devCSR_A = (CSR_Matrix*)malloc(sizeof(CSR_Matrix));
    CSR_Matrix *devCSR_M = (CSR_Matrix*)malloc(sizeof(CSR_Matrix));

    devCSR_A->n = CSC_A->n;
    devCSR_A->nonzeroes = CSC_A->nonzeroes;
    devCSR_A->nCol = CSC_A->nCol;
    devCSR_A->nRow = CSC_A->nRow;
    hipMalloc((void**)&devCSR_A->mPtr,   sizeof(int)*(devCSR_A->nRow+1));
    hipMalloc((void**)&devCSR_A->mIndex, sizeof(int)*devCSR_A->nonzeroes);
    hipMalloc((void**)&devCSR_A->mData,  sizeof(double)*devCSR_A->nonzeroes);

    devCSR_M->n = CSC_M->n;
    devCSR_M->nonzeroes = CSC_M->nonzeroes;
    devCSR_M->nCol = CSC_M->nCol;
    devCSR_M->nRow = CSC_M->nRow;
    hipMalloc((void**)&devCSR_M->mPtr,   sizeof(int)*(devCSR_M->nRow+1));
    hipMalloc((void**)&devCSR_M->mIndex, sizeof(int)*devCSR_M->nonzeroes);
    hipMalloc((void**)&devCSR_M->mData,  sizeof(double)*devCSR_M->nonzeroes);

    cuCSC2CSR(CSC_A->nRow, CSC_A->nCol, CSC_A->nonzeroes,
              CSC_A->mData, CSC_A->mIndex, CSC_A->mPtr,
              devCSR_A->mData, devCSR_A->mIndex, devCSR_A->mPtr);

    cuCSC2CSR(CSC_M->nRow, CSC_M->nCol, CSC_M->nonzeroes,
              CSC_M->mData, CSC_M->mIndex, CSC_M->mPtr,
              devCSR_M->mData, devCSR_M->mIndex, devCSR_M->mPtr);

    // === BiCGStab 解方程 ===
    int MAX_ITER = 10000;
    double TOL = 1e-7;
    double *b = (double*)malloc(sizeof(double)*devCSR_A->n);
    double *x = (double*)malloc(sizeof(double)*devCSR_A->n);
    for (int j=0; j<devCSR_A->n; j++) { b[j]=1.0; x[j]=1.0; }

    double *dev_x, *dev_b;
    hipMalloc((void**)&dev_x, sizeof(double)*devCSR_A->n);
    hipMalloc((void**)&dev_b, sizeof(double)*devCSR_A->n);
    hipMemcpy(dev_x, x, devCSR_A->n*sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(dev_b, b, devCSR_A->n*sizeof(double), hipMemcpyHostToDevice);

    cublas2_pbicgstabv2(devCSR_A, devCSR_M, dev_b, dev_x, TOL, MAX_ITER);

    // === 输出结果 ===
    printf("求解完成，总预处理时间: %.4f ms\n", totalPrecondTime);

    // === 清理内存 ===
    free(b); free(x);
    free(CSC_A); free(CSC_M);
    free(devCSR_A); free(devCSR_M);
    hipFree(devCSR_A->mData); hipFree(devCSR_A->mIndex); hipFree(devCSR_A->mPtr);
    hipFree(devCSR_M->mData); hipFree(devCSR_M->mIndex); hipFree(devCSR_M->mPtr);
    hipFree(dev_x); hipFree(dev_b);

    return 0;
}
