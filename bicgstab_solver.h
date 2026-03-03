/* -----------------------------------------------------------------------------
    The programming is licensed to you under the NJNU Consortium license:
    Copyright (C)  2012  Jiaquan Gao 
	  E-mail: gaojiaquan@njnu.edu.cn

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
* ----------------------------------------------------------------------------*/

/* -----------------------------------------------------------------------------
 *  This .h file embraces many Conjugate Gradient (CG) algorithm solvers on the
 *  GPU
** ---------------------------------------------------------------------------*/

#include <time.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <hipblas.h>   
#include <hipsparse.h>
//#include <rocblas/rocblas.h>
//#include <rocsparse/rocsparse.h>
//#include "hipblas.h"
//#include "hipsparse.h"
//#include "hipsparse.h"

int cublas2_pbicgstabv2(CSR_Matrix *devA, CSR_Matrix *devM, double *dev_b, double *dev_x, double TOL, int MAX_ITER) {

    int n = devA->n;

    //-----------d_alpha, d_beta for hipsparseDcsrmv
    double d_alpha, d_beta;

    //-----------middle variables
    double *dev_r, *dev_rstar, *dev_p;
    double *dev_y, *dev_ay, *dev_s, *dev_z, *dev_az;

    hipMalloc((void **) &dev_r, sizeof(double) * n);
    hipMalloc((void **) &dev_rstar, sizeof(double) * n);
    hipMalloc((void **) &dev_p, sizeof(double) * n);
    hipMalloc((void **) &dev_y, sizeof(double) * n);
    hipMalloc((void **) &dev_ay, sizeof(double) * n);
    hipMalloc((void **) &dev_s, sizeof(double) * n);
    hipMalloc((void **) &dev_z, sizeof(double) * n);
    hipMalloc((void **) &dev_az, sizeof(double) * n);

    double err0, err, temp, temp1, temp2, alpha, beta, w;

    printf("-----------------The preconditioned BICGSTAB on CUBLAS is running---------------------\n");

    //-----------initialize cuBlas2 library
    hipblasHandle_t cuBlasHandle;
    hipblasStatus_t cuBlasStat;

    cuBlasStat = hipblasCreate(&cuBlasHandle);
    if (cuBlasStat != HIPBLAS_STATUS_SUCCESS) {
        printf("CUBLAS2 initialization failed \n");
        return 1;
    }

    //-----------initialize cuSparse2 library
    hipsparseStatus_t cuSparseStat;
    hipsparseHandle_t cuSparseHandle = 0;

    cuSparseStat = hipsparseCreate(&cuSparseHandle);
    if (cuSparseStat != HIPSPARSE_STATUS_SUCCESS) {
        printf("CUSPARSE2 Library initialization failed!");
        return 1;
    }

    //create matrix descriptior A and M in CSR
    hipsparseSpMatDescr_t matA;
    hipsparseCreateCsr(&matA, n, n, devA->nonzeroes, devA->mPtr, devA->mIndex, devA->mData,
                      HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);

    hipsparseSpMatDescr_t matM;
    hipsparseCreateCsr(&matM, n, n, devM->nonzeroes, devM->mPtr, devM->mIndex, devM->mData,
                      HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);

    //create Dense vector
    hipsparseDnVecDescr_t vecX, vecY, vecR, vecP, vecAY, vecS, vecZ, vecAZ, vecB;
    hipsparseCreateDnVec(&vecX, n, dev_x, HIP_R_64F);
    hipsparseCreateDnVec(&vecY, n, dev_y, HIP_R_64F);
    hipsparseCreateDnVec(&vecR, n, dev_r, HIP_R_64F);
    hipsparseCreateDnVec(&vecP, n, dev_p,HIP_R_64F);
    hipsparseCreateDnVec(&vecAY, n, dev_ay, HIP_R_64F);
    hipsparseCreateDnVec(&vecS, n, dev_s, HIP_R_64F);
    hipsparseCreateDnVec(&vecZ, n, dev_z, HIP_R_64F);
    hipsparseCreateDnVec(&vecAZ, n, dev_az, HIP_R_64F);
    hipsparseCreateDnVec(&vecB, n, dev_b, HIP_R_64F);

    //compute buffersize
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    float alpha2 = 1.0f, beta2 = 0.0f;
    hipsparseSpMV_bufferSize(cuSparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha2, matA, vecX,
                            &beta2, vecY, HIP_R_64F, HIPSPARSE_MV_ALG_DEFAULT, &bufferSize);
    hipMalloc(&dBuffer, bufferSize);

    //--------------r = b - Ax
    d_alpha = 1.0;
    d_beta = 0.0;
    // hipsparseDcsrmv( cuSparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, n, n, devA->nonzeroes, &d_alpha,
    //                 cusparseDescrA, devA->mData, devA->mPtr, devA->mIndex, dev_x, &d_beta, dev_r );

    hipsparseSpMV(cuSparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &d_alpha, matA, vecX, &d_beta, vecR,
                HIP_R_64F, HIPSPARSE_MV_ALG_DEFAULT, dBuffer);// r = Ax

    d_alpha = -1.0;
    hipblasDscal(cuBlasHandle, n, &d_alpha, dev_r, 1);// r = -Ax
    d_alpha = 1.0;
    hipblasDaxpy(cuBlasHandle, n, &d_alpha, dev_b, 1, dev_r, 1);// r = b - Ax

    //----------- r* = r
    hipblasDcopy(cuBlasHandle, n, dev_r, 1, dev_rstar, 1);

    //----------- p = r
    hipblasDcopy(cuBlasHandle, n, dev_r, 1, dev_p, 1);

    //----------- err0 = ||r||_2
    hipblasDnrm2(cuBlasHandle, n, dev_r, 1, &err0);

    //----------- temp1=(r*, r_j)
    hipblasDdot(cuBlasHandle, n, dev_rstar, 1, dev_r, 1, &temp1);

    int iter_times = 1;

    if (MAX_ITER <= 0) MAX_ITER = n;

    //printf( "iter_times = %d, err0 = %28.16f\n", iter_times, err0 ) ;

    while (iter_times < MAX_ITER) {
        //while( iter_times < 3 ){
        //-------y_j = M^-1*p_j, a_j = (r*, r_j)/(r*, Ay_j)
        //******y_j=M^-1*p_j
        d_alpha = 1.0;
        d_beta = 0.0;
        // hipsparseDcsrmv( cuSparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, n, n, devM->nonzeroes, &d_alpha,
        //         cusparseDescrM, devM->mData, devM->mPtr, devM->mIndex, dev_p, &d_beta, dev_y );

        hipsparseSpMV(cuSparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &d_alpha, matM, vecP, &d_beta, vecY,
            HIP_R_64F, HIPSPARSE_MV_ALG_DEFAULT, dBuffer);// y = Mp

        //******temp=(r*, r_j)
        temp = temp1;

        //******//ay = Ay_j
        d_alpha = 1.0;
        d_beta = 0.0;
        // hipsparseDcsrmv( cuSparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, n, n, devA->nonzeroes, &d_alpha,
        //         cusparseDescrA, devA->mData, devA->mPtr, devA->mIndex, dev_y, &d_beta, dev_ay );

        hipsparseSpMV(cuSparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &d_alpha, matA, vecY, &d_beta, vecAY,
            HIP_R_64F, HIPSPARSE_MV_ALG_DEFAULT, dBuffer);// ay = Ay

        //******temp2 = (r*, Ay_j)
        hipblasDdot(cuBlasHandle, n, dev_rstar, 1, dev_ay, 1, &temp2);

        //******alpha = temp/temp2
        alpha = temp / temp2;

        //------s_j = r_j - a_j*Ay_j
        hipblasDcopy(cuBlasHandle, n, dev_r, 1, dev_s, 1);
        d_alpha = -1.0 * alpha;
        hipblasDaxpy(cuBlasHandle, n, &d_alpha, dev_ay, 1, dev_s, 1);

        //------z_j= M^-1*s_j, w_j = (Az_j, s_j)/(Az_j, Az_j)
        //******z_j=M^-1*s_j
        d_alpha = 1.0;
        d_beta = 0.0;
        // hipsparseDcsrmv( cuSparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, n, n, devM->nonzeroes, &d_alpha,
        //         cusparseDescrM, devM->mData, devM->mPtr, devM->mIndex, dev_s, &d_beta, dev_z );

        hipsparseSpMV(cuSparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &d_alpha, matM, vecS, &d_beta, vecZ,
            HIP_R_64F, HIPSPARSE_MV_ALG_DEFAULT, dBuffer);// z = Ms

        //******az= Az_j
        d_alpha = 1.0;
        d_beta = 0.0;
        // hipsparseDcsrmv( cuSparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, n, n, devA->nonzeroes, &d_alpha,
        //         cusparseDescrA, devA->mData, devA->mPtr, devA->mIndex, dev_z, &d_beta, dev_az );

        hipsparseSpMV(cuSparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &d_alpha, matA, vecZ, &d_beta, vecAZ,
            HIP_R_64F, HIPSPARSE_MV_ALG_DEFAULT, dBuffer);// az = Az

        //******temp1=(Az_j, s_j)
        hipblasDdot(cuBlasHandle, n, dev_az, 1, dev_s, 1, &temp1);

        //******temp2=(Az_j, Az_j)
        hipblasDdot(cuBlasHandle, n, dev_az, 1, dev_az, 1, &temp2);

        //******w = temp1/temp2
        w = temp1 / temp2;

        //------x_j+1 = x_j + a_j*y_j + w_j * z_j ;
        d_alpha = alpha;
        hipblasDaxpy(cuBlasHandle, n, &d_alpha, dev_y, 1, dev_x, 1);
        d_alpha = w;
        hipblasDaxpy(cuBlasHandle, n, &d_alpha, dev_z, 1, dev_x, 1);

        //------r_j+1 = s_j - w_j * Az_j
        hipblasDcopy(cuBlasHandle, n, dev_s, 1, dev_r, 1);
        d_alpha = -1.0 * w;
        hipblasDaxpy(cuBlasHandle, n, &d_alpha, dev_az, 1, dev_r, 1);

        //-----------beta_j = (r*, r_j+1)/(r*, r_j) * a_j/w_j
        hipblasDdot(cuBlasHandle, n, dev_rstar, 1, dev_r, 1, &temp1);
        beta = (temp1 / temp) * (alpha / w);

        //----------p_j+1 = r_j+1 + beta_j *(p_j - w_j * Ay_j)
        d_alpha = beta;
        hipblasDscal(cuBlasHandle, n, &d_alpha, dev_p, 1);
        d_alpha = 1.0;
        hipblasDaxpy(cuBlasHandle, n, &d_alpha, dev_r, 1, dev_p, 1);
        d_alpha = -1.0 * beta * w;
        hipblasDaxpy(cuBlasHandle, n, &d_alpha, dev_ay, 1, dev_p, 1);

        //----------err = ||r_j+1||_2
        hipblasDnrm2(cuBlasHandle, n, dev_r, 1, &err);

        //printf( "iter_times = %d, err/err0=%18.14f \n", iter_times, err/err0 ) ;

        if ((err) < TOL) break;

        iter_times++;
    } //end while

    //printf("-----------------End the preconditioned BICGSTAB on CUBLAS---------------------\n") ;
    printf("-----------------The preconditioned BICGSTAB on CUBLAS stops---------------------\n");

    d_alpha = 1.0;
    d_beta = 0.0;
    hipsparseSpMV(cuSparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &d_alpha, matA, vecX, &d_beta, vecB,
        HIP_R_64F, HIPSPARSE_MV_ALG_DEFAULT, dBuffer);// b = Ax
    double *b = (double *) malloc(sizeof(double) * n);
    hipMemcpy(b, dev_b, n * sizeof(double), hipMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("b[%d] = %f\n", i, b[i]);
    }

    hipFree(dev_r);
    hipFree(dev_rstar);
    hipFree(dev_p);
    hipFree(dev_y);
    hipFree(dev_ay);
    hipFree(dev_s);
    hipFree(dev_z);
    hipFree(dev_az);

    hipsparseDestroy(cuSparseHandle);
    hipsparseDestroySpMat(matA);
    hipsparseDestroySpMat(matM);
    hipsparseDestroyDnVec(vecX);
    hipsparseDestroyDnVec(vecY);
    hipsparseDestroyDnVec(vecR);
    hipsparseDestroyDnVec(vecP);
    hipsparseDestroyDnVec(vecAY);
    hipsparseDestroyDnVec(vecS);
    hipsparseDestroyDnVec(vecZ);
    hipsparseDestroyDnVec(vecAZ);
    hipsparseDestroyDnVec(vecB);

    //printf("-------------+++++++++++++++++++++++++++++++++++++++++++++++++------------\n");
    printf("iter_times = %d, err/err0=%18.14f \n", iter_times, err);

    return 0;
}