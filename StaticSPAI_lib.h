#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <memory.h>
#include <string> 
#include <vector>
#include "dataType.h"
#include "StaticSPAI_Func.h"

using namespace std;



float StaticSPAI_v1(CSC_Matrix *devA, CSC_Matrix *devM){
    
    /*+++++++++++++++++Pre-GSPAI-Adaptive++++++++++++++++++++++++*/
    printf("Pre-GSPAI is processing..........................\n");
    cudaEvent_t start, stop ;
    float elapsedTime ;
    float preTime = 0.0;


    printf("-------------------Compute n2max\n");
    cudaEventCreate( &start ) ;
    cudaEventCreate( &stop  ) ;
    cudaEventRecord( start, 0 ) ;
    
    const int threadsPerBlock = 256;
    int blocksPerGrid = 15 * 8 * 4;
    
    int *n2 = (int*)malloc(sizeof(int) * blocksPerGrid);
    int *dev_n2;
    cudaMalloc((void**)&dev_n2, sizeof(int) * blocksPerGrid) ;
    
    cuComputeN2MAXwithSparityofA_SSPAIv1<threadsPerBlock><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->n, dev_n2);
    
    cudaMemcpyAsync( n2, dev_n2, blocksPerGrid * sizeof( int ), cudaMemcpyDeviceToHost, 0 ) ;
    int n2max = 0;
    for(int i = 0; i < blocksPerGrid; i++){
    	if(n2max < n2[i]) n2max = n2[i];
    }
    
    cudaFree(dev_n2);
    free(n2);
	  
    //int n2max = computeN2MAX(CSC_A);
    
    cudaEventRecord( stop, 0 ) ;
    cudaEventSynchronize( stop ) ;
    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
  
    printf( "Time = : %8.4f ms \n ", elapsedTime ) ;
    preTime += elapsedTime;

    cudaEventDestroy( start ) ;
    cudaEventDestroy( stop  ) ;
    
    printf("n2max=%d\n", n2max);
   
    printf("-------------------Compute n1max\n");
    cudaEventCreate( &start ) ;
    cudaEventCreate( &stop  ) ;
    cudaEventRecord( start, 0 ) ;
    
    int WarpSize;
    
    if(n2max <= 2){
      WarpSize = 2;
    }else if(n2max >2 && n2max <= 4){
    	WarpSize = 4;
    }else if(n2max >4 && n2max <= 8){
    	WarpSize = 8;
    }else if(n2max >8 && n2max <= 16){
    	WarpSize = 16;
    }else{
    	WarpSize = 32;
    } 
    
    blocksPerGrid = (devM->n - 1)/(threadsPerBlock/WarpSize) + 1;
    
    
    int *dev_n1;
    cudaMalloc((void**)&dev_n1, sizeof(int) * devM->nCol) ;
    int *dev_tI;
    int ssize = n2max * 5 ;
    cudaMalloc((void**)&dev_tI, sizeof(int) * devM->nCol * ssize) ;
    
    if(n2max <= 2){
    	cuComputeN1withSparityofA_SSPAIv1<2,256/2,256><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >2 && n2max <= 4){
    	cuComputeN1withSparityofA_SSPAIv1<4,256/4,256><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >4 && n2max <= 8){
    	cuComputeN1withSparityofA_SSPAIv1<8,256/8,256><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >8 && n2max <= 16){
    	cuComputeN1withSparityofA_SSPAIv1<16,256/16,256><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >16 && n2max <= 32){
    	cuComputeN1withSparityofA_SSPAIv1<32,256/32,256><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >32 && n2max <= 64){
    	cuComputeN1withSparityofA_SSPAIv1<32,256/32,512><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >64 && n2max <= 128){
    	cuComputeN1withSparityofA_SSPAIv1<32,256/32,1024><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >128 && n2max <= 256){
    	cuComputeN1withSparityofA_SSPAIv1<32,256/32,2048><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >256 && n2max <= 512){
    	cuComputeN1withSparityofA_SSPAIv1<32,256/32,4096><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >512 && n2max <= 1024){
      // printf("enter this n2max branch. \n");
	  cuComputeN1withSparityofA_SSPAIv1<32,256/32,8192><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else{
		cuComputeN1withSparityofA2_SSPAIv1<32,256/32><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    	// printf("Sorry, exceed the maximum shared memory in computing n1max\n");
      // exit(0);
    }
    
    cudaFree(dev_tI);

    
    blocksPerGrid = 15 * 8 * 4;
    int *n1 = (int*)malloc(sizeof(int) * blocksPerGrid);
    int *dev_n1max;
    cudaMalloc((void**)&dev_n1max, sizeof(int) * blocksPerGrid) ;
    
    cuComputeN1MAX_SSPAIv1<threadsPerBlock><< <blocksPerGrid, threadsPerBlock>> >(dev_n1, devM->n, dev_n1max);
    
    cudaMemcpyAsync( n1, dev_n1max, blocksPerGrid * sizeof( int ), cudaMemcpyDeviceToHost, 0 ) ;
    
    int n1max = 0;
    for(int i = 0; i < blocksPerGrid; i++){
    	if(n1max < n1[i]) n1max = n1[i];
    }
    
    cudaFree(dev_n1);
    cudaFree(dev_n1max);
    free(n1); 

    
    cudaEventRecord( stop, 0 ) ;
    cudaEventSynchronize( stop ) ;
    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
  
    printf( "Time = : %8.4f ms \n ", elapsedTime ) ;
    preTime += elapsedTime;

    cudaEventDestroy( start ) ;
    cudaEventDestroy( stop  ) ;
    
    printf("n1max=%d\n", n1max);


    // exit(0);

    
    cudaEventCreate( &start ) ;
    cudaEventCreate( &stop  ) ;
    cudaEventRecord( start, 0 ) ;
    
    if(n2max <= 2){
      WarpSize = 2;
    }else if(n2max >2 && n2max <= 4){
    	WarpSize = 4;
    }else if(n2max >4 && n2max <= 8){
    	WarpSize = 8;
    }else if(n2max >8 && n2max <= 16){
    	WarpSize = 16;
    }else if(n2max >16 && n2max <= 32){
    	WarpSize = 32;
    }else if(n2max >32 && n2max <= 64){
    	WarpSize = 64;
    }else if(n2max >64 && n2max <= 128){
    	WarpSize = 128;
    }else {
    	WarpSize = 256;
    }
    
    blocksPerGrid = (devM->nCol-1)/(256/WarpSize) + 1;
    
    cudaEventRecord( stop, 0 ) ;
    cudaEventSynchronize( stop ) ;
    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
  
    printf( "Time = : %8.4f ms \n ", elapsedTime ) ;
    preTime += elapsedTime;

    cudaEventDestroy( start ) ;
    cudaEventDestroy( stop  ) ;
    
    printf("blocksPerGrid = %d\n", blocksPerGrid);
    
    printf("\n");
    printf("preTime = : %12.4f ms \n ", preTime ) ;
    
    /*+++++++++++++++++Compute-GSPAI-Adaptive++++++++++++++++++++++++*/
    printf("Compute-GSPAI is processing......................\n");
    
    float computeTime = 0.0;
    
    /*********************************************************************************
    **                                                                               *
    ** Judge whether the memroy is exceeded?                                         *
    **                                                                               *
    **********************************************************************************/
    
    int compareSize = 1024*1024*100*4;
    long realSize = (long)devM->nCol * n1max * n2max;
     
    int itertions = (realSize-1)/compareSize + 1;
    //int itertions = 4;
    
    printf("compareSize=%d, realSize=%ld, iterations =%d\n", compareSize, realSize, itertions);
    
    //exit(0);
    if( itertions != 1){
    	const int nCols = (devM->nCol-1)/itertions + 1;
    	
    	printf("------Allocate GPU global memroy\n");
    	
    	cudaEventCreate( &start ) ;
        cudaEventCreate( &stop  ) ;
        cudaEventRecord( start, 0 ) ;
		  
	    double *dev_tildeA, *dev_R;

	    int *dev_J, *dev_jPTR;
	    int *dev_I, *dev_iPTR;
	    int *dev_E;
	    double *dev_X;
	 
	    cudaMalloc((void**)&dev_J, sizeof(int) * nCols * n2max) ;
	    cudaMalloc((void**)&dev_jPTR, sizeof(int) * nCols) ;

	    cudaMalloc((void**)&dev_I, sizeof(int) * nCols * n1max) ;
	    cudaMalloc((void**)&dev_iPTR, sizeof(int) * nCols) ;
	    
	    cudaMalloc((void**)&dev_tildeA, sizeof(double) * nCols * n1max * n2max) ;
	    cudaMalloc((void**)&dev_R, sizeof(double) * nCols * n2max * n2max) ;
	    cudaMalloc((void**)&dev_X, sizeof(double) * nCols * n2max) ;
	    cudaMalloc((void**)&dev_E, sizeof(int) * nCols) ;

        cudaEventRecord( stop, 0 ) ;
	    cudaEventSynchronize( stop ) ;
	    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
	  
	    printf( "Time = : %8.4f ms \n ", elapsedTime ) ;
	    computeTime += elapsedTime;

	    cudaEventDestroy( start ) ;
	    cudaEventDestroy( stop  ) ;
	    
//	    int *dev_mTmpPtr, *dev_mTmpIndex;  
	    double *dev_mTmpData;
//	    
//	    cudaMalloc((void**)&dev_mTmpPtr, sizeof(int) * (devM->nCol + 1)) ;
//      cudaMalloc((void**)&dev_mTmpIndex, sizeof(int) * devM->nonzeroes ) ;
        cudaMalloc((void**)&dev_mTmpData, sizeof(double) * devM->nonzeroes) ;
	    
	    for(int iter = 0; iter < itertions; iter++){
	    	
	    	int sK = iter * nCols;
	    	int eGrid = nCols;
            // if( iter == (itertions - 1) ) eGrid = devM->nCol - (itertions - 1) * nCols;
            if( sK + eGrid > devM->nCol ) 
            {
                eGrid = devM->nCol - sK;
                iter = itertions;
            }
            printf("iter = %d, eGrid=%d, sK=%d\n", iter, eGrid, sK);
            printf("warpSize =%d\n", WarpSize);
            
            blocksPerGrid = (eGrid - 1)/(threadsPerBlock/WarpSize) + 1;
	    	
	    	printf("---------------------find jIndex\n");
		  
		    cudaEventCreate( &start ) ;
            cudaEventCreate( &stop  ) ;
            cudaEventRecord( start, 0 ) ;
            
            if(n2max <= 2){
		        computeJ_iter_SSPAIv1<2><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, eGrid, dev_J, dev_jPTR, n2max, sK);
		    }else if(n2max >2 && n2max <= 4){
		    	computeJ_iter_SSPAIv1<4><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, eGrid, dev_J, dev_jPTR, n2max, sK);
		    }else if(n2max >4 && n2max <= 8){
		    	computeJ_iter_SSPAIv1<8><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, eGrid, dev_J, dev_jPTR, n2max, sK);
		    }else if(n2max >8 && n2max <= 16){
		    	computeJ_iter_SSPAIv1<16><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, eGrid, dev_J, dev_jPTR, n2max, sK);
		    }else if(n2max >16 && n2max <= 32){
		    	computeJ_iter_SSPAIv1<32><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, eGrid, dev_J, dev_jPTR, n2max, sK);
		    }else if(n2max >32 && n2max <= 64){
		    	computeJ_iter_SSPAIv1<64><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, eGrid, dev_J, dev_jPTR, n2max, sK);
		    }else if(n2max >64 && n2max <= 128){
		    	computeJ_iter_SSPAIv1<128><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, eGrid, dev_J, dev_jPTR, n2max, sK);
		    }else{
		    	computeJ_iter_SSPAIv1<256><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, eGrid, dev_J, dev_jPTR, n2max, sK);
		    }
		    
		    
		    cudaEventRecord( stop, 0 ) ;
		    cudaEventSynchronize( stop ) ;
		    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
		  
		    printf( "Time = : %8.4f ms \n ", elapsedTime ) ;
		    computeTime += elapsedTime;

		    cudaEventDestroy( start ) ;
		    cudaEventDestroy( stop  ) ;
		    
		    printf("---------------------find iIndex\n");
	    
		    cudaEventCreate( &start ) ;
			  cudaEventCreate( &stop  ) ;
			  cudaEventRecord( start, 0 ) ;
			  
			  
            if(n2max <= 2){
		        computeI_iter_SSPAIv1<2,256/2><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
		    }else if(n2max >2 && n2max <= 4){
		    	computeI_iter_SSPAIv1<4,256/4><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
		    }else if(n2max >4 && n2max <= 8){
		    	computeI_iter_SSPAIv1<8,256/8><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
		    }else if(n2max >8 && n2max <= 16){
		    	computeI_iter_SSPAIv1<16,256/16><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
		    }else if(n2max >16 && n2max <= 32){
		    	computeI_iter_SSPAIv1<32,256/32><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
		    }else if(n2max >32 && n2max <= 64){
		    	computeI_iter_SSPAIv1<64,256/64><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
		    }else if(n2max >64 && n2max <= 128){
		    	computeI_iter_SSPAIv1<128,256/128><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
		    }else{
		    	computeI_iter_SSPAIv1<256,256/256><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
		    }

		    //int blocksPerGrid1 = (eGrid - 1)/(threadsPerBlock/32) + 1;
		    //computeI1_SSPAIv10<32,256/32><< <blocksPerGrid1, threadsPerBlock>> >(dev_aPtr, dev_aIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
		    
		    cudaEventRecord( stop, 0 ) ;
		    cudaEventSynchronize( stop ) ;
		    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
		  
		    printf( "Time = : %8.4f ms \n ", elapsedTime ) ;
		    computeTime += elapsedTime;

		    cudaEventDestroy( start ) ;
		    cudaEventDestroy( stop  ) ;
		    
		    printf("-----------------find tilde of A\n");
	    
		    cudaEventCreate( &start ) ;
            cudaEventCreate( &stop  ) ;
            cudaEventRecord( start, 0 ) ;
			  
		    if(n2max <= 2){
		        ComputeTildeACSR_iter_SSPAIv1<2><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
		        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
		    }else if(n2max >2 && n2max <= 4){
		    	ComputeTildeACSR_iter_SSPAIv1<4><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
		        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
		    }else if(n2max >4 && n2max <= 8){
		    	ComputeTildeACSR_iter_SSPAIv1<8><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
		        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
		    }else if(n2max >8 && n2max <= 16){
		    	ComputeTildeACSR_iter_SSPAIv1<16><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
		        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
		    }else if(n2max >16 && n2max <= 32){
		    	ComputeTildeACSR_iter_SSPAIv1<32><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
		        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
		    }else if(n2max >32 && n2max <= 64){
		    	ComputeTildeACSR_iter_SSPAIv1<64><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
		        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
		    }else if(n2max >64 && n2max <= 128){
		    	ComputeTildeACSR_iter_SSPAIv1<128><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
		        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
		    }else{
		    	ComputeTildeACSR_iter_SSPAIv1<256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
		        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
		    }
		    
		 
		    cudaEventRecord( stop, 0 ) ;
		    cudaEventSynchronize( stop ) ;
		    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
		  
		    printf( "Time = : %8.4f ms \n ", elapsedTime ) ;
		    computeTime += elapsedTime;

		    cudaEventDestroy( start ) ;
		    cudaEventDestroy( stop  ) ;
		    
		    printf("++++find QR\n");
	    
		    cudaEventCreate( &start ) ;
            cudaEventCreate( &stop  ) ;
            cudaEventRecord( start, 0 ) ;
			  
		    if(n2max <= 2){
		    	QR_RShared_iter_SSPAIv1<2, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
		    }else if(n2max >2 && n2max <= 4){
		    	QR_RShared_iter_SSPAIv1<4, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
		    }else if(n2max >4 && n2max <= 8){
		    	QR_RShared_iter_SSPAIv1<8, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
		    }else if(n2max >8 && n2max <= 16){
		    	QR_RShared_iter_SSPAIv1<16, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
		    }else if(n2max >16 && n2max <= 32){
		    	QR_RShared_iter_SSPAIv1<32, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
		    }else if(n2max >32 && n2max <= 64){
		    	QR_RShared_iter_SSPAIv1<64, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
		    }else if(n2max >64 && n2max <= 128){
		    	QR_RShared_iter_SSPAIv1<128, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
		    }else{
		    	if(n2max >128 && n2max <= 256){
		    		QR_RShared_iter_SSPAIv1<256, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         	dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
		      }else if(n2max >256 && n2max <= 512){
					QR_RShared_iter_SSPAIv1<256, 512><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         	dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
		      }else if(n2max >512 && n2max <= 1024){
					QR_RShared_iter_SSPAIv1<256, 1024><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         	dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
		      }else if(n2max >1024 && n2max <= 2048){
					QR_RShared_iter_SSPAIv1<256, 2048><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         	dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
		      }else{
		      	printf("Sorry, exceed the maximum shared memory in the QR decomposition\n");
		        exit(0);
		      }
		    }
			  
            cudaEventRecord( stop, 0 ) ;
		    cudaEventSynchronize( stop ) ;
		    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
		  
		    printf( "Time = : %8.4f ms \n ", elapsedTime ) ;
		    computeTime += elapsedTime;

		    cudaEventDestroy( start ) ;
		    cudaEventDestroy( stop  ) ;
		    
		    printf("--------------Compute tilde of E\n");
		    
		    cudaEventCreate( &start ) ;
            cudaEventCreate( &stop  ) ;
            cudaEventRecord( start, 0 ) ;
			  
            if(n2max <= 2){
		        ComputeTildeE_iter_SSPAIv1<2><< <blocksPerGrid, threadsPerBlock>> >(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);
		    }else if(n2max >2 && n2max <= 4){
		    	ComputeTildeE_iter_SSPAIv1<4><< <blocksPerGrid, threadsPerBlock>> >(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);
		    }else if(n2max >4 && n2max <= 8){
		    	ComputeTildeE_iter_SSPAIv1<8><< <blocksPerGrid, threadsPerBlock>> >(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);
		    }else if(n2max >8 && n2max <= 16){
		    	ComputeTildeE_iter_SSPAIv1<16><< <blocksPerGrid, threadsPerBlock>> >(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);
		    }else if(n2max >16 && n2max <= 32){
		    	ComputeTildeE_iter_SSPAIv1<32><< <blocksPerGrid, threadsPerBlock>> >(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);
		    }else if(n2max >32 && n2max <= 64){
		    	ComputeTildeE_iter_SSPAIv1<64><< <blocksPerGrid, threadsPerBlock>> >(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);
		    }else if(n2max >64 && n2max <= 128){
		    	ComputeTildeE_iter_SSPAIv1<128><< <blocksPerGrid, threadsPerBlock>> >(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);
		    }else{
		    	ComputeTildeE_iter_SSPAIv1<256><< <blocksPerGrid, threadsPerBlock>> >(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);
		    }

			cudaEventRecord( stop, 0 ) ;
		    cudaEventSynchronize( stop ) ;
		    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
		  
		    printf( "Time = : %8.4f ms \n ", elapsedTime ) ;
		    computeTime += elapsedTime;

		    cudaEventDestroy( start ) ;
		    cudaEventDestroy( stop  ) ;
		    
		    printf("-------------------------Solve X\n");
		    
		    cudaEventCreate( &start ) ;
            cudaEventCreate( &stop  ) ;
            cudaEventRecord( start, 0 ) ;
			 
		    if(n2max <= 2){
				Sol_iter_SSPAIv1<2><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, dev_R,  dev_X,
			                  dev_E, dev_jPTR, n1max, n2max, eGrid);
		    }else if(n2max >2 && n2max <= 4){
		    	Sol_iter_SSPAIv1<4><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, dev_R,  dev_X,
			                  dev_E, dev_jPTR, n1max, n2max, eGrid);
		    }else if(n2max >4 && n2max <= 8){
		    	Sol_iter_SSPAIv1<8><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, dev_R,  dev_X,
			                  dev_E, dev_jPTR, n1max, n2max, eGrid);
		    }else if(n2max >8 && n2max <= 16){
		    	Sol_iter_SSPAIv1<16><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, dev_R,  dev_X,
			                  dev_E, dev_jPTR, n1max, n2max, eGrid);
		    }else if(n2max >16 && n2max <= 32){
		    	Sol_iter_SSPAIv1<32><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, dev_R,  dev_X,
			                  dev_E, dev_jPTR, n1max, n2max, eGrid);
		    }else if(n2max >32 && n2max <= 64){
		    	Sol_iter_SSPAIv1<64><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, dev_R,  dev_X,
			                  dev_E, dev_jPTR, n1max, n2max, eGrid);
		    }else if(n2max >64 && n2max <= 128){
		    	Sol_iter_SSPAIv1<128><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, dev_R,  dev_X,
			                  dev_E, dev_jPTR, n1max, n2max, eGrid);
		    }else{
		    	Sol_iter_SSPAIv1<256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, dev_R,  dev_X,
			                  dev_E, dev_jPTR, n1max, n2max, eGrid);
		    }
		    
			  
			cudaEventRecord( stop, 0 ) ;
		    cudaEventSynchronize( stop ) ;
		    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
		  
		    printf( "Time = : %8.4f ms \n ", elapsedTime ) ;
		    computeTime += elapsedTime;

		    cudaEventDestroy( start ) ;
		    cudaEventDestroy( stop  ) ;
		    
		    printf("\n");
		    printf("computeTime = : %12.4f ms \n", computeTime );
		    
		    /*---------Store arrays templately------------------*/
	      
	      //cudaMemcpyAsync( dev_mTmpPtr + sK, dev_jPTR, eGrid * sizeof( int ), cudaMemcpyDeviceToDevice, 0 ) ;
            int WarpSize1;
            if(n2max <= 2){
                WarpSize1 = 2;
		    }else if(n2max >2 && n2max <= 4){
		    	WarpSize1 = 4;
		    }else if(n2max >4 && n2max <= 8){
		    	WarpSize1 = 8;
		    }else if(n2max >8 && n2max <= 16){
		    	WarpSize1 = 16;
		    }else{
		    	WarpSize1 = 32;
		    } 
		    
		    blocksPerGrid = (eGrid - 1)/(threadsPerBlock/WarpSize1) + 1; 
		    
		    if(n2max <= 2){
		    	modifyData_iter_SSPAIv1<2><< <blocksPerGrid, threadsPerBlock>> >(dev_mTmpData, 
		                      devM->mPtr, dev_X, dev_jPTR, n2max, eGrid, sK);
		    }else if(n2max >2 && n2max <= 4){
		    	modifyData_iter_SSPAIv1<4><< <blocksPerGrid, threadsPerBlock>> >(dev_mTmpData, 
		                      devM->mPtr, dev_X, dev_jPTR, n2max, eGrid, sK);
		    }else if(n2max >4 && n2max <= 8){
		    	modifyData_iter_SSPAIv1<8><< <blocksPerGrid, threadsPerBlock>> >(dev_mTmpData, 
		                      devM->mPtr, dev_X, dev_jPTR, n2max, eGrid, sK);
		    }else if(n2max >8 && n2max <= 16){
		    	modifyData_iter_SSPAIv1<16><< <blocksPerGrid, threadsPerBlock>> >(dev_mTmpData, 
		                      devM->mPtr, dev_X, dev_jPTR, n2max, eGrid, sK);
		    }else{
		    	modifyData_iter_SSPAIv1<32><< <blocksPerGrid, threadsPerBlock>> >(dev_mTmpData, 
		                      devM->mPtr, dev_X, dev_jPTR, n2max, eGrid, sK);
		    }
	    }
	    //exit(0);
	    /*+++++++++++++++++Post-GSPAI-Adaptive++++++++++++++++++++++++*/
	    
	    printf("Post-GSPAI is processing.........................\n");
	    cudaEventCreate( &start ) ;
        cudaEventCreate( &stop  ) ;
        cudaEventRecord( start, 0 ) ;

        cudaMemcpy( devM->mData, dev_mTmpData, sizeof( double ) * devM->nonzeroes, cudaMemcpyDeviceToDevice);
	    
	    cudaEventRecord( stop, 0 ) ;
	    cudaEventSynchronize( stop ) ;
	    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
	  
	    printf("\n");
	    printf( "postTime = : %12.4f ms \n ", elapsedTime ) ;
	    
	    cudaEventDestroy( start ) ;
	    cudaEventDestroy( stop  ) ;
	    
//      exit(0);
        printf("*************************************************\n");
	    printf("TotalPreTime = : %12.4f ms \n", preTime );
	    printf("TotalComputeTime = : %12.4f ms \n", computeTime );
	    printf( "TotalPostTime = : %12.4f ms \n ", elapsedTime ) ;
	    
	    printf( "TotalTime = : %12.4f ms \n ", preTime + computeTime + elapsedTime ) ;
        printf("\n");
	    
	    printf("Transfering CSC_M to CSR_M is processing.........\n");


	    cudaFree(dev_mTmpData);

	    cudaFree(dev_J);
	    cudaFree(dev_I);
	    cudaFree(dev_jPTR);
	    cudaFree(dev_iPTR);
	    
	    cudaFree(dev_tildeA);
	    cudaFree(dev_R);
	    cudaFree(dev_E);
	    cudaFree(dev_X);
	    
    }else{
	    printf("------Allocate GPU global memroy\n");
	    
	    cudaEventCreate( &start ) ;
        cudaEventCreate( &stop  ) ;
        cudaEventRecord( start, 0 ) ;
		  
	    double *dev_tildeA, *dev_R;

	    int *dev_J, *dev_jPTR;
	    int *dev_I, *dev_iPTR;
	    int *dev_E;
	    double *dev_X;
	 
	    cudaMalloc((void**)&dev_J, sizeof(int) * devM->nCol * n2max) ;
	    cudaMalloc((void**)&dev_jPTR, sizeof(int) * devM->nCol) ;

	    cudaMalloc((void**)&dev_I, sizeof(int) * devM->nCol * n1max) ;
	    cudaMalloc((void**)&dev_iPTR, sizeof(int) * devM->nCol) ;
	    
	    cudaMalloc((void**)&dev_tildeA, sizeof(double) * devM->nCol * n1max * n2max) ;
	    cudaMalloc((void**)&dev_R, sizeof(double) * devM->nCol * n2max * n2max) ;
	    cudaMalloc((void**)&dev_X, sizeof(double) * devM->nCol * n2max) ;
	    cudaMalloc((void**)&dev_E, sizeof(int) * devM->nCol) ;

		cudaEventRecord( stop, 0 ) ;
	    cudaEventSynchronize( stop ) ;
	    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
	  
	    printf( "Time = : %8.4f ms \n ", elapsedTime ) ;
	    computeTime += elapsedTime;

	    cudaEventDestroy( start ) ;
	    cudaEventDestroy( stop  ) ;
		  
		printf("---------------------find jIndex\n");
		  
	    cudaEventCreate( &start ) ;
        cudaEventCreate( &stop  ) ;
        cudaEventRecord( start, 0 ) ;
		if(n2max <= 2){
	        computeJ_SSPAIv1<2><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->nCol, dev_J, dev_jPTR, n2max);
	    }else if(n2max >2 && n2max <= 4){
	    	computeJ_SSPAIv1<4><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->nCol, dev_J, dev_jPTR, n2max);
	    }else if(n2max >4 && n2max <= 8){
	    	computeJ_SSPAIv1<8><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->nCol, dev_J, dev_jPTR, n2max);
	    }else if(n2max >8 && n2max <= 16){
	    	computeJ_SSPAIv1<16><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->nCol, dev_J, dev_jPTR, n2max);
	    }else if(n2max >16 && n2max <= 32){
	    	computeJ_SSPAIv1<32><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->nCol, dev_J, dev_jPTR, n2max);
	    }else if(n2max >32 && n2max <= 64){
	    	computeJ_SSPAIv1<64><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->nCol, dev_J, dev_jPTR, n2max);
	    }else if(n2max >64 && n2max <= 128){
	    	computeJ_SSPAIv1<128><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->nCol, dev_J, dev_jPTR, n2max);
	    }else{
	    	computeJ_SSPAIv1<256><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->nCol, dev_J, dev_jPTR, n2max);
	    }
	    
	    cudaEventRecord( stop, 0 ) ;
	    cudaEventSynchronize( stop ) ;
	    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
	  
	    printf( "Time = : %8.4f ms \n ", elapsedTime ) ;
	    computeTime += elapsedTime;

	    cudaEventDestroy( start ) ;
	    cudaEventDestroy( stop  ) ;

	    printf("---------------------find iIndex\n");
	    
	    cudaEventCreate( &start ) ;
        cudaEventCreate( &stop  ) ;
        cudaEventRecord( start, 0 ) ;

	    if(n2max <= 2){
	        computeI_SSPAIv1<2,256/2><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
	    }else if(n2max >2 && n2max <= 4){
	    	computeI_SSPAIv1<4,256/4><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
	    }else if(n2max >4 && n2max <= 8){
	    	computeI_SSPAIv1<8,256/8><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
	    }else if(n2max >8 && n2max <= 16){
	    	computeI_SSPAIv1<16,256/16><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
	    }else if(n2max >16 && n2max <= 32){
	    	computeI_SSPAIv1<32,256/32><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
	    }else if(n2max >32 && n2max <= 64){
	    	computeI_SSPAIv1<64,256/64><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
	    }else if(n2max >64 && n2max <= 128){
	    	computeI_SSPAIv1<128,256/128><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
	    }else{
	    	computeI_SSPAIv1<256,256/256><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
	    }
	    
	    cudaEventRecord( stop, 0 ) ;
	    cudaEventSynchronize( stop ) ;
	    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
	  
	    printf( "Time = : %8.4f ms \n ", elapsedTime ) ;
	    computeTime += elapsedTime;

	    cudaEventDestroy( start ) ;
	    cudaEventDestroy( stop  ) ;
	    
	    printf("-----------------find tilde of A\n");
	    
	    cudaEventCreate( &start ) ;
        cudaEventCreate( &stop  ) ;
        cudaEventRecord( start, 0 ) ;
	    if(n2max <= 2){
	        ComputeTildeACSR_SSPAIv1<2><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
	        devA->mIndex, devM->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
	    }else if(n2max >2 && n2max <= 4){
	    	ComputeTildeACSR_SSPAIv1<4><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
	        devA->mIndex, devM->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
	    }else if(n2max >4 && n2max <= 8){
	    	ComputeTildeACSR_SSPAIv1<8><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
	        devA->mIndex, devM->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
	    }else if(n2max >8 && n2max <= 16){
	    	ComputeTildeACSR_SSPAIv1<16><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
	        devA->mIndex, devM->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
	    }else if(n2max >16 && n2max <= 32){
	    	ComputeTildeACSR_SSPAIv1<32><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
	        devA->mIndex, devM->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
	    }else if(n2max >32 && n2max <= 64){
	    	ComputeTildeACSR_SSPAIv1<64><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
	        devA->mIndex, devM->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
	    }else if(n2max >64 && n2max <= 128){
	    	ComputeTildeACSR_SSPAIv1<128><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
	        devA->mIndex, devM->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
	    }else{
	    	ComputeTildeACSR_SSPAIv1<256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
	        devA->mIndex, devM->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
	    }
	 
	    cudaEventRecord( stop, 0 ) ;
	    cudaEventSynchronize( stop ) ;
	    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
	  
	    printf( "Time = : %8.4f ms \n ", elapsedTime ) ;
	    computeTime += elapsedTime;

	    cudaEventDestroy( start ) ;
	    cudaEventDestroy( stop  ) ;
	    
	    printf("-----------------------find QR\n");

	    cudaEventCreate( &start ) ;
        cudaEventCreate( &stop  ) ;
        cudaEventRecord( start, 0 ) ;
        
        if(n2max <= 2){
	    	QR_RShared_SSPAIv1<2, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
	         dev_R, dev_iPTR, dev_jPTR, n1max, n2max, devM->nCol);
	    }else if(n2max >2 && n2max <= 4){
	    	QR_RShared_SSPAIv1<4, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
	         dev_R, dev_iPTR, dev_jPTR, n1max, n2max, devM->nCol);
	    }else if(n2max >4 && n2max <= 8){
	    	QR_RShared_SSPAIv1<8, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
	         dev_R, dev_iPTR, dev_jPTR, n1max, n2max, devM->nCol);
	    }else if(n2max >8 && n2max <= 16){
	    	QR_RShared_SSPAIv1<16, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
	         dev_R, dev_iPTR, dev_jPTR, n1max, n2max, devM->nCol);
	    }else if(n2max >16 && n2max <= 32){
	    	QR_RShared_SSPAIv1<32, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
	         dev_R, dev_iPTR, dev_jPTR, n1max, n2max, devM->nCol);
	    }else if(n2max >32 && n2max <= 64){
	    	QR_RShared_SSPAIv1<64, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
	         dev_R, dev_iPTR, dev_jPTR, n1max, n2max, devM->nCol);
	    }else if(n2max >64 && n2max <= 128){
	    	QR_RShared_SSPAIv1<128, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
	         dev_R, dev_iPTR, dev_jPTR, n1max, n2max, devM->nCol);
	    }else{
	    	if(n2max >128 && n2max <= 256){
	    		QR_RShared_SSPAIv1<256, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
	         	dev_R, dev_iPTR, dev_jPTR, n1max, n2max, devM->nCol);
	      }else if(n2max >256 && n2max <= 512){
	    		QR_RShared_SSPAIv1<256, 512><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
	         	dev_R, dev_iPTR, dev_jPTR, n1max, n2max, devM->nCol);
	      }else if(n2max >512 && n2max <= 1024){
	    		QR_RShared_SSPAIv1<256, 1024><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
	         	dev_R, dev_iPTR, dev_jPTR, n1max, n2max, devM->nCol);
	      }else if(n2max >1024 && n2max <= 2048){
	    		QR_RShared_SSPAIv1<256, 2048><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
	         	dev_R, dev_iPTR, dev_jPTR, n1max, n2max, devM->nCol);
	      }else{
	      	printf("Sorry, exceed the maximum shared memory in the QR decomposition\n");
	        exit(0);
	      }
	    }
		  
        cudaEventRecord( stop, 0 ) ;
	    cudaEventSynchronize( stop ) ;
	    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
	  
	    printf( "Time = : %8.4f ms \n ", elapsedTime ) ;
	    computeTime += elapsedTime;

	    cudaEventDestroy( start ) ;
	    cudaEventDestroy( stop  ) ;
	    
	    printf("--------------Compute tilde of E\n");
	    
	    cudaEventCreate( &start ) ;
        cudaEventCreate( &stop  ) ;
        cudaEventRecord( start, 0 ) ;
		  
        if(n2max <= 2){
	        ComputeTildeE_SSPAIv1<2><< <blocksPerGrid, threadsPerBlock>> >(dev_E, dev_I, dev_iPTR, n1max, devM->nCol);
	    }else if(n2max >2 && n2max <= 4){
	    	ComputeTildeE_SSPAIv1<4><< <blocksPerGrid, threadsPerBlock>> >(dev_E, dev_I, dev_iPTR, n1max, devM->nCol);
	    }else if(n2max >4 && n2max <= 8){
	    	ComputeTildeE_SSPAIv1<8><< <blocksPerGrid, threadsPerBlock>> >(dev_E, dev_I, dev_iPTR, n1max, devM->nCol);
	    }else if(n2max >8 && n2max <= 16){
	    	ComputeTildeE_SSPAIv1<16><< <blocksPerGrid, threadsPerBlock>> >(dev_E, dev_I, dev_iPTR, n1max, devM->nCol);
	    }else if(n2max >16 && n2max <= 32){
	    	ComputeTildeE_SSPAIv1<32><< <blocksPerGrid, threadsPerBlock>> >(dev_E, dev_I, dev_iPTR, n1max, devM->nCol);
	    }else if(n2max >32 && n2max <= 64){
	    	ComputeTildeE_SSPAIv1<64><< <blocksPerGrid, threadsPerBlock>> >(dev_E, dev_I, dev_iPTR, n1max, devM->nCol);
	    }else if(n2max >64 && n2max <= 128){
	    	ComputeTildeE_SSPAIv1<128><< <blocksPerGrid, threadsPerBlock>> >(dev_E, dev_I, dev_iPTR, n1max, devM->nCol);
	    }else{
	    	ComputeTildeE_SSPAIv1<256><< <blocksPerGrid, threadsPerBlock>> >(dev_E, dev_I, dev_iPTR, n1max, devM->nCol);
	    }
		  
        cudaEventRecord( stop, 0 ) ;
	    cudaEventSynchronize( stop ) ;
	    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
	  
	    printf( "Time = : %8.4f ms \n ", elapsedTime ) ;
	    computeTime += elapsedTime;

	    cudaEventDestroy( start ) ;
	    cudaEventDestroy( stop  ) ;
	    
	    printf("-------------------------Solve X\n");
	    
	    cudaEventCreate( &start ) ;
        cudaEventCreate( &stop  ) ;
        cudaEventRecord( start, 0 ) ;
        
        if(n2max <= 2){
	        Sol_SSPAIv1<2><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, dev_R,  dev_X,
		                  dev_E, dev_jPTR, n1max, n2max, devM->nCol);
	    }else if(n2max >2 && n2max <= 4){
	    	Sol_SSPAIv1<4><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, dev_R,  dev_X,
		                  dev_E, dev_jPTR, n1max, n2max, devM->nCol);
	    }else if(n2max >4 && n2max <= 8){
	    	Sol_SSPAIv1<8><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, dev_R,  dev_X,
		                  dev_E, dev_jPTR, n1max, n2max, devM->nCol);
	    }else if(n2max >8 && n2max <= 16){
	    	Sol_SSPAIv1<16><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, dev_R,  dev_X,
		                  dev_E, dev_jPTR, n1max, n2max, devM->nCol);
	    }else if(n2max >16 && n2max <= 32){
	    	Sol_SSPAIv1<32><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, dev_R,  dev_X,
		                  dev_E, dev_jPTR, n1max, n2max, devM->nCol);
	    }else if(n2max >32 && n2max <= 64){
	    	Sol_SSPAIv1<64><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, dev_R,  dev_X,
		                  dev_E, dev_jPTR, n1max, n2max, devM->nCol);
	    }else if(n2max >64 && n2max <= 128){
	    	Sol_SSPAIv1<128><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, dev_R,  dev_X,
		                  dev_E, dev_jPTR, n1max, n2max, devM->nCol);
	    }else{
	    	Sol_SSPAIv1<256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, dev_R,  dev_X,
		                  dev_E, dev_jPTR, n1max, n2max, devM->nCol);
	    }
		  
        cudaEventRecord( stop, 0 ) ;
	    cudaEventSynchronize( stop ) ;
	    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
	  
	    printf( "Time = : %8.4f ms \n ", elapsedTime ) ;
	    computeTime += elapsedTime;

	    cudaEventDestroy( start ) ;
	    cudaEventDestroy( stop  ) ;
	    
	    printf("\n");
	    printf("computeTime = : %12.4f ms \n", computeTime );
	   
	    /*+++++++++++++++++Post-GSPAI-Adaptive++++++++++++++++++++++++*/
	    
	    printf("Post-GSPAI is processing.........................\n");
	    cudaEventCreate( &start ) ;
        cudaEventCreate( &stop  ) ;
        cudaEventRecord( start, 0 ) ;
        	    
	    if(n2max <= 2){
	      WarpSize = 2;
	    }else if(n2max >2 && n2max <= 4){
	    	WarpSize = 4;
	    }else if(n2max >4 && n2max <= 8){
	    	WarpSize = 8;
	    }else if(n2max >8 && n2max <= 16){
	    	WarpSize = 16;
	    }else{
	    	WarpSize = 32;
	    } 
	    
	    blocksPerGrid = (devM->nCol - 1)/(threadsPerBlock/WarpSize) + 1; 
	    
	    if(n2max <= 2){
	    	modifyData_SSPAIv1<2><< <blocksPerGrid, threadsPerBlock>> >(devM->mData, 
	                      devM->mPtr, dev_X, dev_jPTR, n2max, devM->nCol);
	    }else if(n2max >2 && n2max <= 4){
	    	modifyData_SSPAIv1<4><< <blocksPerGrid, threadsPerBlock>> >(devM->mData, 
	                      devM->mPtr, dev_X, dev_jPTR, n2max, devM->nCol);
	    }else if(n2max >4 && n2max <= 8){
	    	modifyData_SSPAIv1<8><< <blocksPerGrid, threadsPerBlock>> >(devM->mData, 
	                      devM->mPtr, dev_X, dev_jPTR, n2max, devM->nCol);
	    }else if(n2max >8 && n2max <= 16){
	    	modifyData_SSPAIv1<16><< <blocksPerGrid, threadsPerBlock>> >(devM->mData, 
	                      devM->mPtr, dev_X, dev_jPTR, n2max, devM->nCol);
	    }else{
	    	modifyData_SSPAIv1<32><< <blocksPerGrid, threadsPerBlock>> >(devM->mData, 
	                      devM->mPtr, dev_X, dev_jPTR, n2max, devM->nCol);
	    }
	    
	    
	    cudaEventRecord( stop, 0 ) ;
	    cudaEventSynchronize( stop ) ;
	    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
	  
	    printf("\n");
	    printf( "postTime = : %12.4f ms \n ", elapsedTime ) ;
	    
	    cudaEventDestroy( start ) ;
	    cudaEventDestroy( stop  ) ;
	    
	    //exit(0);
	    
	    printf("******************The preconditioner time*************************\n");
	    printf("TotalPreTime = : %12.4f ms \n", preTime );
	    printf("TotalComputeTime = : %12.4f ms \n", computeTime );
	    printf( "TotalPostTime = : %12.4f ms \n ", elapsedTime ) ;
	    
	    printf( "TotalTime = : %12.4f ms \n ", preTime + computeTime + elapsedTime ) ;
	    printf("\n");
	    


	    cudaFree(dev_J);
	    cudaFree(dev_I);
	    cudaFree(dev_jPTR);
	    cudaFree(dev_iPTR);
	    
	    cudaFree(dev_tildeA);
	    cudaFree(dev_R);
	    cudaFree(dev_E);
	    cudaFree(dev_X);
	  }
	  
	  return preTime + computeTime + elapsedTime;
}

