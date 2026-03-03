#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <memory.h>
#include <string> 
#include <vector>
#include "dataType.h"
#include "SpMM.h"

using namespace std;


float SpMMWithoutSparsity(CSC_Matrix *devA, CSC_Matrix *devM, CSC_Matrix *devC){
    
    /*+++++++++++++++++Pre-SpMM+++++++++++++++++++++++++++++++*/
    printf("Pre-SpMM is processing..........................\n");
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
    
    cuComputeN2MAXwithSparityofA_SpMMv1<threadsPerBlock><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->n, dev_n2);
    
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
    	cuComputeN1withSparityofA_SpMMv1<2,256/2,256><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >2 && n2max <= 4){
    	cuComputeN1withSparityofA_SpMMv1<4,256/4,256><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >4 && n2max <= 8){
    	cuComputeN1withSparityofA_SpMMv1<8,256/8,256><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >8 && n2max <= 16){
    	cuComputeN1withSparityofA_SpMMv1<16,256/16,256><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >16 && n2max <= 32){
    	cuComputeN1withSparityofA_SpMMv1<32,256/32,256><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >32 && n2max <= 64){
    	cuComputeN1withSparityofA_SpMMv1<32,256/32,512><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >64 && n2max <= 128){
    	cuComputeN1withSparityofA_SpMMv1<32,256/32,1024><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >128 && n2max <= 256){
    	cuComputeN1withSparityofA_SpMMv1<32,256/32,2048><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >256 && n2max <= 512){
    	cuComputeN1withSparityofA_SpMMv1<32,256/32,4096><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >512 && n2max <= 1024){
      // printf("enter this n2max branch. \n");
	  cuComputeN1withSparityofA_SpMMv1<32,256/32,8192><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else{
		cuComputeN1withSparityofA2_SpMMv1<32,256/32><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    	// printf("Sorry, exceed the maximum shared memory in computing n1max\n");
      // exit(0);
    }
    
    cudaFree(dev_tI);

    
    blocksPerGrid = 15 * 8 * 4;
    int *n1 = (int*)malloc(sizeof(int) * blocksPerGrid);
    int *dev_n1max;
    cudaMalloc((void**)&dev_n1max, sizeof(int) * blocksPerGrid) ;
    
    cuComputeN1MAX_SpMMv1<threadsPerBlock><< <blocksPerGrid, threadsPerBlock>> >(dev_n1, devM->n, dev_n1max);
    
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
    printf("Compute-SpMM is processing......................\n");
    
    float computeTime = 0.0;
    
    /*********************************************************************************
    **                                                                               *
    ** Judge whether the memroy is exceeded?                                         *
    **                                                                               *
    **********************************************************************************/
    
    int compareSize = 1024*1024*100*4;
    long realSize = (long)devM->nCol * n1max * n2max;
     
    int itertions = (realSize-1)/compareSize + 1;
    //int itertions = 2;
    
    printf("compareSize=%d, realSize=%ld, iterations =%d\n", compareSize, realSize, itertions);
    
    int *dev_tempPTR, *dev_tempIndex;
    double *dev_tempData;
    cudaMalloc((void**)&dev_tempIndex, sizeof(int) * devM->nCol * n1max) ;
	  cudaMalloc((void**)&dev_tempPTR, sizeof(int) * devM->nCol) ;
	    
	  cudaMalloc((void**)&dev_tempData, sizeof(double) * devM->nCol * n1max) ;
    
    
    //exit(0);
    if( itertions != 1){
    	const int nCols = (devM->nCol-1)/itertions + 1;
    	
    	printf("------Allocate GPU global memroy\n");
    	
    	cudaEventCreate( &start ) ;
      cudaEventCreate( &stop  ) ;
      cudaEventRecord( start, 0 ) ;
		  
	    double *dev_tildeA;

	    int *dev_J, *dev_jPTR;
	    int *dev_I, *dev_iPTR;
	    double *dev_X, *dev_JV;
	 
	    cudaMalloc((void**)&dev_J, sizeof(int) * nCols * n2max) ;
	    cudaMalloc((void**)&dev_jPTR, sizeof(int) * nCols) ;
	    cudaMalloc((void**)&dev_JV, sizeof(double) * nCols * n2max) ;

	    cudaMalloc((void**)&dev_I, sizeof(int) * nCols * n1max) ;
	    cudaMalloc((void**)&dev_iPTR, sizeof(int) * nCols) ;
	    
	    cudaMalloc((void**)&dev_tildeA, sizeof(double) * nCols * n1max * n2max) ;
	    cudaMalloc((void**)&dev_X, sizeof(double) * nCols * n1max) ;

      cudaEventRecord( stop, 0 ) ;
	    cudaEventSynchronize( stop ) ;
	    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
	  
	    printf( "Time = : %8.4f ms \n ", elapsedTime ) ;
	    computeTime += elapsedTime;

	    cudaEventDestroy( start ) ;
	    cudaEventDestroy( stop  ) ;
	    
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
		        computeJ_iter_SpMMv1<2><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, eGrid, dev_J, dev_jPTR, dev_JV, n2max, sK);
		    }else if(n2max >2 && n2max <= 4){
		    	computeJ_iter_SpMMv1<4><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, eGrid, dev_J, dev_jPTR, dev_JV, n2max, sK);
		    }else if(n2max >4 && n2max <= 8){
		    	computeJ_iter_SpMMv1<8><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, eGrid, dev_J, dev_jPTR, dev_JV, n2max, sK);
		    }else if(n2max >8 && n2max <= 16){
		    	computeJ_iter_SpMMv1<16><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, eGrid, dev_J, dev_jPTR, dev_JV, n2max, sK);
		    }else if(n2max >16 && n2max <= 32){
		    	computeJ_iter_SpMMv1<32><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, eGrid, dev_J, dev_jPTR, dev_JV, n2max, sK);
		    }else if(n2max >32 && n2max <= 64){
		    	computeJ_iter_SpMMv1<64><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, eGrid, dev_J, dev_jPTR, dev_JV, n2max, sK);
		    }else if(n2max >64 && n2max <= 128){
		    	computeJ_iter_SpMMv1<128><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, eGrid, dev_J, dev_jPTR, dev_JV, n2max, sK);
		    }else{
		    	computeJ_iter_SpMMv1<256><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, eGrid, dev_J, dev_jPTR, dev_JV, n2max, sK);
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
		        computeI_iter_SpMMv1<2,256/2><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
		    }else if(n2max >2 && n2max <= 4){
		    	computeI_iter_SpMMv1<4,256/4><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
		    }else if(n2max >4 && n2max <= 8){
		    	computeI_iter_SpMMv1<8,256/8><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
		    }else if(n2max >8 && n2max <= 16){
		    	computeI_iter_SpMMv1<16,256/16><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
		    }else if(n2max >16 && n2max <= 32){
		    	computeI_iter_SpMMv1<32,256/32><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
		    }else if(n2max >32 && n2max <= 64){
		    	computeI_iter_SpMMv1<64,256/64><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
		    }else if(n2max >64 && n2max <= 128){
		    	computeI_iter_SpMMv1<128,256/128><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
		    }else{
		    	computeI_iter_SpMMv1<256,256/256><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
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
		        ComputeTildeACSR_iter_SpMMv1<2><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
		        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
		    }else if(n2max >2 && n2max <= 4){
		    	ComputeTildeACSR_iter_SpMMv1<4><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
		        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
		    }else if(n2max >4 && n2max <= 8){
		    	ComputeTildeACSR_iter_SpMMv1<8><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
		        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
		    }else if(n2max >8 && n2max <= 16){
		    	ComputeTildeACSR_iter_SpMMv1<16><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
		        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
		    }else if(n2max >16 && n2max <= 32){
		    	ComputeTildeACSR_iter_SpMMv1<32><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
		        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
		    }else if(n2max >32 && n2max <= 64){
		    	ComputeTildeACSR_iter_SpMMv1<64><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
		        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
		    }else if(n2max >64 && n2max <= 128){
		    	ComputeTildeACSR_iter_SpMMv1<128><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
		        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
		    }else{
		    	ComputeTildeACSR_iter_SpMMv1<256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
		        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
		    }
		    
		 
		    cudaEventRecord( stop, 0 ) ;
		    cudaEventSynchronize( stop ) ;
		    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
		  
		    printf( "Time = : %8.4f ms \n ", elapsedTime ) ;
		    computeTime += elapsedTime;

		    cudaEventDestroy( start ) ;
		    cudaEventDestroy( stop  ) ;
		    
		    printf("-----------------Compute A*B\n");
	    
		    cudaEventCreate( &start ) ;
        cudaEventCreate( &stop  ) ;
        cudaEventRecord( start, 0 ) ;
			  
		    if(n2max <= 2){
		    	ComputeAB_iter_SpMMv1<2, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, eGrid);
		    }else if(n2max >2 && n2max <= 4){
		    	ComputeAB_iter_SpMMv1<4, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, eGrid);
		    }else if(n2max >4 && n2max <= 8){
		    	ComputeAB_iter_SpMMv1<8, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, eGrid);
		    }else if(n2max >8 && n2max <= 16){
		    	ComputeAB_iter_SpMMv1<16, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, eGrid);
		    }else if(n2max >16 && n2max <= 32){
		    	ComputeAB_iter_SpMMv1<32, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, eGrid);
		    }else if(n2max >32 && n2max <= 64){
		    	ComputeAB_iter_SpMMv1<64, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, eGrid);
		    }else if(n2max >64 && n2max <= 128){
		    	ComputeAB_iter_SpMMv1<128, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, eGrid);
		    }else if(n2max >128 && n2max <= 256){
		    	ComputeAB_iter_SpMMv1<256, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, eGrid);
		    }else{
		    	ComputeAB_iter_SpMMv1NoSharedM<256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, eGrid);
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
        // if( iter == (itertions - 1) ) eGrid = devM->nCol - (itertions - 1) * nCols;
        
	      
	      cudaMemcpyAsync( dev_tempPTR + sK, dev_iPTR, eGrid * sizeof( int ), cudaMemcpyDeviceToDevice, 0 ) ;
	      cudaMemcpyAsync( dev_tempIndex + sK * n1max, dev_I, eGrid * n1max * sizeof( int ), cudaMemcpyDeviceToDevice, 0 ) ;
	      cudaMemcpyAsync( dev_tempData + sK * n1max, dev_X, eGrid * n1max * sizeof( double ), cudaMemcpyDeviceToDevice, 0 ) ;
	      
	      int *cpuPtr = (int*)malloc(sizeof(int) * eGrid);
				cudaMemcpyAsync( cpuPtr, dev_iPTR, eGrid * sizeof( int ), cudaMemcpyDeviceToHost, 0 ) ; 
					
//			  for(int i = 0; i < eGrid; i++){
//					printf("i=%d, p[%d]=%d\n", i, i, cpuPtr[i]);
//			  }
      }   
	    //exit(0);
	    /*+++++++++++++++++Post-SpMM++++++++++++++++++++++++++++++*/
	    
	    printf("Post-SpMM is processing.........................\n");
	    cudaEventCreate( &start ) ;
      cudaEventCreate( &stop  ) ;
      cudaEventRecord( start, 0 ) ;

      devC->n = devM->nCol;
	    devC->nCol = devM->nCol;
	    devC->nRow = devM->nCol;
	    cudaMalloc((void**)&devC->mPtr, sizeof(int) * (devC->n + 1)) ;
	    cudaMemcpyAsync( devC->mPtr + 1, dev_tempPTR, devM->nCol * sizeof( int ), cudaMemcpyDeviceToDevice, 0 ) ;
	    

	    int threadsPerBlock1 = 1024;
	    int blocksPerGrid1 = 1;
	    summation_SpMMv1<<<blocksPerGrid1, threadsPerBlock1>>>(devC->mPtr, devM->nCol+1);
	    
	    int *nnz = (int*)malloc(sizeof(int));
		  cudaMemcpyAsync( nnz, devC->mPtr+devC->n, sizeof( int ), cudaMemcpyDeviceToHost, 0 ) ; 
		  
//			
//			for(int i = 0; i < (devM->nCol +1); i++){
//				printf("i=%d, p[%d]=%d\n", i, i, cpuPtr[i]);
//			}
	    
	    devC->nonzeroes = *nnz ;
	    cudaMalloc((void**)&devC->mIndex, sizeof(int) * devC->nonzeroes ) ;
      cudaMalloc((void**)&devC->mData, sizeof(double) * devC->nonzeroes) ;
      
      if(n1max < 3){
		   	 //threadsPerBlock = 256;
			   blocksPerGrid = (devC->n - 1)/(threadsPerBlock/2) + 1;
			   WriteABIntoC_SpMMv1<2><<<blocksPerGrid, threadsPerBlock>>>(dev_tempIndex, dev_tempData, devC->mPtr, devC->mIndex, devC->mData, n1max, devC->n);
			}else if(n1max >= 3 && n1max < 6){
				 //threadsPerBlock = 256;
			   blocksPerGrid = (devC->n - 1)/(threadsPerBlock/4) + 1;
				 WriteABIntoC_SpMMv1<4><<<blocksPerGrid, threadsPerBlock>>>(dev_tempIndex, dev_tempData, devC->mPtr, devC->mIndex, devC->mData, n1max, devC->n);
			}else if(n1max >= 6 && n1max < 12){
				 //threadsPerBlock = 256;
			   blocksPerGrid = (devC->n - 1)/(threadsPerBlock/8) + 1;
				 WriteABIntoC_SpMMv1<8><<<blocksPerGrid, threadsPerBlock>>>(dev_tempIndex, dev_tempData, devC->mPtr, devC->mIndex, devC->mData, n1max, devC->n);
			}else if(n1max >= 12 && n1max < 24){
				 //threadsPerBlock = 256;
			   blocksPerGrid = (devC->n - 1)/(threadsPerBlock/16) + 1;
				 WriteABIntoC_SpMMv1<16><<<blocksPerGrid, threadsPerBlock>>>(dev_tempIndex, dev_tempData, devC->mPtr, devC->mIndex, devC->mData, n1max, devC->n);
			}else {
				 //threadsPerBlock = 256;
			   blocksPerGrid = (devC->n - 1)/(threadsPerBlock/32) + 1;
				 WriteABIntoC_SpMMv1<32><<<blocksPerGrid, threadsPerBlock>>>(dev_tempIndex, dev_tempData, devC->mPtr, devC->mIndex, devC->mData, n1max, devC->n);
			}
	    
	    cudaEventRecord( stop, 0 ) ;
	    cudaEventSynchronize( stop ) ;
	    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
	  
	    printf("\n");
	    printf( "postTime = : %12.4f ms \n ", elapsedTime ) ;
	    
	    cudaEventDestroy( start ) ;
	    cudaEventDestroy( stop  ) ;
	    
	    printf("nnz = %d\n", *nnz);
	    
	    free(nnz);
	    
      //exit(0);
      printf("*************************************************\n");
	    printf("TotalPreTime = : %12.4f ms \n", preTime );
	    printf("TotalComputeTime = : %12.4f ms \n", computeTime );
	    printf( "TotalPostTime = : %12.4f ms \n ", elapsedTime ) ;
	    
	    printf( "TotalTime = : %12.4f ms \n ", preTime + computeTime + elapsedTime ) ;
      printf("\n");
	    
	    //printf("Transfering CSC_M to CSR_M is processing.........\n");


	    cudaFree(dev_tempPTR);
	    cudaFree(dev_tempIndex);
	    cudaFree(dev_tempData);

	    cudaFree(dev_J);
	    cudaFree(dev_JV);
	    cudaFree(dev_I);
	    cudaFree(dev_jPTR);
	    cudaFree(dev_iPTR);
	    
	    cudaFree(dev_tildeA);
	    cudaFree(dev_X);
	    
    }else{
	    printf("------Allocate GPU global memroy\n");
	    
	    cudaEventCreate( &start ) ;
      cudaEventCreate( &stop  ) ;
      cudaEventRecord( start, 0 ) ;
		  
	    double *dev_tildeA;

	    int *dev_J, *dev_jPTR;
	    int *dev_I, *dev_iPTR;
	    double *dev_X, *dev_JV;
	 
	    cudaMalloc((void**)&dev_J, sizeof(int) * devM->nCol * n2max) ;
	    cudaMalloc((void**)&dev_jPTR, sizeof(int) * devM->nCol) ;
	    cudaMalloc((void**)&dev_JV, sizeof(double) * devM->nCol * n2max) ;

	    cudaMalloc((void**)&dev_I, sizeof(int) * devM->nCol * n1max) ;
	    cudaMalloc((void**)&dev_iPTR, sizeof(int) * devM->nCol) ;
	    
	    cudaMalloc((void**)&dev_tildeA, sizeof(double) * devM->nCol * n1max * n2max) ;
	    cudaMalloc((void**)&dev_X, sizeof(double) * devM->nCol * n1max) ;

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
	      computeJ_SpMMv1<2><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, devM->nCol, dev_J, dev_jPTR, dev_JV, n2max);
	    }else if(n2max >2 && n2max <= 4){
	    	computeJ_SpMMv1<4><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, devM->nCol, dev_J, dev_jPTR, dev_JV, n2max);
	    }else if(n2max >4 && n2max <= 8){
	    	computeJ_SpMMv1<8><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, devM->nCol, dev_J, dev_jPTR, dev_JV, n2max);
	    }else if(n2max >8 && n2max <= 16){
	    	computeJ_SpMMv1<16><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, devM->nCol, dev_J, dev_jPTR, dev_JV, n2max);
	    }else if(n2max >16 && n2max <= 32){
	    	computeJ_SpMMv1<32><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, devM->nCol, dev_J, dev_jPTR, dev_JV, n2max);
	    }else if(n2max >32 && n2max <= 64){
	    	computeJ_SpMMv1<64><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, devM->nCol, dev_J, dev_jPTR, dev_JV, n2max);
	    }else if(n2max >64 && n2max <= 128){
	    	computeJ_SpMMv1<128><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, devM->nCol, dev_J, dev_jPTR, dev_JV, n2max);
	    }else{
	    	computeJ_SpMMv1<256><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, devM->nCol, dev_J, dev_jPTR, dev_JV, n2max);
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
	      computeI_SpMMv1<2,256/2><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
	    }else if(n2max >2 && n2max <= 4){
	    	computeI_SpMMv1<4,256/4><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
	    }else if(n2max >4 && n2max <= 8){
	    	computeI_SpMMv1<8,256/8><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
	    }else if(n2max >8 && n2max <= 16){
	    	computeI_SpMMv1<16,256/16><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
	    }else if(n2max >16 && n2max <= 32){
	    	computeI_SpMMv1<32,256/32><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
	    }else if(n2max >32 && n2max <= 64){
	    	computeI_SpMMv1<64,256/64><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
	    }else if(n2max >64 && n2max <= 128){
	    	computeI_SpMMv1<128,256/128><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
	    }else{
	    	computeI_SpMMv1<256,256/256><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
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
	        ComputeTildeACSR_SpMMv1<2><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
	        devA->mIndex, devM->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
	    }else if(n2max >2 && n2max <= 4){
	    	ComputeTildeACSR_SpMMv1<4><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
	        devA->mIndex, devM->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
	    }else if(n2max >4 && n2max <= 8){
	    	ComputeTildeACSR_SpMMv1<8><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
	        devA->mIndex, devM->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
	    }else if(n2max >8 && n2max <= 16){
	    	ComputeTildeACSR_SpMMv1<16><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
	        devA->mIndex, devM->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
	    }else if(n2max >16 && n2max <= 32){
	    	ComputeTildeACSR_SpMMv1<32><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
	        devA->mIndex, devM->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
	    }else if(n2max >32 && n2max <= 64){
	    	ComputeTildeACSR_SpMMv1<64><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
	        devA->mIndex, devM->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
	    }else if(n2max >64 && n2max <= 128){
	    	ComputeTildeACSR_SpMMv1<128><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
	        devA->mIndex, devM->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
	    }else{
	    	ComputeTildeACSR_SpMMv1<256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
	        devA->mIndex, devM->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
	    }
	 
	    cudaEventRecord( stop, 0 ) ;
	    cudaEventSynchronize( stop ) ;
	    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
	  
	    printf( "Time = : %8.4f ms \n ", elapsedTime ) ;
	    computeTime += elapsedTime;

	    cudaEventDestroy( start ) ;
	    cudaEventDestroy( stop  ) ;
	    
	    printf("-----------------Compute A * B\n");

	    cudaEventCreate( &start ) ;
      cudaEventCreate( &stop  ) ;
      cudaEventRecord( start, 0 ) ;
     
      if(n2max <= 2){
		    ComputeAB_iter_SpMMv1<2, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		        dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, devM->nCol);
		  }else if(n2max >2 && n2max <= 4){
		    ComputeAB_iter_SpMMv1<4, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		        dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, devM->nCol);
		  }else if(n2max >4 && n2max <= 8){
		    ComputeAB_iter_SpMMv1<8, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		        dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, devM->nCol);
		  }else if(n2max >8 && n2max <= 16){
		    ComputeAB_iter_SpMMv1<16, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		        dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, devM->nCol);
		  }else if(n2max >16 && n2max <= 32){
		    ComputeAB_iter_SpMMv1<32, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		        dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, devM->nCol);
		  }else if(n2max >32 && n2max <= 64){
		    ComputeAB_iter_SpMMv1<64, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		        dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, devM->nCol);
		  }else if(n2max >64 && n2max <= 128){
		    ComputeAB_iter_SpMMv1<128, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		        dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, devM->nCol);
		  }else if(n2max >128 && n2max <= 256){
		    ComputeAB_iter_SpMMv1<256, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		        dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, devM->nCol);
		  }else{
		    ComputeAB_iter_SpMMv1NoSharedM<256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		        dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, devM->nCol);
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
	   
	    /*+++++++++++++++++Post-SpMM++++++++++++++++++++++++++++++*/
	    
	    printf("Post-SpMM is processing.........................\n");
	    cudaEventCreate( &start ) ;
      cudaEventCreate( &stop  ) ;
      cudaEventRecord( start, 0 ) ;
      
      devC->n = devM->nCol;
	    devC->nCol = devM->nCol;
	    devC->nRow = devM->nCol;
	    cudaMalloc((void**)&devC->mPtr, sizeof(int) * (devC->n + 1)) ;
	    cudaMemcpyAsync( devC->mPtr + 1, dev_iPTR, devM->nCol * sizeof( int ), cudaMemcpyDeviceToDevice, 0 ) ;
	    

	    int threadsPerBlock1 = 1024;
	    int blocksPerGrid1 = 1;
	    summation_SpMMv1<<<blocksPerGrid1, threadsPerBlock1>>>(devC->mPtr, devM->nCol+1);
	    
	    int *nnz = (int*)malloc(sizeof(int));
		  cudaMemcpyAsync( nnz, devC->mPtr+devC->n, sizeof( int ), cudaMemcpyDeviceToHost, 0 ) ; 
		  
		  
			
//			for(int i = 0; i < (devM->nCol +1); i++){
//				printf("i=%d, p[%d]=%d\n", i, i, cpuPtr[i]);
//			}
	    
	    devC->nonzeroes = *nnz ;
	    cudaMalloc((void**)&devC->mIndex, sizeof(int) * devC->nonzeroes ) ;
      cudaMalloc((void**)&devC->mData, sizeof(double) * devC->nonzeroes) ;
      
      if(n1max < 3){
		   	 //threadsPerBlock = 256;
			   blocksPerGrid = (devC->n - 1)/(threadsPerBlock/2) + 1;
			   WriteABIntoC_SpMMv1<2><<<blocksPerGrid, threadsPerBlock>>>(dev_I, dev_X, devC->mPtr, devC->mIndex, devC->mData, n1max, devC->n);
			}else if(n1max >= 3 && n1max < 6){
				 //threadsPerBlock = 256;
			   blocksPerGrid = (devC->n - 1)/(threadsPerBlock/4) + 1;
				 WriteABIntoC_SpMMv1<4><<<blocksPerGrid, threadsPerBlock>>>(dev_I, dev_X, devC->mPtr, devC->mIndex, devC->mData, n1max, devC->n);
			}else if(n1max >= 6 && n1max < 12){
				 //threadsPerBlock = 256;
			   blocksPerGrid = (devC->n - 1)/(threadsPerBlock/8) + 1;
				 WriteABIntoC_SpMMv1<8><<<blocksPerGrid, threadsPerBlock>>>(dev_I, dev_X, devC->mPtr, devC->mIndex, devC->mData, n1max, devC->n);
			}else if(n1max >= 12 && n1max < 24){
				 //threadsPerBlock = 256;
			   blocksPerGrid = (devC->n - 1)/(threadsPerBlock/16) + 1;
				 WriteABIntoC_SpMMv1<16><<<blocksPerGrid, threadsPerBlock>>>(dev_I, dev_X, devC->mPtr, devC->mIndex, devC->mData, n1max, devC->n);
			}else {
				 //threadsPerBlock = 256;
			   blocksPerGrid = (devC->n - 1)/(threadsPerBlock/32) + 1;
				 WriteABIntoC_SpMMv1<32><<<blocksPerGrid, threadsPerBlock>>>(dev_I, dev_X, devC->mPtr, devC->mIndex, devC->mData, n1max, devC->n);
			}
	    
	    cudaEventRecord( stop, 0 ) ;
	    cudaEventSynchronize( stop ) ;
	    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
	  
	    printf("\n");
	    printf( "postTime = : %12.4f ms \n ", elapsedTime ) ;
	    
	    cudaEventDestroy( start ) ;
	    cudaEventDestroy( stop  ) ;
	    
	    printf("nnz = %d\n", *nnz);
	    free(nnz);
	    
	    //exit(0);
	    
	    printf("******************The SpMM time**********************************\n");
	    printf("TotalPreTime = : %12.4f ms \n", preTime );
	    printf("TotalComputeTime = : %12.4f ms \n", computeTime );
	    printf( "TotalPostTime = : %12.4f ms \n ", elapsedTime ) ;
	    
	    printf( "TotalTime = : %12.4f ms \n ", preTime + computeTime + elapsedTime ) ;
	    printf("\n");
	    

      cudaFree(dev_tempPTR);
	    cudaFree(dev_tempIndex);
	    cudaFree(dev_tempData);
      
	    cudaFree(dev_J);
	    cudaFree(dev_I);
	    cudaFree(dev_jPTR);
	    cudaFree(dev_iPTR);
	    cudaFree(dev_JV);
	    
	    cudaFree(dev_tildeA);
	    cudaFree(dev_X);
	  }
	  
	  return preTime + computeTime + elapsedTime;
}

float SpMMWithGuassSparsity(CSC_Matrix *devA, CSC_Matrix *devM, CSC_Matrix *devC, double GuassCoeff = 3.0){
    
    /*+++++++++++++++++Pre-SpMM+++++++++++++++++++++++++++++++*/
    printf("Pre-SpMM is processing..........................\n");
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
    
    cuComputeN2MAXwithSparityofA_SpMMv1<threadsPerBlock><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->n, dev_n2);
    
    cudaMemcpyAsync( n2, dev_n2, blocksPerGrid * sizeof( int ), cudaMemcpyDeviceToHost, 0 ) ;
    int n2max = 0;
    for(int i = 0; i < blocksPerGrid; i++){
    	if(n2max < n2[i]) n2max = n2[i];
    }
    
    cudaFree(dev_n2);
    free(n2);
	  
    //int n2max = computeN2MAX(CSC_A);
    
    int *maxV = (int*)malloc(sizeof(int) * blocksPerGrid);
    int *minV = (int*)malloc(sizeof(int) * blocksPerGrid);
    int *dev_maxV, *dev_minV;
    cudaMalloc((void**)&dev_maxV, sizeof(int) * blocksPerGrid) ;
    cudaMalloc((void**)&dev_minV, sizeof(int) * blocksPerGrid) ;
    
    cuComputeN2MAXwithSparityofA_SpMMv1<threadsPerBlock><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->n, dev_maxV);
    cuComputeN2MINwithSparityofA_SpMMv2<threadsPerBlock><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->n, dev_minV);
    
    cudaMemcpyAsync( maxV, dev_maxV, blocksPerGrid * sizeof( int ), cudaMemcpyDeviceToHost, 0 ) ;
    cudaMemcpyAsync( minV, dev_minV, blocksPerGrid * sizeof( int ), cudaMemcpyDeviceToHost, 0 ) ;
    
    int maxA = maxV[0];
    for(int i = 1; i < blocksPerGrid; i++){
    	if(maxA < maxV[i]) maxA = maxV[i];
    }
    
    int minA = minV[0];
    for(int i = 1; i < blocksPerGrid; i++){
    	if(minA > minV[i]) minA = minV[i];
    }
    
    
    int *dev_NZUPP;
    cudaMalloc((void**)&dev_NZUPP, sizeof(int) * devA->n) ; 
    computeNZUPPWithGuass_SpMMv2<< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->n, devA->nonzeroes, 
                                 dev_NZUPP, GuassCoeff, maxA, minA);
                                 
    cudaFree(dev_maxV);
    cudaFree(dev_minV);
    free(maxV);
    free(minV);
    
    cudaEventRecord( stop, 0 ) ;
    cudaEventSynchronize( stop ) ;
    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
  
    printf( "Time = : %8.4f ms \n ", elapsedTime ) ;
    preTime += elapsedTime;

    cudaEventDestroy( start ) ;
    cudaEventDestroy( stop  ) ;
    
    printf("maxA = %d, minA = %d\n", maxA, minA);
    
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
    	cuComputeN1withSparityofA_SpMMv1<2,256/2,256><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >2 && n2max <= 4){
    	cuComputeN1withSparityofA_SpMMv1<4,256/4,256><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >4 && n2max <= 8){
    	cuComputeN1withSparityofA_SpMMv1<8,256/8,256><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >8 && n2max <= 16){
    	cuComputeN1withSparityofA_SpMMv1<16,256/16,256><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >16 && n2max <= 32){
    	cuComputeN1withSparityofA_SpMMv1<32,256/32,256><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >32 && n2max <= 64){
    	cuComputeN1withSparityofA_SpMMv1<32,256/32,512><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >64 && n2max <= 128){
    	cuComputeN1withSparityofA_SpMMv1<32,256/32,1024><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >128 && n2max <= 256){
    	cuComputeN1withSparityofA_SpMMv1<32,256/32,2048><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >256 && n2max <= 512){
    	cuComputeN1withSparityofA_SpMMv1<32,256/32,4096><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else if(n2max >512 && n2max <= 1024){
      // printf("enter this n2max branch. \n");
	  cuComputeN1withSparityofA_SpMMv1<32,256/32,8192><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    }else{
		cuComputeN1withSparityofA2_SpMMv1<32,256/32><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->mPtr, devM->mIndex, dev_tI, ssize, devM->n, dev_n1);
    	// printf("Sorry, exceed the maximum shared memory in computing n1max\n");
      // exit(0);
    }
    
    cudaFree(dev_tI);

    
    blocksPerGrid = 15 * 8 * 4;
    int *n1 = (int*)malloc(sizeof(int) * blocksPerGrid);
    int *dev_n1max;
    cudaMalloc((void**)&dev_n1max, sizeof(int) * blocksPerGrid) ;
    
    cuComputeN1MAX_SpMMv1<threadsPerBlock><< <blocksPerGrid, threadsPerBlock>> >(dev_n1, devM->n, dev_n1max);
    
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
    printf("Compute-SpMM is processing......................\n");
    
    float computeTime = 0.0;
    
    /*********************************************************************************
    **                                                                               *
    ** Judge whether the memroy is exceeded?                                         *
    **                                                                               *
    **********************************************************************************/
    
    int compareSize = 1024*1024*100*4;
    long realSize = (long)devM->nCol * n1max * n2max;
     
    int itertions = (realSize-1)/compareSize + 1;
    //int itertions = 2;
    
    printf("compareSize=%d, realSize=%ld, iterations =%d\n", compareSize, realSize, itertions);
    
    int *dev_tempPTR, *dev_tempIndex;
    double *dev_tempData;
    cudaMalloc((void**)&dev_tempIndex, sizeof(int) * devM->nCol * n1max) ;
	  cudaMalloc((void**)&dev_tempPTR, sizeof(int) * devM->nCol) ;
	    
	  cudaMalloc((void**)&dev_tempData, sizeof(double) * devM->nCol * n1max) ;
    
    
    //exit(0);
    if( itertions != 1){
    	const int nCols = (devM->nCol-1)/itertions + 1;
    	
    	printf("------Allocate GPU global memroy\n");
    	
    	cudaEventCreate( &start ) ;
      cudaEventCreate( &stop  ) ;
      cudaEventRecord( start, 0 ) ;
		  
	    double *dev_tildeA;

	    int *dev_J, *dev_jPTR;
	    int *dev_I, *dev_iPTR;
	    double *dev_X, *dev_JV;
	 
	    cudaMalloc((void**)&dev_J, sizeof(int) * nCols * n2max) ;
	    cudaMalloc((void**)&dev_jPTR, sizeof(int) * nCols) ;
	    cudaMalloc((void**)&dev_JV, sizeof(double) * nCols * n2max) ;

	    cudaMalloc((void**)&dev_I, sizeof(int) * nCols * n1max) ;
	    cudaMalloc((void**)&dev_iPTR, sizeof(int) * nCols) ;
	    
	    cudaMalloc((void**)&dev_tildeA, sizeof(double) * nCols * n1max * n2max) ;
	    cudaMalloc((void**)&dev_X, sizeof(double) * nCols * n1max) ;

      cudaEventRecord( stop, 0 ) ;
	    cudaEventSynchronize( stop ) ;
	    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
	  
	    printf( "Time = : %8.4f ms \n ", elapsedTime ) ;
	    computeTime += elapsedTime;

	    cudaEventDestroy( start ) ;
	    cudaEventDestroy( stop  ) ;
	    
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
		        computeJ_iter_SpMMv1<2><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, eGrid, dev_J, dev_jPTR, dev_JV, n2max, sK);
		    }else if(n2max >2 && n2max <= 4){
		    	computeJ_iter_SpMMv1<4><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, eGrid, dev_J, dev_jPTR, dev_JV, n2max, sK);
		    }else if(n2max >4 && n2max <= 8){
		    	computeJ_iter_SpMMv1<8><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, eGrid, dev_J, dev_jPTR, dev_JV, n2max, sK);
		    }else if(n2max >8 && n2max <= 16){
		    	computeJ_iter_SpMMv1<16><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, eGrid, dev_J, dev_jPTR, dev_JV, n2max, sK);
		    }else if(n2max >16 && n2max <= 32){
		    	computeJ_iter_SpMMv1<32><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, eGrid, dev_J, dev_jPTR, dev_JV, n2max, sK);
		    }else if(n2max >32 && n2max <= 64){
		    	computeJ_iter_SpMMv1<64><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, eGrid, dev_J, dev_jPTR, dev_JV, n2max, sK);
		    }else if(n2max >64 && n2max <= 128){
		    	computeJ_iter_SpMMv1<128><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, eGrid, dev_J, dev_jPTR, dev_JV, n2max, sK);
		    }else{
		    	computeJ_iter_SpMMv1<256><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, eGrid, dev_J, dev_jPTR, dev_JV, n2max, sK);
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
		        computeI_iter_SpMMv1<2,256/2><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
		    }else if(n2max >2 && n2max <= 4){
		    	computeI_iter_SpMMv1<4,256/4><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
		    }else if(n2max >4 && n2max <= 8){
		    	computeI_iter_SpMMv1<8,256/8><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
		    }else if(n2max >8 && n2max <= 16){
		    	computeI_iter_SpMMv1<16,256/16><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
		    }else if(n2max >16 && n2max <= 32){
		    	computeI_iter_SpMMv1<32,256/32><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
		    }else if(n2max >32 && n2max <= 64){
		    	computeI_iter_SpMMv1<64,256/64><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
		    }else if(n2max >64 && n2max <= 128){
		    	computeI_iter_SpMMv1<128,256/128><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
		    }else{
		    	computeI_iter_SpMMv1<256,256/256><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
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
		        ComputeTildeACSR_iter_SpMMv1<2><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
		        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
		    }else if(n2max >2 && n2max <= 4){
		    	ComputeTildeACSR_iter_SpMMv1<4><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
		        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
		    }else if(n2max >4 && n2max <= 8){
		    	ComputeTildeACSR_iter_SpMMv1<8><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
		        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
		    }else if(n2max >8 && n2max <= 16){
		    	ComputeTildeACSR_iter_SpMMv1<16><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
		        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
		    }else if(n2max >16 && n2max <= 32){
		    	ComputeTildeACSR_iter_SpMMv1<32><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
		        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
		    }else if(n2max >32 && n2max <= 64){
		    	ComputeTildeACSR_iter_SpMMv1<64><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
		        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
		    }else if(n2max >64 && n2max <= 128){
		    	ComputeTildeACSR_iter_SpMMv1<128><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
		        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
		    }else{
		    	ComputeTildeACSR_iter_SpMMv1<256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
		        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
		    }
		    
		 
		    cudaEventRecord( stop, 0 ) ;
		    cudaEventSynchronize( stop ) ;
		    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
		  
		    printf( "Time = : %8.4f ms \n ", elapsedTime ) ;
		    computeTime += elapsedTime;

		    cudaEventDestroy( start ) ;
		    cudaEventDestroy( stop  ) ;
		    
		    printf("-----------------Compute A*B\n");
	    
		    cudaEventCreate( &start ) ;
        cudaEventCreate( &stop  ) ;
        cudaEventRecord( start, 0 ) ;
			  
		    if(n2max <= 2){
		    	ComputeAB_iter_SpMMv1<2, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, eGrid);
		    }else if(n2max >2 && n2max <= 4){
		    	ComputeAB_iter_SpMMv1<4, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, eGrid);
		    }else if(n2max >4 && n2max <= 8){
		    	ComputeAB_iter_SpMMv1<8, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, eGrid);
		    }else if(n2max >8 && n2max <= 16){
		    	ComputeAB_iter_SpMMv1<16, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, eGrid);
		    }else if(n2max >16 && n2max <= 32){
		    	ComputeAB_iter_SpMMv1<32, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, eGrid);
		    }else if(n2max >32 && n2max <= 64){
		    	ComputeAB_iter_SpMMv1<64, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, eGrid);
		    }else if(n2max >64 && n2max <= 128){
		    	ComputeAB_iter_SpMMv1<128, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, eGrid);
		    }else if(n2max >128 && n2max <= 256){
		    	ComputeAB_iter_SpMMv1<256, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, eGrid);
		    }else{
		    	ComputeAB_iter_SpMMv1NoSharedM<256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		         dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, eGrid);
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
        // if( iter == (itertions - 1) ) eGrid = devM->nCol - (itertions - 1) * nCols;
        
	      
	      cudaMemcpyAsync( dev_tempPTR + sK, dev_iPTR, eGrid * sizeof( int ), cudaMemcpyDeviceToDevice, 0 ) ;
	      cudaMemcpyAsync( dev_tempIndex + sK * n1max, dev_I, eGrid * n1max * sizeof( int ), cudaMemcpyDeviceToDevice, 0 ) ;
	      cudaMemcpyAsync( dev_tempData + sK * n1max, dev_X, eGrid * n1max * sizeof( double ), cudaMemcpyDeviceToDevice, 0 ) ;
	      
	      int *cpuPtr = (int*)malloc(sizeof(int) * eGrid);
				cudaMemcpyAsync( cpuPtr, dev_iPTR, eGrid * sizeof( int ), cudaMemcpyDeviceToHost, 0 ) ; 
					
//			  for(int i = 0; i < eGrid; i++){
//					printf("i=%d, p[%d]=%d\n", i, i, cpuPtr[i]);
//			  }
      }   
	    //exit(0);
	    /*+++++++++++++++++Post-SpMM++++++++++++++++++++++++++++++*/
	    
	    printf("Post-SpMM in Iteration is processing.........................\n");
	    cudaEventCreate( &start ) ;
      cudaEventCreate( &stop  ) ;
      cudaEventRecord( start, 0 ) ;
      
      if(n1max <= 2){
      	blocksPerGrid = (devM->nCol - 1)/(threadsPerBlock/2) + 1;
		    SparistyGuass_SpMMv2<2><< <blocksPerGrid, threadsPerBlock>> >(dev_tempPTR, 
		        dev_tempData, dev_tempIndex, dev_NZUPP, n1max, devM->nCol);
		  }else if(n1max >2 && n1max <= 4){
		  	blocksPerGrid = (devM->nCol - 1)/(threadsPerBlock/4) + 1;
		    SparistyGuass_SpMMv2<4><< <blocksPerGrid, threadsPerBlock>> >(dev_tempPTR, 
		        dev_tempData, dev_tempIndex, dev_NZUPP, n1max, devM->nCol);
		  }else if(n1max >4 && n1max <= 8){
		  	blocksPerGrid = (devM->nCol - 1)/(threadsPerBlock/8) + 1;
		    SparistyGuass_SpMMv2<8><< <blocksPerGrid, threadsPerBlock>> >(dev_tempPTR, 
		        dev_tempData, dev_tempIndex, dev_NZUPP, n1max, devM->nCol);
		  }else if(n1max >8 && n1max <= 16){
		  	blocksPerGrid = (devM->nCol - 1)/(threadsPerBlock/16) + 1;
		    SparistyGuass_SpMMv2<16><< <blocksPerGrid, threadsPerBlock>> >(dev_tempPTR, 
		        dev_tempData, dev_tempIndex, dev_NZUPP, n1max, devM->nCol);
		  }else if(n1max >16 && n1max <= 32){
		  	blocksPerGrid = (devM->nCol - 1)/(threadsPerBlock/32) + 1;
		    SparistyGuass_SpMMv2<32><< <blocksPerGrid, threadsPerBlock>> >(dev_tempPTR, 
		        dev_tempData, dev_tempIndex, dev_NZUPP, n1max, devM->nCol);
		  }else if(n1max >32 && n1max <= 64){
		  	blocksPerGrid = (devM->nCol - 1)/(threadsPerBlock/64) + 1;
		    SparistyGuass_SpMMv2<64><< <blocksPerGrid, threadsPerBlock>> >(dev_tempPTR, 
		        dev_tempData, dev_tempIndex, dev_NZUPP, n1max, devM->nCol);
		  }else if(n1max >64 && n1max <= 128){
		  	blocksPerGrid = (devM->nCol - 1)/(threadsPerBlock/128) + 1;
		    SparistyGuass_SpMMv2<128><< <blocksPerGrid, threadsPerBlock>> >(dev_tempPTR, 
		        dev_tempData, dev_tempIndex, dev_NZUPP, n1max, devM->nCol);
		  }else{
		  	blocksPerGrid = (devM->nCol - 1)/(threadsPerBlock/256) + 1;
		    SparistyGuass_SpMMv2<256><< <blocksPerGrid, threadsPerBlock>> >(dev_tempPTR, 
		        dev_tempData, dev_tempIndex, dev_NZUPP, n1max, devM->nCol);
		  }
      
//      int *iPTR = (int*)malloc(sizeof(int)*devM->nCol);
//      cudaMemcpyAsync( iPTR, dev_tempPTR, devM->nCol * sizeof( int ), cudaMemcpyDeviceToHost, 0 ) ; 
//      
//      int *IIndex = (int*)malloc(sizeof(int)*n1max*devM->nCol);
//      cudaMemcpyAsync( IIndex, dev_tempIndex, devM->nCol * n1max * sizeof( int ), cudaMemcpyDeviceToHost, 0 ) ; 
//      
//      double *idata = (double*)malloc(sizeof(double)*n1max*devM->nCol);
//      cudaMemcpyAsync( idata, dev_tempData, devM->nCol * n1max * sizeof( double ), cudaMemcpyDeviceToHost, 0 ) ; 
//
//    
//    
//    
//    //for(int i = 0 ; i < devM->nCol; i++){
//    //for(int i = 0 ; i < 10; i++){
//    for(int i = devM->nCol -10 ; i < devM->nCol; i++){
//    	for(int j = 0; j < iPTR[i]; j++){
//    	  printf("%d ", IIndex[i*n1max + j]);
//    	}
//    	printf("\n");
//    }
//    
//    
//     //for(int i = 0 ; i < devM->nCol; i++){
//     //for(int i = 0 ; i < 10; i++){
//     for(int i = devM->nCol -10 ; i < devM->nCol; i++){
//    	for(int j = 0; j < iPTR[i]; j++){
//    	  printf("%lf ", idata[i*n1max +j]);
//    	}
//    	printf("\n");
//    }

      devC->n = devM->nCol;
	    devC->nCol = devM->nCol;
	    devC->nRow = devM->nCol;
	    cudaMalloc((void**)&devC->mPtr, sizeof(int) * (devC->n + 1)) ;
	    cudaMemcpyAsync( devC->mPtr + 1, dev_tempPTR, devM->nCol * sizeof( int ), cudaMemcpyDeviceToDevice, 0 ) ;
	    

	    int threadsPerBlock1 = 1024;
	    int blocksPerGrid1 = 1;
	    summation_SpMMv1<<<blocksPerGrid1, threadsPerBlock1>>>(devC->mPtr, devM->nCol+1);
	    
	    int *nnz = (int*)malloc(sizeof(int));
		  cudaMemcpyAsync( nnz, devC->mPtr+devC->n, sizeof( int ), cudaMemcpyDeviceToHost, 0 ) ; 
//			
//			for(int i = 0; i < (devM->nCol +1); i++){
//				printf("i=%d, p[%d]=%d\n", i, i, cpuPtr[i]);
//			}
	    
	    devC->nonzeroes = *nnz ;
	    cudaMalloc((void**)&devC->mIndex, sizeof(int) * devC->nonzeroes ) ;
      cudaMalloc((void**)&devC->mData, sizeof(double) * devC->nonzeroes) ;
      
      if(n1max < 3){
		   	 //threadsPerBlock = 256;
			   blocksPerGrid = (devC->n - 1)/(threadsPerBlock/2) + 1;
			   WriteABIntoC_SpMMv1<2><<<blocksPerGrid, threadsPerBlock>>>(dev_tempIndex, dev_tempData, devC->mPtr, devC->mIndex, devC->mData, n1max, devC->n);
			}else if(n1max >= 3 && n1max < 6){
				 //threadsPerBlock = 256;
			   blocksPerGrid = (devC->n - 1)/(threadsPerBlock/4) + 1;
				 WriteABIntoC_SpMMv1<4><<<blocksPerGrid, threadsPerBlock>>>(dev_tempIndex, dev_tempData, devC->mPtr, devC->mIndex, devC->mData, n1max, devC->n);
			}else if(n1max >= 6 && n1max < 12){
				 //threadsPerBlock = 256;
			   blocksPerGrid = (devC->n - 1)/(threadsPerBlock/8) + 1;
				 WriteABIntoC_SpMMv1<8><<<blocksPerGrid, threadsPerBlock>>>(dev_tempIndex, dev_tempData, devC->mPtr, devC->mIndex, devC->mData, n1max, devC->n);
			}else if(n1max >= 12 && n1max < 24){
				 //threadsPerBlock = 256;
			   blocksPerGrid = (devC->n - 1)/(threadsPerBlock/16) + 1;
				 WriteABIntoC_SpMMv1<16><<<blocksPerGrid, threadsPerBlock>>>(dev_tempIndex, dev_tempData, devC->mPtr, devC->mIndex, devC->mData, n1max, devC->n);
			}else {
				 //threadsPerBlock = 256;
			   blocksPerGrid = (devC->n - 1)/(threadsPerBlock/32) + 1;
				 WriteABIntoC_SpMMv1<32><<<blocksPerGrid, threadsPerBlock>>>(dev_tempIndex, dev_tempData, devC->mPtr, devC->mIndex, devC->mData, n1max, devC->n);
			}
	    
	    cudaEventRecord( stop, 0 ) ;
	    cudaEventSynchronize( stop ) ;
	    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
	  
	    printf("\n");
	    printf( "postTime = : %12.4f ms \n ", elapsedTime ) ;
	    
	    cudaEventDestroy( start ) ;
	    cudaEventDestroy( stop  ) ;
	    
	    printf("nnz = %d\n", *nnz);
	    free(nnz);
	    
      //exit(0);
      printf("*************************************************\n");
	    printf("TotalPreTime = : %12.4f ms \n", preTime );
	    printf("TotalComputeTime = : %12.4f ms \n", computeTime );
	    printf( "TotalPostTime = : %12.4f ms \n ", elapsedTime ) ;
	    
	    printf( "TotalTime = : %12.4f ms \n ", preTime + computeTime + elapsedTime ) ;
      printf("\n");
	    
	    //printf("Transfering CSC_M to CSR_M is processing.........\n");


	    cudaFree(dev_tempPTR);
	    cudaFree(dev_tempIndex);
	    cudaFree(dev_tempData);

	    cudaFree(dev_J);
	    cudaFree(dev_JV);
	    cudaFree(dev_I);
	    cudaFree(dev_jPTR);
	    cudaFree(dev_iPTR);
	    
	    cudaFree(dev_tildeA);
	    cudaFree(dev_X);
	    
    }else{
	    printf("------Allocate GPU global memroy\n");
	    
	    cudaEventCreate( &start ) ;
      cudaEventCreate( &stop  ) ;
      cudaEventRecord( start, 0 ) ;
		  
	    double *dev_tildeA;

	    int *dev_J, *dev_jPTR;
	    int *dev_I, *dev_iPTR;
	    double *dev_X, *dev_JV;
	 
	    cudaMalloc((void**)&dev_J, sizeof(int) * devM->nCol * n2max) ;
	    cudaMalloc((void**)&dev_jPTR, sizeof(int) * devM->nCol) ;
	    cudaMalloc((void**)&dev_JV, sizeof(double) * devM->nCol * n2max) ;

	    cudaMalloc((void**)&dev_I, sizeof(int) * devM->nCol * n1max) ;
	    cudaMalloc((void**)&dev_iPTR, sizeof(int) * devM->nCol) ;
	    
	    cudaMalloc((void**)&dev_tildeA, sizeof(double) * devM->nCol * n1max * n2max) ;
	    cudaMalloc((void**)&dev_X, sizeof(double) * devM->nCol * n1max) ;

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
	      computeJ_SpMMv1<2><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, devM->nCol, dev_J, dev_jPTR, dev_JV, n2max);
	    }else if(n2max >2 && n2max <= 4){
	    	computeJ_SpMMv1<4><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, devM->nCol, dev_J, dev_jPTR, dev_JV, n2max);
	    }else if(n2max >4 && n2max <= 8){
	    	computeJ_SpMMv1<8><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, devM->nCol, dev_J, dev_jPTR, dev_JV, n2max);
	    }else if(n2max >8 && n2max <= 16){
	    	computeJ_SpMMv1<16><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, devM->nCol, dev_J, dev_jPTR, dev_JV, n2max);
	    }else if(n2max >16 && n2max <= 32){
	    	computeJ_SpMMv1<32><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, devM->nCol, dev_J, dev_jPTR, dev_JV, n2max);
	    }else if(n2max >32 && n2max <= 64){
	    	computeJ_SpMMv1<64><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, devM->nCol, dev_J, dev_jPTR, dev_JV, n2max);
	    }else if(n2max >64 && n2max <= 128){
	    	computeJ_SpMMv1<128><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, devM->nCol, dev_J, dev_jPTR, dev_JV, n2max);
	    }else{
	    	computeJ_SpMMv1<256><< <blocksPerGrid, threadsPerBlock>> >(devM->mPtr, devM->mIndex, devM->mData, devM->nCol, dev_J, dev_jPTR, dev_JV, n2max);
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
	      computeI_SpMMv1<2,256/2><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
	    }else if(n2max >2 && n2max <= 4){
	    	computeI_SpMMv1<4,256/4><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
	    }else if(n2max >4 && n2max <= 8){
	    	computeI_SpMMv1<8,256/8><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
	    }else if(n2max >8 && n2max <= 16){
	    	computeI_SpMMv1<16,256/16><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
	    }else if(n2max >16 && n2max <= 32){
	    	computeI_SpMMv1<32,256/32><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
	    }else if(n2max >32 && n2max <= 64){
	    	computeI_SpMMv1<64,256/64><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
	    }else if(n2max >64 && n2max <= 128){
	    	computeI_SpMMv1<128,256/128><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
	    }else{
	    	computeI_SpMMv1<256,256/256><< <blocksPerGrid, threadsPerBlock>> >(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J,dev_jPTR, n2max);
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
	        ComputeTildeACSR_SpMMv1<2><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
	        devA->mIndex, devM->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
	    }else if(n2max >2 && n2max <= 4){
	    	ComputeTildeACSR_SpMMv1<4><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
	        devA->mIndex, devM->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
	    }else if(n2max >4 && n2max <= 8){
	    	ComputeTildeACSR_SpMMv1<8><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
	        devA->mIndex, devM->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
	    }else if(n2max >8 && n2max <= 16){
	    	ComputeTildeACSR_SpMMv1<16><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
	        devA->mIndex, devM->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
	    }else if(n2max >16 && n2max <= 32){
	    	ComputeTildeACSR_SpMMv1<32><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
	        devA->mIndex, devM->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
	    }else if(n2max >32 && n2max <= 64){
	    	ComputeTildeACSR_SpMMv1<64><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
	        devA->mIndex, devM->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
	    }else if(n2max >64 && n2max <= 128){
	    	ComputeTildeACSR_SpMMv1<128><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
	        devA->mIndex, devM->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
	    }else{
	    	ComputeTildeACSR_SpMMv1<256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, devA->mData, devA->mPtr, 
	        devA->mIndex, devM->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
	    }
	 
	    cudaEventRecord( stop, 0 ) ;
	    cudaEventSynchronize( stop ) ;
	    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
	  
	    printf( "Time = : %8.4f ms \n ", elapsedTime ) ;
	    computeTime += elapsedTime;

	    cudaEventDestroy( start ) ;
	    cudaEventDestroy( stop  ) ;
	    
	    printf("-----------------Compute A * B\n");

	    cudaEventCreate( &start ) ;
      cudaEventCreate( &stop  ) ;
      cudaEventRecord( start, 0 ) ;
     
      if(n2max <= 2){
		    ComputeAB_iter_SpMMv1<2, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		        dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, devM->nCol);
		  }else if(n2max >2 && n2max <= 4){
		    ComputeAB_iter_SpMMv1<4, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		        dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, devM->nCol);
		  }else if(n2max >4 && n2max <= 8){
		    ComputeAB_iter_SpMMv1<8, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		        dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, devM->nCol);
		  }else if(n2max >8 && n2max <= 16){
		    ComputeAB_iter_SpMMv1<16, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		        dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, devM->nCol);
		  }else if(n2max >16 && n2max <= 32){
		    ComputeAB_iter_SpMMv1<32, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		        dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, devM->nCol);
		  }else if(n2max >32 && n2max <= 64){
		    ComputeAB_iter_SpMMv1<64, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		        dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, devM->nCol);
		  }else if(n2max >64 && n2max <= 128){
		    ComputeAB_iter_SpMMv1<128, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		        dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, devM->nCol);
		  }else if(n2max >128 && n2max <= 256){
		    ComputeAB_iter_SpMMv1<256, 256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		        dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, devM->nCol);
		  }else{
		    ComputeAB_iter_SpMMv1NoSharedM<256><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA, 
		        dev_iPTR, dev_jPTR, dev_JV, dev_X, n1max, n2max, devM->nCol);
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
	   
	    /*+++++++++++++++++Post-SpMM++++++++++++++++++++++++++++++*/
	    
	    printf("Post-SpMM is processing.........................\n");
	    cudaEventCreate( &start ) ;
      cudaEventCreate( &stop  ) ;
      cudaEventRecord( start, 0 ) ;
		        
		        
      if(n1max <= 2){
      	blocksPerGrid = (devM->nCol - 1)/(threadsPerBlock/2) + 1;
		    SparistyGuass_SpMMv2<2><< <blocksPerGrid, threadsPerBlock>> >(dev_iPTR, 
		        dev_X, dev_I, dev_NZUPP, n1max, devM->nCol);
		  }else if(n1max >2 && n1max <= 4){
		  	blocksPerGrid = (devM->nCol - 1)/(threadsPerBlock/4) + 1;
		    SparistyGuass_SpMMv2<4><< <blocksPerGrid, threadsPerBlock>> >(dev_iPTR, 
		        dev_X, dev_I, dev_NZUPP, n1max, devM->nCol);
		  }else if(n1max >4 && n1max <= 8){
		  	blocksPerGrid = (devM->nCol - 1)/(threadsPerBlock/8) + 1;
		    SparistyGuass_SpMMv2<8><< <blocksPerGrid, threadsPerBlock>> >(dev_iPTR, 
		        dev_X, dev_I, dev_NZUPP, n1max, devM->nCol);
		  }else if(n1max >8 && n1max <= 16){
		  	blocksPerGrid = (devM->nCol - 1)/(threadsPerBlock/16) + 1;
		    SparistyGuass_SpMMv2<16><< <blocksPerGrid, threadsPerBlock>> >(dev_iPTR, 
		        dev_X, dev_I, dev_NZUPP, n1max, devM->nCol);
		  }else if(n1max >16 && n1max <= 32){
		  	blocksPerGrid = (devM->nCol - 1)/(threadsPerBlock/32) + 1;
		    SparistyGuass_SpMMv2<32><< <blocksPerGrid, threadsPerBlock>> >(dev_iPTR, 
		        dev_X, dev_I, dev_NZUPP, n1max, devM->nCol);
		  }else if(n1max >32 && n1max <= 64){
		  	blocksPerGrid = (devM->nCol - 1)/(threadsPerBlock/64) + 1;
		    SparistyGuass_SpMMv2<64><< <blocksPerGrid, threadsPerBlock>> >(dev_iPTR, 
		        dev_X, dev_I, dev_NZUPP, n1max, devM->nCol);
		  }else if(n1max >64 && n1max <= 128){
		  	blocksPerGrid = (devM->nCol - 1)/(threadsPerBlock/128) + 1;
		    SparistyGuass_SpMMv2<128><< <blocksPerGrid, threadsPerBlock>> >(dev_iPTR, 
		        dev_X, dev_I, dev_NZUPP, n1max, devM->nCol);
		  }else{
		  	blocksPerGrid = (devM->nCol - 1)/(threadsPerBlock/256) + 1;
		    SparistyGuass_SpMMv2<256><< <blocksPerGrid, threadsPerBlock>> >(dev_iPTR, 
		        dev_X, dev_I, dev_NZUPP, n1max, devM->nCol);
		  }
      
//      int *iPTR = (int*)malloc(sizeof(int)*devM->nCol);
//      cudaMemcpyAsync( iPTR, dev_iPTR, devM->nCol * sizeof( int ), cudaMemcpyDeviceToHost, 0 ) ; 
//      
//      int *IIndex = (int*)malloc(sizeof(int)*n1max*devM->nCol);
//      cudaMemcpyAsync( IIndex, dev_I, devM->nCol * n1max * sizeof( int ), cudaMemcpyDeviceToHost, 0 ) ; 
//      
//      double *idata = (double*)malloc(sizeof(double)*n1max*devM->nCol);
//      cudaMemcpyAsync( idata, dev_X, devM->nCol * n1max * sizeof( double ), cudaMemcpyDeviceToHost, 0 ) ; 
//
//    
//    
//    
//    //for(int i = 0 ; i < devM->nCol; i++){
//    //for(int i = 0 ; i < 10; i++){
//    for(int i = devM->nCol -10 ; i < devM->nCol; i++){
//    	for(int j = 0; j < iPTR[i]; j++){
//    	  printf("%d ", IIndex[i*n1max + j]);
//    	}
//    	printf("\n");
//    }
//    
//    
//     //for(int i = 0 ; i < devM->nCol; i++){
//     //for(int i = 0 ; i < 10; i++){
//     for(int i = devM->nCol -10 ; i < devM->nCol; i++){
//    	for(int j = 0; j < iPTR[i]; j++){
//    	  printf("%lf ", idata[i*n1max +j]);
//    	}
//    	printf("\n");
//    }
      
     //exit(0);
      
      devC->n = devM->nCol;
	    devC->nCol = devM->nCol;
	    devC->nRow = devM->nCol;
	    cudaMalloc((void**)&devC->mPtr, sizeof(int) * (devC->n + 1)) ;
	    cudaMemcpyAsync( devC->mPtr + 1, dev_iPTR, devM->nCol * sizeof( int ), cudaMemcpyDeviceToDevice, 0 ) ;
	    

	    int threadsPerBlock1 = 1024;
	    int blocksPerGrid1 = 1;
	    summation_SpMMv1<<<blocksPerGrid1, threadsPerBlock1>>>(devC->mPtr, devM->nCol+1);
	    
	    int *nnz = (int*)malloc(sizeof(int));
		  cudaMemcpyAsync( nnz, devC->mPtr+devC->n, sizeof( int ), cudaMemcpyDeviceToHost, 0 ) ; 
		  
		  
			
//			for(int i = 0; i < (devM->nCol +1); i++){
//				printf("i=%d, p[%d]=%d\n", i, i, cpuPtr[i]);
//			}
	    
	    devC->nonzeroes = *nnz ;
	    cudaMalloc((void**)&devC->mIndex, sizeof(int) * devC->nonzeroes ) ;
      cudaMalloc((void**)&devC->mData, sizeof(double) * devC->nonzeroes) ;
      
      if(n1max < 3){
		   	 //threadsPerBlock = 256;
			   blocksPerGrid = (devC->n - 1)/(threadsPerBlock/2) + 1;
			   WriteABIntoC_SpMMv1<2><<<blocksPerGrid, threadsPerBlock>>>(dev_I, dev_X, devC->mPtr, devC->mIndex, devC->mData, n1max, devC->n);
			}else if(n1max >= 3 && n1max < 6){
				 //threadsPerBlock = 256;
			   blocksPerGrid = (devC->n - 1)/(threadsPerBlock/4) + 1;
				 WriteABIntoC_SpMMv1<4><<<blocksPerGrid, threadsPerBlock>>>(dev_I, dev_X, devC->mPtr, devC->mIndex, devC->mData, n1max, devC->n);
			}else if(n1max >= 6 && n1max < 12){
				 //threadsPerBlock = 256;
			   blocksPerGrid = (devC->n - 1)/(threadsPerBlock/8) + 1;
				 WriteABIntoC_SpMMv1<8><<<blocksPerGrid, threadsPerBlock>>>(dev_I, dev_X, devC->mPtr, devC->mIndex, devC->mData, n1max, devC->n);
			}else if(n1max >= 12 && n1max < 24){
				 //threadsPerBlock = 256;
			   blocksPerGrid = (devC->n - 1)/(threadsPerBlock/16) + 1;
				 WriteABIntoC_SpMMv1<16><<<blocksPerGrid, threadsPerBlock>>>(dev_I, dev_X, devC->mPtr, devC->mIndex, devC->mData, n1max, devC->n);
			}else {
				 //threadsPerBlock = 256;
			   blocksPerGrid = (devC->n - 1)/(threadsPerBlock/32) + 1;
				 WriteABIntoC_SpMMv1<32><<<blocksPerGrid, threadsPerBlock>>>(dev_I, dev_X, devC->mPtr, devC->mIndex, devC->mData, n1max, devC->n);
			}
	    
	    cudaEventRecord( stop, 0 ) ;
	    cudaEventSynchronize( stop ) ;
	    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
	  
	    printf("\n");
	    printf( "postTime = : %12.4f ms \n ", elapsedTime ) ;
	    
	    cudaEventDestroy( start ) ;
	    cudaEventDestroy( stop  ) ;
	    
	    printf("nnz = %d\n", *nnz);
	    
	    free(nnz);
	    
	    //exit(0);
	    
	    printf("******************The SpMM time**********************************\n");
	    printf("TotalPreTime = : %12.4f ms \n", preTime );
	    printf("TotalComputeTime = : %12.4f ms \n", computeTime );
	    printf( "TotalPostTime = : %12.4f ms \n ", elapsedTime ) ;
	    
	    printf( "TotalTime = : %12.4f ms \n ", preTime + computeTime + elapsedTime ) ;
	    printf("\n");
	    

      cudaFree(dev_tempPTR);
	    cudaFree(dev_tempIndex);
	    cudaFree(dev_tempData);
      
	    cudaFree(dev_J);
	    cudaFree(dev_I);
	    cudaFree(dev_jPTR);
	    cudaFree(dev_iPTR);
	    cudaFree(dev_JV);
	    
	    cudaFree(dev_tildeA);
	    cudaFree(dev_X);
	  }
	  
	  return preTime + computeTime + elapsedTime;
}