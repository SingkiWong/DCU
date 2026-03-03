#include "hip/hip_runtime.h"
/* -----------------------------------------------------------------------------
    The programming is licensed to you under the ZJUT Consortium license:
    Copyright (C)  2019  Jiaquan Gao 
	  E-mail: 73025@njnu.edu.cn

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
* ----------------------------------------------------------------------------*/

/* ----------------------------------------------------------------------------
 *  This .h file is used to format conversion
** ---------------------------------------------------------------------------*/

#ifndef CUFORMATCONVERSION_H
#define CUFORMATCONVERSION_H

#include "dataType.h"

using namespace std;

/* ----------------------------------------------------------------------------
 * ++The method is used to transfer CSC into CSR format (GPU)
 * @parameters
 * @CSR_A:  the matrix with a compressed storage of rows (CSR)
 * @CSC_A:  the matrix with a compressed storage of columns (CSC)
** ---------------------------------------------------------------------------*/
template<unsigned int WarpSize>
__global__ void computeNonzerosPerRow(int *cscIndex, int *cscPtr, int *csrPtr, int n){
	
	int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
  int offset = blockDim.x / WarpSize * gridDim.x;
  int warp_id = gid / WarpSize; //global warp index
  int lane = gid & (WarpSize-1);// index of threads in warp
  
  int col, col_s, col_e, bv;
  int idx;
  
  for(col = warp_id; col < n; col += offset)
  {
  	col_s = cscPtr[col];
  	col_e = cscPtr[col + 1];
  	
  	bv = col_e - col_s;
  	for(int i = lane; i < bv; i += WarpSize){
  		idx = cscIndex[i + col_s];
  		atomicAdd(&csrPtr[idx+1], 1);
  	}
  }
}

__global__ void InitializeArraysValues(int *csrPtr, int value, int m){
	int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
  int offset = blockDim.x * gridDim.x;
  
  for(int row = gid; row < (m + 1); row += offset){
  	csrPtr[row] = value;
  }
}

__global__ void summation(int *csrPtr, int m){
	int gid = threadIdx.x; //global index
  int offset = blockDim.x ;
  
  int size = m/blockDim.x + 1;
  int z ;
  
  //compute blockDim.x number of csrPtr per thread
  for(int i = gid ; i < size ; i += offset){
  	
  	z = min(m,(i+1)*blockDim.x);
  	for(int j = i * blockDim.x; j < z - 1; j++){
  		csrPtr[j+1] += csrPtr[j];
  	}
  }
  
  __syncthreads() ;
  
  //compute blockDim.x number of csrPtr in parallel
  for(int i = 1; i < size -1; i++){
  	csrPtr[threadIdx.x + i * blockDim.x] += csrPtr[i * blockDim.x -1];
  	__syncthreads() ;
  }
  
  z = threadIdx.x + (size -1) * blockDim.x;
  if(size > 1 && z < m){
  	csrPtr[z] += csrPtr[(size -1) * blockDim.x -1];
  }
  
}

template<unsigned int WarpSize>
__global__ void FindRowIndexandValues(double *cscData, int *cscIndex, int *cscPtr, 
double *csrData, int *csrIndex, int *csrPtr, int n, int *atomic){
	int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
  int offset = blockDim.x / WarpSize * gridDim.x;
  int warp_id = gid / WarpSize; //global warp index
  int lane = gid & (WarpSize-1);// index of threads in warp
  
  int col, col_s, col_e, bv;
  int idx, row_s;
  
  for(col = warp_id; col < n; col += offset)
  {
  	col_s = cscPtr[col];
  	col_e = cscPtr[col + 1];
  	
  	bv = col_e - col_s;
  	for(int i = lane; i < bv; i += WarpSize){
  		idx = cscIndex[i + col_s];
  		row_s = csrPtr[idx];
  		
  		int tv = atomicAdd(&atomic[idx], 1);
  		
  		csrIndex[row_s + tv] = col;
  		csrData[row_s + tv] = cscData[i + col_s];
  	}
  }
}

template<unsigned int WarpSize>
__global__ void SortInCSC2CSR(double *csrData, int *csrIndex, int *csrPtr, int n){
	int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
  int offset = blockDim.x / WarpSize * gridDim.x;
  int warp_id = gid / WarpSize; //global warp index
  int lane = gid & (WarpSize-1);// index of threads in warp
  
  int col;
  int row_s, len, bV1;
  double bV2;
  
 
  for(col = warp_id; col < n; col += offset)
  {	
  	len = csrPtr[col+1] - csrPtr[col];
  	row_s = csrPtr[col];
  	//printf("col=%d,len = %d\n", col, len);
  	//sort using bubbit
  	for( int i = 1; i <= len; i++)
		{
			if( i%2 == 1){
				for(int j1 = lane; j1 < len; j1+=WarpSize)
				{
					if(((j1+1)%2==1)&&(csrIndex[row_s+j1]> csrIndex[row_s+j1+1])&&(j1+1)<len)
					{
						bV1 = csrIndex[row_s+j1] ;
						csrIndex[row_s+j1]=csrIndex[row_s+j1+1] ;
						csrIndex[row_s+j1+1] = bV1 ; 
						
						bV2 = csrData[row_s+j1] ;
						csrData[row_s+j1]=csrData[row_s+j1+1] ;
						csrData[row_s+j1+1] = bV2 ; 
					}
				}
				__syncthreads() ;
			}
			else{
				for( int j2 = lane; j2 < len; j2+=WarpSize)
				{
					if(((j2+1)%2==0)&&(csrIndex[row_s+j2]>csrIndex[row_s+j2+1])&&(j2+1)<len)
					{
						bV1 = csrIndex[row_s+j2] ;
						csrIndex[row_s+j2] = csrIndex[row_s+j2+1] ;
						csrIndex[row_s+j2+1] = bV1 ;
						
						bV2 = csrData[row_s+j2] ;
						csrData[row_s+j2] = csrData[row_s+j2+1] ;
						csrData[row_s+j2+1] = bV2 ;
					}
				}
				__syncthreads() ;
			}
		} 
		
		
  }
}

template<unsigned int N2SIZE>
__global__ void cuComputeN2MAX(int *aPtr, int nCol, int *n2max){
	
	__shared__ int n2_s[N2SIZE];
	int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
  int offset = blockDim.x * gridDim.x;
  int tid = threadIdx.x;
 
	int col, col_s, col_e, tV; 
	int i;
  
  int value = 0; 
  for(col = gid; col < nCol; col += offset)
  {
     col_s = aPtr[col];
     col_e = aPtr[col+1];
     tV = col_e - col_s ;
     if( value < tV) value = tV;
    
  }//for col
  
  n2_s[tid] = value;
  
  __syncthreads();
      
  i =  N2SIZE/2;
  while( tid < i )
  {
    if(n2_s[tid] < n2_s[tid + i])  n2_s[tid] =  n2_s[tid + i];
    i = i/2;
	  __syncthreads();
  }
     
  if(tid == 0) n2max[blockIdx.x] = n2_s[0];
	
}

template<unsigned int N2SIZE>
__global__ void cuComputeN2MAX2(int *aPtr, int nCol, int *n2max){
	
	__shared__ int n2_s[N2SIZE];
	int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
  int offset = blockDim.x * gridDim.x;
  int tid = threadIdx.x;
 
	int col, tV; 
	int i;
  
  int value = 0; 
  for(col = gid; col < nCol; col += offset)
  {
     tV = aPtr[col+1];
     if( value < tV) value = tV;
    
  }//for col
  
  n2_s[tid] = value;
  
  __syncthreads();
      
  i =  N2SIZE/2;
  while( tid < i )
  {
    if(n2_s[tid] < n2_s[tid + i])  n2_s[tid] =  n2_s[tid + i];
    i = i/2;
	  __syncthreads();
  }
     
  if(tid == 0) n2max[blockIdx.x] = n2_s[0];
	
}

__global__ void WritePtrIntoSparsityA(int *csrAPtr, int *cscSparsityAPtr, int n){
	int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
  int offset = blockDim.x * gridDim.x;
  
  int i;
  
  for(i = gid; i <= n; i += offset)
  {
  	cscSparsityAPtr[i] = csrAPtr[i];
  	//printf("i = %d, ptr=%d\n",i, cscSparsityAPtr[i]);
  }
}

template<unsigned int WarpSize>
__global__ void WriteATIntoSparsityA(int *csrAIndex, double *csrAData, int *csrAPtr, int *csrSparsityAIndex, double *csrSparsityAData, int n){
	int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
  int offset = blockDim.x / WarpSize * gridDim.x;
  int warp_id = gid / WarpSize; //global warp index
  int lane = gid & (WarpSize-1);// index of threads in warp
  
  int col, col_s, col_e, bv;
  
  for(col = warp_id; col < n; col += offset)
  {
  	col_s = csrAPtr[col];
  	col_e = csrAPtr[col + 1];
  	
  	bv = col_e - col_s;
  	for(int i = lane; i < bv; i += WarpSize){
  		csrSparsityAIndex[col_s + i] = csrAIndex[col_s + i];
  		csrSparsityAData[col_s + i] = abs(csrAData[col_s + i]);
  	}
  }
}

template<unsigned int WarpSize>
__global__ void WriteATIntoSparsityAIn1(int *devTempIndex, double *devTempData, int *csrSparsityAPTR, int *csrSparsityAIndex, double *csrSparsityAData, int N2MAX, int n)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
  int offset = blockDim.x / WarpSize * gridDim.x;
  int warp_id = gid / WarpSize; //global warp index
  int lane = gid & (WarpSize-1);// index of threads in warp
  
  int col, col_s, col_e, bv, COLJ;
  
  for(col = warp_id; col < n; col += offset)
  {
  	
  	col_s = csrSparsityAPTR[col];
  	col_e = csrSparsityAPTR[col + 1];
  	
  	bv = col_e - col_s;
  	COLJ = N2MAX * col;
  	for(int i = lane; i < bv; i += WarpSize){
  		csrSparsityAIndex[col_s + i] = devTempIndex[COLJ + i];
  		csrSparsityAData[col_s + i] = abs(devTempData[COLJ + i]);
  	}
  }
}

template<unsigned int WarpSize>
__global__ void WriteATIntoSparsityAIn2(int *csrAIndex, double *csrAData, int *csrAPtr, 
                int *devTempIndex, double *devTempData, int *devTempPTR, int *atomic, int N2MAX, int n)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
  int offset = blockDim.x / WarpSize * gridDim.x;
  int warp_id = gid / WarpSize; //global warp index
  int lane = gid & (WarpSize-1);// index of threads in warp
  
  int col, col_s, col_e, bv, bV1, COLJ, tv;
  int idx, isExt;
  
  for(col = warp_id; col < n; col += offset)
  {
  	bV1 = devTempPTR[col];
  	col_s = csrAPtr[col];
  	col_e = csrAPtr[col + 1];
  	
  	COLJ = N2MAX * col;
  	atomic[col] = 0;
  	
  	bv = col_e - col_s;
  	//if(lane == 0) printf("col = %d, bv=%d\n", col, bv);
  	for(int i = lane; i < bv; i += WarpSize){
  		idx = csrAIndex[i + col_s];
  		isExt = 0;
  		
  		for(int j = 0 ; j < bV1; j++){
  			if( idx == devTempIndex[COLJ + j] ){
  				isExt = 1;
  				break;
  			}
  	  }
  		
  		if(isExt == 0){
  		
  		  tv = atomicAdd(&atomic[col], 1);
  		  //printf("tv =%d\n",tv);
  		  devTempIndex[COLJ + bV1 + tv] = idx;
  		  devTempData[COLJ + bV1 + tv] = abs(csrAData[i + col_s]);
  	  }
  	}
  	
  	__syncthreads();
  	
  	devTempPTR[col] += atomic[col];
  	//if(lane == 0) printf("col = %d, num=%d\n", col, atomic[col]);
  }
}

template<unsigned int WarpSize>
__global__ void ComputeTmpDataIn1(int *cscAIndex, double *cscAData, int *cscAPtr, int *devTempIndex, double *devTempData, int *devTempPTR, int N2MAX, int n)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
  int offset = blockDim.x / WarpSize * gridDim.x;
  int warp_id = gid / WarpSize; //global warp index
  int lane = gid & (WarpSize-1);// index of threads in warp
  
  int col, col_s, col_e, bv;
  
  for(col = warp_id; col < n; col += offset)
  {
  	col_s = cscAPtr[col];
  	col_e = cscAPtr[col + 1];
  	
  	bv = col_e - col_s;
  	devTempPTR[col] = bv;
  	for(int i = lane; i < bv; i += WarpSize){
  		devTempIndex[col * N2MAX + i] = cscAIndex[col_s + i];
  		devTempData[col * N2MAX  + i] = cscAData[col_s + i];
  	}
  }
}

__global__ void JudgeE(int *devTempIndex, double *devTempData, int *devTempPTR, int N2MAX, int n)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
  int offset = blockDim.x * gridDim.x;
  
  int col, bv, COLJ, flag, i;
  
  for(col = gid; col < n; col += offset)
  {
  	bv = devTempPTR[col];
  	COLJ = col * N2MAX;
  	flag = 0;
  	for(i = 0; i < bv; i++){
  		//printf("gid=%d,col ==%d, v=%d\n", gid, col, devTempIndex[COLJ + i]);
  		if(devTempIndex[COLJ + i] == col){
  			flag = 1;
  			break;
  		}
//      if(gid == 1) {
//      	printf("gid=%d,col ==%d, v=%d\n", gid, col, devTempIndex[COLJ + i]);
//      }
  	}
  	
  	if(flag == 0){
  		devTempPTR[col] += 1;
  		devTempIndex[COLJ + bv] = col;
  		devTempData[COLJ + bv] = 1.0;
  	}
  }
  
}

template<unsigned int SHARED_SIZE>
__global__ void  cuDcomputeNNZIn1(int *jPTR, int eGrid, int *nnz)
{
		__shared__ int nz_s[SHARED_SIZE];
		int tid = threadIdx.x; //global index
	  int offset = blockDim.x ;
	  int col;
	  
	  int nV = 0;
	  for(col = tid; col < eGrid; col += offset){
	  	nV += jPTR[col];
	  	//printf("col=%d, jPTR=%d\n", col, jPTR[col]);
	  }
	  
	  nz_s[tid] = nV;
	  __syncthreads();
	  
	  int i =  SHARED_SIZE/2;
	  while( tid < i )
	  {
	    nz_s[tid] +=  nz_s[tid + i];
	    i = i/2;
		  __syncthreads();
	  }
	     
	  if(tid == 0) *nnz = nz_s[0];
  
}

void cuCSC2CSR(int m, int n, int nnz, double *cscData, int *cscIndex, int *cscPtr, double *csrData, int *csrIndex, int *csrPtr)
{
	int *atomic ;
	//hipMalloc( (void**)&atomic, sizeof( int ) * n ) ;
	
	int threadsPerBlock, blocksPerGrid;
	
	threadsPerBlock = 256;
	blocksPerGrid = 15 * 8 * 4;
  int *dev_maxV;
  int *maxV = (int*)malloc(sizeof(int) * blocksPerGrid);
  hipMalloc((void**)&dev_maxV, sizeof(int) * blocksPerGrid) ;
  
  cuComputeN2MAX<256><<<blocksPerGrid, threadsPerBlock>>>(cscPtr, n, dev_maxV);
  
  hipMemcpyAsync( maxV, dev_maxV, blocksPerGrid * sizeof( int ), hipMemcpyDeviceToHost, 0 ) ; 
  int n2max = maxV[0];
  for(int i = 1; i < blocksPerGrid; i++){
      if(n2max < maxV[i]) n2max = maxV[i];
  }
  
  printf("n2max = %d\n", n2max);
	
	
	//Initailize cscPtr = 0
	threadsPerBlock = 256;
	blocksPerGrid = (m - 1)/threadsPerBlock + 1;
	InitializeArraysValues<<<blocksPerGrid, threadsPerBlock>>>(csrPtr, 0, m);
  
	//compute the nonzero number per row
	if(n2max < 3){
		threadsPerBlock = 256;
	  blocksPerGrid = (n - 1)/(threadsPerBlock/2) + 1;
	  computeNonzerosPerRow<2><<<blocksPerGrid, threadsPerBlock>>>(cscIndex, cscPtr, csrPtr, n);
	}else if(n2max >= 3 && n2max < 6){
		threadsPerBlock = 256;
	  blocksPerGrid = (n - 1)/(threadsPerBlock/4) + 1;
		computeNonzerosPerRow<4><<<blocksPerGrid, threadsPerBlock>>>(cscIndex, cscPtr, csrPtr, n);
	}else if(n2max >= 6 && n2max < 12){
		threadsPerBlock = 256;
	  blocksPerGrid = (n - 1)/(threadsPerBlock/8) + 1;
		computeNonzerosPerRow<8><<<blocksPerGrid, threadsPerBlock>>>(cscIndex, cscPtr, csrPtr, n);
	}else if(n2max >= 12 && n2max < 24){
		threadsPerBlock = 256;
	  blocksPerGrid = (n - 1)/(threadsPerBlock/16) + 1;
		computeNonzerosPerRow<16><<<blocksPerGrid, threadsPerBlock>>>(cscIndex, cscPtr, csrPtr, n);
	}else {
		threadsPerBlock = 256;
	  blocksPerGrid = (n - 1)/(threadsPerBlock/32) + 1;
		computeNonzerosPerRow<32><<<blocksPerGrid, threadsPerBlock>>>(cscIndex, cscPtr, csrPtr, n);
	}
	
	blocksPerGrid = 15 * 8 * 4;
	cuComputeN2MAX2<256><<<blocksPerGrid, threadsPerBlock>>>(csrPtr, m, dev_maxV);
  
  hipMemcpyAsync( maxV, dev_maxV, blocksPerGrid * sizeof( int ), hipMemcpyDeviceToHost, 0 ) ; 
  int n2rmax = maxV[0];
  for(int i = 1; i < blocksPerGrid; i++){
      if(n2rmax < maxV[i]) n2rmax = maxV[i];
  }
	
	hipFree(dev_maxV);
  free(maxV);
  
  printf("n2rmax = %d\n", n2rmax);
	
	//summation
//	if(m <= 256 * 128){
//		threadsPerBlock = 128;
//		blocksPerGrid = 1;
//		summation<<<blocksPerGrid, threadsPerBlock>>>(csrPtr, m+1);
//	}else if(m > 256 * 128 && m <= 256 * 512){
//		threadsPerBlock = 256;
//		blocksPerGrid = 1;
//		summation<<<blocksPerGrid, threadsPerBlock>>>(csrPtr, m+1);
//	}else if(m > 256 * 512 && m <= 512 * 1024){
//		threadsPerBlock = 512;
//		blocksPerGrid = 1;
//		summation<<<blocksPerGrid, threadsPerBlock>>>(csrPtr, m+1);
//	}else{
//		threadsPerBlock = 1024;
//		blocksPerGrid = 1;
//		summation<<<blocksPerGrid, threadsPerBlock>>>(csrPtr, m+1);
//	}
	threadsPerBlock = 1024;
	blocksPerGrid = 1;
	summation<<<blocksPerGrid, threadsPerBlock>>>(csrPtr, m+1);
	
//	int *cpuPtr = (int*)malloc(sizeof(int) * (m+1));
//	hipMemcpyAsync( cpuPtr, csrPtr, (m+1) * sizeof( int ), hipMemcpyDeviceToHost, 0 ) ; 
//	
//	for(int i = 0; i < (m +1); i++){
//		printf("i=%d, p[%d]=%d\n", i, i, cpuPtr[i]);
//	}

   //Initailize atomic = 0
   hipMalloc( (void**)&atomic, sizeof( int ) * m ) ;
	 threadsPerBlock = 256;
	 blocksPerGrid = (n - 1)/threadsPerBlock + 1;
	 InitializeArraysValues<<<blocksPerGrid, threadsPerBlock>>>(atomic, 0, m);
   
   //store the element and vlaues and sort in parallel
   if(n2rmax < 3){
   	 threadsPerBlock = 256;
	   blocksPerGrid = (n - 1)/(threadsPerBlock/2) + 1;
	   FindRowIndexandValues<2><<<blocksPerGrid, threadsPerBlock>>>(cscData, cscIndex, cscPtr, csrData, csrIndex, csrPtr, n, atomic);
	   SortInCSC2CSR<2><<<blocksPerGrid, threadsPerBlock>>>(csrData, csrIndex, csrPtr, n);
	}else if(n2rmax >= 3 && n2rmax < 6){
		 threadsPerBlock = 256;
	   blocksPerGrid = (n - 1)/(threadsPerBlock/4) + 1;
		 FindRowIndexandValues<4><<<blocksPerGrid, threadsPerBlock>>>(cscData, cscIndex, cscPtr, csrData, csrIndex, csrPtr, n, atomic);
		 SortInCSC2CSR<4><<<blocksPerGrid, threadsPerBlock>>>(csrData, csrIndex, csrPtr, n);
	}else if(n2rmax >= 6 && n2rmax < 12){
		 threadsPerBlock = 256;
	   blocksPerGrid = (n - 1)/(threadsPerBlock/8) + 1;
		 FindRowIndexandValues<8><<<blocksPerGrid, threadsPerBlock>>>(cscData, cscIndex, cscPtr, csrData, csrIndex, csrPtr, n, atomic);
		 SortInCSC2CSR<8><<<blocksPerGrid, threadsPerBlock>>>(csrData, csrIndex, csrPtr, n);
	}else if(n2rmax >= 12 && n2rmax < 24){
		 threadsPerBlock = 256;
	   blocksPerGrid = (n - 1)/(threadsPerBlock/16) + 1;
		 FindRowIndexandValues<16><<<blocksPerGrid, threadsPerBlock>>>(cscData, cscIndex, cscPtr, csrData, csrIndex, csrPtr, n, atomic);
		 SortInCSC2CSR<16><<<blocksPerGrid, threadsPerBlock>>>(csrData, csrIndex, csrPtr, n);
	}else {
		 threadsPerBlock = 256;
	   blocksPerGrid = (n - 1)/(threadsPerBlock/32) + 1;
		 FindRowIndexandValues<32><<<blocksPerGrid, threadsPerBlock>>>(cscData, cscIndex, cscPtr, csrData, csrIndex, csrPtr, n, atomic);
		 SortInCSC2CSR<32><<<blocksPerGrid, threadsPerBlock>>>(csrData, csrIndex, csrPtr, n);
	}
//		 threadsPerBlock = 256;
//	   blocksPerGrid = (n - 1)/(threadsPerBlock/32) + 1;
//		 FindRowIndexandValues<32><<<blocksPerGrid, threadsPerBlock>>>(cscData, cscIndex, cscPtr, csrData, csrIndex, csrPtr, n, atomic);
//		 SortInCSC2CSR<32><<<blocksPerGrid, threadsPerBlock>>>(csrData, csrIndex, csrPtr, n);
}

/* ----------------------------------------------------------------------------
 * ++The method is used to Get Sparsity pattern 
 * @parameters
 * @CSR_A:  the matrix with a compressed storage of rows (CSR)
 * @CSC_A:  the matrix with a compressed storage of columns (CSC)
 * @CSC_SparsityA: the matrix with a compressed storage of columns (CSC)
 * @flag 
 *     0 --- A^T
 *     1 --- E + A
 *     2 --- E + A + A^T
** ---------------------------------------------------------------------------*/
void GetSparsityPattern(CSC_Matrix *CSC_A, CSR_Matrix *CSR_A, CSC_Matrix *CSC_SparsityA, int flag)
{
	int threadsPerBlock, blocksPerGrid;
	

   if( flag == 0){//A^T
  		CSC_SparsityA->nRow = CSC_A->n ;
			CSC_SparsityA->nCol = CSC_A->n ;
			CSC_SparsityA->n = CSC_A->n ;
			CSC_SparsityA->nonzeroes = CSC_A->nonzeroes ;
			
			threadsPerBlock = 256;
			blocksPerGrid = 15 * 8 * 4;
  		int *dev_maxV;
  		int *maxV = (int*)malloc(sizeof(int) * blocksPerGrid);
  		hipMalloc((void**)&dev_maxV, sizeof(int) * blocksPerGrid) ;
  
  		cuComputeN2MAX<256><<<blocksPerGrid, threadsPerBlock>>>(CSR_A->mPtr, CSR_A->n, dev_maxV);
  
  		hipMemcpyAsync( maxV, dev_maxV, blocksPerGrid * sizeof( int ), hipMemcpyDeviceToHost, 0 ) ; 
  		int n2max = maxV[0];
  		for(int i = 1; i < blocksPerGrid; i++){
     		 if(n2max < maxV[i]) n2max = maxV[i];
  		}
  
  		printf("n2max = %d\n", n2max);
  		
  		printf("n=%d,nonzeroes=%d\n", CSC_SparsityA->n, CSC_SparsityA->nonzeroes);
  		
  		hipMalloc((void**)&CSC_SparsityA->mPtr, sizeof(int) * (CSC_SparsityA->n + 1)) ;
      hipMalloc((void**)&CSC_SparsityA->mIndex, sizeof(int) * CSC_SparsityA->nonzeroes ) ;
      hipMalloc((void**)&CSC_SparsityA->mData, sizeof(double) * CSC_SparsityA->nonzeroes) ;
     
      
      /*Put A^T into CSC_SparsityA */
      threadsPerBlock = 256;
      blocksPerGrid = (CSC_SparsityA->n - 1)/(threadsPerBlock) + 1;
      WritePtrIntoSparsityA<<<blocksPerGrid, threadsPerBlock>>>(CSR_A->mPtr, CSC_SparsityA->mPtr, CSC_SparsityA->n);
      
      
      
      if(n2max < 3){
		   	 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/2) + 1;
			   WriteATIntoSparsityA<2><<<blocksPerGrid, threadsPerBlock>>>(CSR_A->mIndex, CSR_A->mData, CSR_A->mPtr, CSC_SparsityA->mIndex, CSC_SparsityA->mData, CSC_SparsityA->n);
			}else if(n2max >= 3 && n2max < 6){
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/4) + 1;
				 WriteATIntoSparsityA<4><<<blocksPerGrid, threadsPerBlock>>>(CSR_A->mIndex, CSR_A->mData, CSR_A->mPtr, CSC_SparsityA->mIndex, CSC_SparsityA->mData, CSC_SparsityA->n);
			}else if(n2max >= 6 && n2max < 12){
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/8) + 1;
				 WriteATIntoSparsityA<8><<<blocksPerGrid, threadsPerBlock>>>(CSR_A->mIndex, CSR_A->mData, CSR_A->mPtr, CSC_SparsityA->mIndex, CSC_SparsityA->mData, CSC_SparsityA->n);
			}else if(n2max >= 12 && n2max < 24){
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/16) + 1;
				 WriteATIntoSparsityA<16><<<blocksPerGrid, threadsPerBlock>>>(CSR_A->mIndex, CSR_A->mData, CSR_A->mPtr, CSC_SparsityA->mIndex, CSC_SparsityA->mData, CSC_SparsityA->n);
			}else {
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/32) + 1;
				 WriteATIntoSparsityA<32><<<blocksPerGrid, threadsPerBlock>>>(CSR_A->mIndex, CSR_A->mData, CSR_A->mPtr, CSC_SparsityA->mIndex, CSC_SparsityA->mData, CSC_SparsityA->n);
			}
			
  		
   }else if( flag == 1){//E + A
  		threadsPerBlock = 256;
			blocksPerGrid = 15 * 8 * 4;
  		int *dev_maxV;
  		int *maxV = (int*)malloc(sizeof(int) * blocksPerGrid);
  		hipMalloc((void**)&dev_maxV, sizeof(int) * blocksPerGrid) ;
  
  		cuComputeN2MAX<256><<<blocksPerGrid, threadsPerBlock>>>(CSC_A->mPtr, CSC_A->n, dev_maxV);
  
  		hipMemcpyAsync( maxV, dev_maxV, blocksPerGrid * sizeof( int ), hipMemcpyDeviceToHost, 0 ) ; 
  		int n2max = maxV[0];
  		for(int i = 1; i < blocksPerGrid; i++){
     		 if(n2max < maxV[i]) n2max = maxV[i];
  		}
  
  		printf("n2max = %d\n", n2max);
  		
  		int *devTempIndex;
  		double *devTempData;
  		int *devTempPTR;
  		
  		hipMalloc((void**)&devTempPTR, sizeof(int) * CSC_A->n) ;
      hipMalloc((void**)&devTempIndex, sizeof(int) * CSC_A->n * (n2max + 1) ) ;
      hipMalloc((void**)&devTempData, sizeof(double) * CSC_A->n * (n2max + 1)) ;
  		
  		if(n2max < 3){
		   	 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/2) + 1;
			   ComputeTmpDataIn1<2><<<blocksPerGrid, threadsPerBlock>>>(CSC_A->mIndex, CSC_A->mData, CSC_A->mPtr, devTempIndex, devTempData, devTempPTR, (n2max + 1), CSC_A->n);
			}else if(n2max >= 3 && n2max < 6){
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/4) + 1;
				 ComputeTmpDataIn1<4><<<blocksPerGrid, threadsPerBlock>>>(CSC_A->mIndex, CSC_A->mData, CSC_A->mPtr, devTempIndex, devTempData, devTempPTR, (n2max + 1), CSC_A->n);
			}else if(n2max >= 6 && n2max < 12){
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/8) + 1;
				 ComputeTmpDataIn1<8><<<blocksPerGrid, threadsPerBlock>>>(CSC_A->mIndex, CSC_A->mData, CSC_A->mPtr, devTempIndex, devTempData, devTempPTR, (n2max + 1), CSC_A->n);
			}else if(n2max >= 12 && n2max < 24){
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/16) + 1;
				 ComputeTmpDataIn1<16><<<blocksPerGrid, threadsPerBlock>>>(CSC_A->mIndex, CSC_A->mData, CSC_A->mPtr, devTempIndex, devTempData, devTempPTR, (n2max + 1), CSC_A->n);
			}else {
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/32) + 1;
				 ComputeTmpDataIn1<2><<<blocksPerGrid, threadsPerBlock>>>(CSC_A->mIndex, CSC_A->mData, CSC_A->mPtr, devTempIndex, devTempData, devTempPTR, (n2max + 1), CSC_A->n);
			}
			
			threadsPerBlock = 256;
      blocksPerGrid = (CSC_A->n - 1)/(threadsPerBlock) + 1;
			JudgeE<<<blocksPerGrid, threadsPerBlock>>>(devTempIndex, devTempData, devTempPTR, (n2max + 1), CSC_A->n);
			
			int *dev_nnz;
			int *nnz2 = (int*)malloc( sizeof(int) );
			int *ptr2 = (int*)malloc( sizeof(int) * (CSC_A->n+1));
	  	hipMalloc((void**)&dev_nnz, sizeof(int)) ; 
			cuDcomputeNNZIn1<1024><<<1, 1024>>>(devTempPTR, CSC_A->n, dev_nnz);
	    hipMemcpyAsync( nnz2, dev_nnz, sizeof( int ), hipMemcpyDeviceToHost, 0 ) ;  
	    
	    
	    CSC_SparsityA->nRow = CSC_A->n ;
			CSC_SparsityA->nCol = CSC_A->n ;
			CSC_SparsityA->n = CSC_A->n ;
			CSC_SparsityA->nonzeroes = *nnz2 ;
			
			printf("nnz2 = %d\n", *nnz2);
			
			hipMalloc((void**)&CSC_SparsityA->mPtr, sizeof(int) * (CSC_SparsityA->n + 1)) ;
      hipMalloc((void**)&CSC_SparsityA->mIndex, sizeof(int) * CSC_SparsityA->nonzeroes ) ;
      hipMalloc((void**)&CSC_SparsityA->mData, sizeof(double) * CSC_SparsityA->nonzeroes) ;
			
			/*Put A^T into CSC_SparsityA */
			hipMemcpyAsync( ptr2 + 1, devTempPTR, CSC_A->n * sizeof( int ), hipMemcpyDeviceToHost, 0 ) ; 
			
      ptr2[0] = 0;
	  	for(int i1 = 0; i1 < CSC_A->n; i1++){
	  			ptr2[i1+1] += ptr2[i1];
	  			//printf("i1=%d\n", ptr2[i1+1]);
	  	}
	  	printf("CSC_SparsityA->nonzeroes = %d\n", ptr2[CSC_A->n]);
	  		
	  	hipMemcpyAsync(CSC_SparsityA->mPtr, ptr2, (CSC_A->n+1) * sizeof( int ), hipMemcpyHostToDevice, 0 ) ; 
      
      blocksPerGrid = 15 * 8 * 4;
      cuComputeN2MAX<256><<<blocksPerGrid, threadsPerBlock>>>(CSC_SparsityA->mPtr, CSC_SparsityA->n, dev_maxV);
  
  		hipMemcpyAsync( maxV, dev_maxV, blocksPerGrid * sizeof( int ), hipMemcpyDeviceToHost, 0 ) ; 
  		int n2maxE = maxV[0];
  		for(int i = 1; i < blocksPerGrid; i++){
     		 if(n2maxE < maxV[i]) n2maxE = maxV[i];
  		}
  
  		printf("n2maxE = %d\n", n2maxE);
  		
  		if(n2maxE < 3){
		   	 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/2) + 1;
			   WriteATIntoSparsityAIn1<2><<<blocksPerGrid, threadsPerBlock>>>(devTempIndex, devTempData, CSC_SparsityA->mPtr, CSC_SparsityA->mIndex, CSC_SparsityA->mData, (n2max + 1), CSC_SparsityA->n);
			}else if(n2maxE >= 3 && n2maxE < 6){
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/4) + 1;
				 WriteATIntoSparsityAIn1<4><<<blocksPerGrid, threadsPerBlock>>>(devTempIndex, devTempData, CSC_SparsityA->mPtr, CSC_SparsityA->mIndex, CSC_SparsityA->mData, (n2max + 1), CSC_SparsityA->n);
			}else if(n2maxE >= 6 && n2maxE < 12){
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/8) + 1;
				 WriteATIntoSparsityAIn1<8><<<blocksPerGrid, threadsPerBlock>>>(devTempIndex, devTempData, CSC_SparsityA->mPtr, CSC_SparsityA->mIndex, CSC_SparsityA->mData, (n2max + 1), CSC_SparsityA->n);
			}else if(n2maxE >= 12 && n2maxE < 24){
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/16) + 1;
				 WriteATIntoSparsityAIn1<16><<<blocksPerGrid, threadsPerBlock>>>(devTempIndex, devTempData, CSC_SparsityA->mPtr, CSC_SparsityA->mIndex, CSC_SparsityA->mData, (n2max + 1), CSC_SparsityA->n);
			}else {
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/32) + 1;
				 WriteATIntoSparsityAIn1<32><<<blocksPerGrid, threadsPerBlock>>>(devTempIndex, devTempData, CSC_SparsityA->mPtr, CSC_SparsityA->mIndex, CSC_SparsityA->mData, (n2max + 1), CSC_SparsityA->n);
			}
			
			if(n2maxE < 3){
		   	 threadsPerBlock = 256;
			   blocksPerGrid = (CSC_SparsityA->n - 1)/(threadsPerBlock/2) + 1;
			   SortInCSC2CSR<2><<<blocksPerGrid, threadsPerBlock>>>(CSC_SparsityA->mData, CSC_SparsityA->mIndex, CSC_SparsityA->mPtr, CSC_SparsityA->n);
			}else if(n2maxE >= 3 && n2maxE < 6){
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSC_SparsityA->n - 1)/(threadsPerBlock/4) + 1;
				 SortInCSC2CSR<4><<<blocksPerGrid, threadsPerBlock>>>(CSC_SparsityA->mData, CSC_SparsityA->mIndex, CSC_SparsityA->mPtr, CSC_SparsityA->n);
			}else if(n2maxE >= 6 && n2maxE < 12){
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSC_SparsityA->n - 1)/(threadsPerBlock/8) + 1;
				 SortInCSC2CSR<8><<<blocksPerGrid, threadsPerBlock>>>(CSC_SparsityA->mData, CSC_SparsityA->mIndex, CSC_SparsityA->mPtr, CSC_SparsityA->n);
			}else if(n2maxE >= 12 && n2maxE < 24){
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSC_SparsityA->n - 1)/(threadsPerBlock/16) + 1;
				 SortInCSC2CSR<16><<<blocksPerGrid, threadsPerBlock>>>(CSC_SparsityA->mData, CSC_SparsityA->mIndex, CSC_SparsityA->mPtr, CSC_SparsityA->n);
			}else {
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSC_SparsityA->n - 1)/(threadsPerBlock/32) + 1;
				 SortInCSC2CSR<32><<<blocksPerGrid, threadsPerBlock>>>(CSC_SparsityA->mData, CSC_SparsityA->mIndex, CSC_SparsityA->mPtr, CSC_SparsityA->n);
			}
  }else if( flag == 2 ){//E + A + A^T
  	  threadsPerBlock = 256;
			blocksPerGrid = 15 * 8 * 4;
  		int *dev_maxV;
  		int *maxV = (int*)malloc(sizeof(int) * blocksPerGrid);
  		hipMalloc((void**)&dev_maxV, sizeof(int) * blocksPerGrid) ;
  
  		cuComputeN2MAX<256><<<blocksPerGrid, threadsPerBlock>>>(CSC_A->mPtr, CSC_A->n, dev_maxV);
  
  		hipMemcpyAsync( maxV, dev_maxV, blocksPerGrid * sizeof( int ), hipMemcpyDeviceToHost, 0 ) ; 
  		int n2cmax = maxV[0];
  		for(int i = 1; i < blocksPerGrid; i++){
     		 if(n2cmax < maxV[i]) n2cmax = maxV[i];
  		}
  
  		printf("n2cmax = %d\n", n2cmax);
  		
  		cuComputeN2MAX<256><<<blocksPerGrid, threadsPerBlock>>>(CSR_A->mPtr, CSR_A->n, dev_maxV);
  
  		hipMemcpyAsync( maxV, dev_maxV, blocksPerGrid * sizeof( int ), hipMemcpyDeviceToHost, 0 ) ; 
  		int n2rmax = maxV[0];
  		for(int i = 1; i < blocksPerGrid; i++){
     		 if(n2rmax < maxV[i]) n2rmax = maxV[i];
  		}
  
  		printf("n2rmax = %d\n", n2rmax);
  		
  		int n2max = n2cmax + n2rmax;
  		
  		printf("n2max = %d\n", n2max);
  		
  		int *devTempIndex;
  		double *devTempData;
  		int *devTempPTR;
  		
  		hipMalloc((void**)&devTempPTR, sizeof(int) * CSC_A->n) ;
      hipMalloc((void**)&devTempIndex, sizeof(int) * CSC_A->n * (n2max + 1) ) ;
      hipMalloc((void**)&devTempData, sizeof(double) * CSC_A->n * (n2max + 1)) ;
      
  		
  		if(n2cmax < 3){
		   	 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/2) + 1;
			   ComputeTmpDataIn1<2><<<blocksPerGrid, threadsPerBlock>>>(CSC_A->mIndex, CSC_A->mData, CSC_A->mPtr, devTempIndex, devTempData, devTempPTR, (n2max + 1), CSC_A->n);
			}else if(n2cmax >= 3 && n2cmax < 6){
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/4) + 1;
				 ComputeTmpDataIn1<4><<<blocksPerGrid, threadsPerBlock>>>(CSC_A->mIndex, CSC_A->mData, CSC_A->mPtr, devTempIndex, devTempData, devTempPTR, (n2max + 1), CSC_A->n);
			}else if(n2cmax >= 6 && n2cmax < 12){
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/8) + 1;
				 ComputeTmpDataIn1<8><<<blocksPerGrid, threadsPerBlock>>>(CSC_A->mIndex, CSC_A->mData, CSC_A->mPtr, devTempIndex, devTempData, devTempPTR, (n2max + 1), CSC_A->n);
			}else if(n2cmax >= 12 && n2cmax < 24){
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/16) + 1;
				 ComputeTmpDataIn1<16><<<blocksPerGrid, threadsPerBlock>>>(CSC_A->mIndex, CSC_A->mData, CSC_A->mPtr, devTempIndex, devTempData, devTempPTR, (n2max + 1), CSC_A->n);
			}else {
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/32) + 1;
				 ComputeTmpDataIn1<32><<<blocksPerGrid, threadsPerBlock>>>(CSC_A->mIndex, CSC_A->mData, CSC_A->mPtr, devTempIndex, devTempData, devTempPTR, (n2max + 1), CSC_A->n);
			}
			
			threadsPerBlock = 256;
      blocksPerGrid = (CSC_SparsityA->n - 1)/(threadsPerBlock) + 1;
			JudgeE<<<blocksPerGrid, threadsPerBlock>>>(devTempIndex, devTempData, devTempPTR, (n2max + 1), CSC_A->n);
			
			
			int *atomic ;
	    hipMalloc( (void**)&atomic, sizeof( int ) * CSC_A->n ) ;
			if(n2rmax < 3){
		   	 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/2) + 1;
			   WriteATIntoSparsityAIn2<2><<<blocksPerGrid, threadsPerBlock>>>(CSR_A->mIndex, CSR_A->mData, CSR_A->mPtr, devTempIndex, devTempData, devTempPTR, atomic, (n2max+1), CSC_A->n);
			}else if(n2rmax >= 3 && n2rmax < 6){
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/4) + 1;
				 WriteATIntoSparsityAIn2<4><<<blocksPerGrid, threadsPerBlock>>>(CSR_A->mIndex, CSR_A->mData, CSR_A->mPtr, devTempIndex, devTempData, devTempPTR, atomic, (n2max+1), CSC_A->n);
			}else if(n2rmax >= 6 && n2rmax < 12){
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/8) + 1;
				 WriteATIntoSparsityAIn2<8><<<blocksPerGrid, threadsPerBlock>>>(CSR_A->mIndex, CSR_A->mData, CSR_A->mPtr, devTempIndex, devTempData, devTempPTR, atomic, (n2max+1), CSC_A->n);
			}else if(n2rmax >= 12 && n2rmax < 24){
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/16) + 1;
				 WriteATIntoSparsityAIn2<16><<<blocksPerGrid, threadsPerBlock>>>(CSR_A->mIndex, CSR_A->mData, CSR_A->mPtr, devTempIndex, devTempData, devTempPTR, atomic, (n2max+1), CSC_A->n);
			}else {
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/32) + 1;
				 WriteATIntoSparsityAIn2<32><<<blocksPerGrid, threadsPerBlock>>>(CSR_A->mIndex, CSR_A->mData, CSR_A->mPtr, devTempIndex, devTempData, devTempPTR, atomic, (n2max+1), CSC_A->n);
			}
			
      int *dev_nnz;
			int *nnz2 = (int*)malloc( sizeof(int) );
			int *ptr2 = (int*)malloc( sizeof(int) * (CSC_A->n+1));
	  	hipMalloc((void**)&dev_nnz, sizeof(int)) ; 
			cuDcomputeNNZIn1<1024><<<1, 1024>>>(devTempPTR, CSC_A->n, dev_nnz);
	    hipMemcpyAsync( nnz2, dev_nnz, sizeof( int ), hipMemcpyDeviceToHost, 0 ) ;  
	    
	    
	    CSC_SparsityA->nRow = CSC_A->n ;
			CSC_SparsityA->nCol = CSC_A->n ;
			CSC_SparsityA->n = CSC_A->n ;
			CSC_SparsityA->nonzeroes = *nnz2 ;
			
			printf("nnz2 = %d\n", *nnz2);
			
			hipMalloc((void**)&CSC_SparsityA->mPtr, sizeof(int) * (CSC_SparsityA->n + 1)) ;
      hipMalloc((void**)&CSC_SparsityA->mIndex, sizeof(int) * CSC_SparsityA->nonzeroes ) ;
      hipMalloc((void**)&CSC_SparsityA->mData, sizeof(double) * CSC_SparsityA->nonzeroes) ;
			
			/*Put A^T into CSC_SparsityA */
			hipMemcpyAsync( ptr2 + 1, devTempPTR, CSC_A->n * sizeof( int ), hipMemcpyDeviceToHost, 0 ) ; 
			
      ptr2[0] = 0;
	  	for(int i1 = 0; i1 < CSC_A->n; i1++){
	  			ptr2[i1+1] += ptr2[i1];
	  			//printf("i1=%d\n", ptr2[i1+1]);
	  	}
	  		
	  	hipMemcpyAsync(CSC_SparsityA->mPtr, ptr2, (CSC_A->n+1) * sizeof( int ), hipMemcpyHostToDevice, 0 ) ; 
      
      blocksPerGrid = 15 * 8 * 4;
      cuComputeN2MAX<256><<<blocksPerGrid, threadsPerBlock>>>(CSC_SparsityA->mPtr, CSC_SparsityA->n, dev_maxV);
  
  		hipMemcpyAsync( maxV, dev_maxV, blocksPerGrid * sizeof( int ), hipMemcpyDeviceToHost, 0 ) ; 
  		int n2maxE = maxV[0];
  		for(int i = 1; i < blocksPerGrid; i++){
     		 if(n2maxE < maxV[i]) n2maxE = maxV[i];
  		}
  
  		printf("n2maxE = %d\n", n2maxE);
  		
  		if(n2maxE < 3){
		   	 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/2) + 1;
			   WriteATIntoSparsityAIn1<2><<<blocksPerGrid, threadsPerBlock>>>(devTempIndex, devTempData, CSC_SparsityA->mPtr, CSC_SparsityA->mIndex, CSC_SparsityA->mData, (n2max + 1), CSC_SparsityA->n);
			}else if(n2maxE >= 3 && n2maxE < 6){
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/4) + 1;
				 WriteATIntoSparsityAIn1<4><<<blocksPerGrid, threadsPerBlock>>>(devTempIndex, devTempData, CSC_SparsityA->mPtr, CSC_SparsityA->mIndex, CSC_SparsityA->mData, (n2max + 1), CSC_SparsityA->n);
			}else if(n2maxE >= 6 && n2maxE < 12){
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/8) + 1;
				 WriteATIntoSparsityAIn1<8><<<blocksPerGrid, threadsPerBlock>>>(devTempIndex, devTempData, CSC_SparsityA->mPtr, CSC_SparsityA->mIndex, CSC_SparsityA->mData, (n2max + 1), CSC_SparsityA->n);
			}else if(n2maxE >= 12 && n2maxE < 24){
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/16) + 1;
				 WriteATIntoSparsityAIn1<16><<<blocksPerGrid, threadsPerBlock>>>(devTempIndex, devTempData, CSC_SparsityA->mPtr, CSC_SparsityA->mIndex, CSC_SparsityA->mData, (n2max + 1), CSC_SparsityA->n);
			}else {
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSR_A->n - 1)/(threadsPerBlock/32) + 1;
				 WriteATIntoSparsityAIn1<32><<<blocksPerGrid, threadsPerBlock>>>(devTempIndex, devTempData, CSC_SparsityA->mPtr, CSC_SparsityA->mIndex, CSC_SparsityA->mData, (n2max + 1), CSC_SparsityA->n);
			}
			
			if(n2maxE < 3){
		   	 threadsPerBlock = 256;
			   blocksPerGrid = (CSC_SparsityA->n - 1)/(threadsPerBlock/2) + 1;
			   SortInCSC2CSR<2><<<blocksPerGrid, threadsPerBlock>>>(CSC_SparsityA->mData, CSC_SparsityA->mIndex, CSC_SparsityA->mPtr, CSC_SparsityA->n);
			}else if(n2maxE >= 3 && n2maxE < 6){
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSC_SparsityA->n - 1)/(threadsPerBlock/4) + 1;
				 SortInCSC2CSR<4><<<blocksPerGrid, threadsPerBlock>>>(CSC_SparsityA->mData, CSC_SparsityA->mIndex, CSC_SparsityA->mPtr, CSC_SparsityA->n);
			}else if(n2maxE >= 6 && n2maxE < 12){
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSC_SparsityA->n - 1)/(threadsPerBlock/8) + 1;
				 SortInCSC2CSR<8><<<blocksPerGrid, threadsPerBlock>>>(CSC_SparsityA->mData, CSC_SparsityA->mIndex, CSC_SparsityA->mPtr, CSC_SparsityA->n);
			}else if(n2maxE >= 12 && n2maxE < 24){
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSC_SparsityA->n - 1)/(threadsPerBlock/16) + 1;
				 SortInCSC2CSR<16><<<blocksPerGrid, threadsPerBlock>>>(CSC_SparsityA->mData, CSC_SparsityA->mIndex, CSC_SparsityA->mPtr, CSC_SparsityA->n);
			}else {
				 threadsPerBlock = 256;
			   blocksPerGrid = (CSC_SparsityA->n - 1)/(threadsPerBlock/32) + 1;
				 SortInCSC2CSR<32><<<blocksPerGrid, threadsPerBlock>>>(CSC_SparsityA->mData, CSC_SparsityA->mIndex, CSC_SparsityA->mPtr, CSC_SparsityA->n);
			}
  }
}
#endif