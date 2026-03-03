#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <memory.h>
#include <string> 
#include <vector>
#include "dataType.h"

using namespace std;


template<unsigned int N2SIZE>
__global__ void cuComputeN2MAXwithSparityofA_SpMMv1(int *aPtr, int nCol, int *n2max){
	
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


template<unsigned int WarpSize, unsigned int CounterSize, unsigned int ISIZE>
__global__ void cuComputeN1withSparityofA_SpMMv1(int *aPtr, int *aIndex, int *mPtr, int *mIndex, int *I, int n1max, int nCol, int *n1)
{
    __shared__ int I_s[ISIZE];
    __shared__ int counter_s[CounterSize];
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize-1);// index of threads in warp
    int tid = threadIdx.x / WarpSize; 
    
    int col, col_s, col_e;
    int fcol, fcol_s, fcol_e;
    int othercol, ocol, ocol_s, ocol_e;
    int bV, bV1, bV2;
    int j, j1;
    
    int seg = ISIZE/CounterSize;
    
    for(col = warp_id; col < nCol; col += offset)
    {
        //read the columns
        col_s = mPtr[col];
        col_e = mPtr[col+1];
        bV = col_e - col_s;

        //for the first column, all data are written into I
        fcol = mIndex[col_s];
        fcol_s = aPtr[fcol];
        fcol_e = aPtr[fcol+1];
        bV1 = fcol_e - fcol_s ;

        counter_s[tid] = bV1;
        __syncthreads();
        
        for(j = lane ; j < bV1 ; j += WarpSize)
        {
            I[col*n1max+j] = aIndex[j+fcol_s];
        }//for j

        __syncthreads();


        //for other columns
        for(othercol = 1; othercol < bV ; othercol++){
            ocol = mIndex[col_s + othercol];
            ocol_s = aPtr[ocol];
            ocol_e = aPtr[ocol+1];
            
            bV2 = ocol_e - ocol_s ;
            //if the element in other column does not exist in I, append it to I
            for(j = lane ; j < bV2 ; j += WarpSize)
            {
                I_s[tid*seg+j] = aIndex[j+ ocol_s] ;
                for(j1 = 0; j1 < counter_s[tid]; j1++){
                    if(I_s[tid*seg+j] == I[col*n1max+j1] ){
                    I_s[tid*seg+j] = -1;
                    break;
                    }
                }
            }//for j
            
            __syncthreads();
            
            if(lane == 0){
                for(j = 0 ; j < bV2 ; j++){
                if(I_s[tid*seg+j] != -1){
                    I[col*n1max+counter_s[tid]] = I_s[tid*seg+j];
                    //counter++;  
                    counter_s[tid] += 1;
                }
                }
            }
            __syncthreads();

        
        }//for othercol
        
        n1[col] = counter_s[tid];

    }//end for col
}

template<unsigned int N1SIZE>
__global__ void cuComputeN1MAX_SpMMv1(int *n1, int nCol, int *n1max){
		
    __shared__ int n1_s[N1SIZE];
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x * gridDim.x;
    int tid = threadIdx.x;
    
    int col, value; 
    int i;
	  
    value = 0;
    
    for(col = gid; col < nCol; col += offset)
    {
        if(value < n1[col]) value = n1[col];
    
    }//for col
    
    n1_s[tid] = value;
    
    __syncthreads();
	  
    i =  N1SIZE/2;
    while( tid < i )
    {
        if(n1_s[tid] < n1_s[tid + i])  n1_s[tid] =  n1_s[tid + i];
        i = i/2;
        __syncthreads();
    }
	     
	if(tid == 0) n1max[blockIdx.x] = n1_s[0];
	
}


template<unsigned int WarpSize, unsigned int CounterSize>
__global__ void cuComputeN1withSparityofA2_SpMMv1(int *aPtr, int *aIndex, int *mPtr, int *mIndex, int *I, int n1max, int nCol, int *n1){
	  
    __shared__ int counter_s[CounterSize];
	  int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize-1);// index of threads in warp
    
    int tid = threadIdx.x / WarpSize; 

    int col, col_s, col_e;
    int fcol, fcol_s, fcol_e;
    int othercol, ocol, ocol_s, ocol_e;
    int bV, bV1, bV2, tN, flag;
    int j, j1, idx;

    for(col = warp_id; col < nCol; col += offset)
    {
        //read the columns
        col_s = mPtr[col];
        col_e = mPtr[col+1];
        bV = col_e - col_s;
     
        //for the first column, all data are written into I
        fcol = mIndex[col_s];
        fcol_s = aPtr[fcol];
        fcol_e = aPtr[fcol+1];
        bV1 = fcol_e - fcol_s ;

        counter_s[tid] = bV1;
        __syncthreads();
        
        
        for(j = lane ; j < bV1 ; j += WarpSize)
        {
        I[col*n1max + j] = aIndex[j+fcol_s];
        }//for j
        
        __syncthreads();
        
        //for other columns
        for(othercol = 1; othercol < bV ; othercol++){
            ocol = mIndex[col_s + othercol];
            ocol_s = aPtr[ocol];
            ocol_e = aPtr[ocol+1];
            
            bV2 = ocol_e - ocol_s ;
            //if the element in other column does not exist in I, append it to I
            tN = counter_s[tid] ;

            for(j = 0 ; j < bV2 ; j ++ )
            {
                idx = aIndex[j+ocol_s] ; 
                flag = -1;
                for(j1 = 0; j1 < tN; j1++){
                    if( idx == I[col*n1max + j1] ){
                        flag = 1;
                        break;
                    }
                }
                if(lane == 0)
                {
                    if(flag == -1)
                    {
                        I[col*n1max+counter_s[tid]] = idx;
                        //counter++;  
                        counter_s[tid] += 1;
                    }
                }
                __syncthreads();
            }//for j 0:bv2

        }//for othercol
     
        n1[col] = counter_s[tid] ;
    
	}//end for col
}

template<unsigned int WarpSize>
__global__ void computeJ_iter_SpMMv1(int *MPtr, int *MIndex, double *MData,
                         int nCol, int *J, int *jPTR, double *JV, int N2MAX, int sK)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize-1);//index of threads in warp
    
    int col, col_s, col_e, j, bV ;
    int colK;
    
    for(col = warp_id; col < nCol; col += offset)
    {
        colK = col + sK;
        col_s = MPtr[colK];
        col_e = MPtr[colK+1];
        bV = col_e - col_s ;
        jPTR[col] = bV ;
        for(j = lane; j < bV ; j += WarpSize)
        {
          J[col*N2MAX+j] = MIndex[j+col_s]; 
          JV[col*N2MAX+j] = MData[j+col_s]; 
        }
    }//for col
  
} 


template<unsigned int WarpSize, unsigned int CounterSize>
__global__ void computeI_iter_SpMMv1(int *APtr, int *AIndex, int nCol, 
                int *I, int *iPTR, int N1MAX, int *J, int *jPTR, int N2MAX)
{
  
    __shared__ int counter_s[CounterSize];
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize-1);// index of threads in warp
    int tid = threadIdx.x / WarpSize; 

    int col, fcol, fcol_s, fcol_e, bV, bV1;
    int ocol, othercol, othercol_s, othercol_e, j, j1 ;

    //int seg = SIZE_I_SHARED/CounterSize;

    for(col = warp_id; col < nCol; col += offset)
    {
        //read the columns
        fcol = J[col*N2MAX];
        fcol_s = APtr[fcol];
        fcol_e = APtr[fcol+1];
        bV = fcol_e - fcol_s ;
        
        counter_s[tid] = bV;

        __syncthreads();
        
        //for the first column, all data are written into I
        for(j = lane ; j < bV ; j += WarpSize)
        {
        I[col*N1MAX+j] = AIndex[j+fcol_s];
        }//for j
        
        __syncthreads();
        
        //for other columns
        for(othercol = 1; othercol < jPTR[col] ; othercol++){
            ocol = J[col*N2MAX+othercol];
            othercol_s = APtr[ocol];
            othercol_e = APtr[ocol+1];
            
            bV1 = othercol_e - othercol_s ;
            //if the element in other column does not exist in I, append it to I
            int counter = 0;
            for(j = 0 ; j < bV1 ; j += 1)
            {
                int idx = AIndex[j+ othercol_s] ;
                int flag = 1;
                for(j1 = 0 ; j1 < counter_s[tid] ; j1++){
                    if(idx ==  I[col*N1MAX+j1]){
                        flag = -1;
                        break;
                    }
                }
                
                if(flag == 1){
                    I[col*N1MAX+counter_s[tid] + counter] = idx;
                    counter++;
                }
            }//for j
       
            counter_s[tid] += counter;
            __syncthreads();
       
        }//for othercol
     
        iPTR[col] = counter_s[tid] ;
     
     
        //ODD_EVEN
        //if(lane==0){   
        for( int i = 1; i <=counter_s[tid]; i++)
		    {
					if( i%2 == 1){
						for(int j1 = lane; j1 < counter_s[tid]; j1+=WarpSize)
						{
							if(((j1+1)%2==1)&&(I[col*N1MAX+j1]> I[col*N1MAX+j1+1])&&(j1+1)<counter_s[tid])
							{
								bV1 = I[col*N1MAX+j1] ;
								I[col*N1MAX+j1]=I[col*N1MAX+j1+1] ;
								I[col*N1MAX+j1+1] = bV1 ; 
							}
						}
						__syncthreads() ;
					}
					else{
						for( int j2 = lane; j2 < counter_s[tid]; j2+=WarpSize)
						{
							if(((j2+1)%2==0)&&(I[col*N1MAX+j2]>I[col*N1MAX+j2+1])&&(j2+1)<counter_s[tid])
							{
								bV1 = I[col*N1MAX+j2] ;
								I[col*N1MAX+j2] = I[col*N1MAX+j2+1] ;
								I[col*N1MAX+j2+1] = bV1 ;
							}
						}
						__syncthreads() ;
					}
		    } 

    }//for col
}


template<unsigned int WarpSize>
__global__ void ComputeTildeACSR_iter_SpMMv1(double *A1, double *AData, int *APtr, int *AIndex, int nCol, 
                        int *I, int *iPTR, int *J, int *jPTR, int N1MAX, int N2MAX)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index 
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize-1);//index of threads in warp
    
    int col,i,irow,j,jcol,jcol_b,jcol_e,jcol1;
    int AM, AN;
    double idata;
    
    for(col = warp_id; col < nCol; col += offset)
    {
        AM = iPTR[col];
        AN = jPTR[col];
        for(i = 0 ; i < AM ; i++)
        {
	        irow = I[col*N1MAX+i];
	        for(j = lane ; j < AN; j += WarpSize)
	        {
	            jcol = J[col*N2MAX+j];
	            jcol_b = APtr[jcol];
	            jcol_e = APtr[jcol+1];
	            
	            idata = 0.0;
	            for(jcol1 = jcol_b ; jcol1 < jcol_e ; jcol1++)
	            {
	                if(AIndex[jcol1] == irow){
	                    idata = AData[jcol1];
	                    break;
	                }
	            }
	            
	            A1[col*N1MAX*N2MAX + i*N2MAX + j] = idata;
	            
	        }//for j
        
        __syncthreads() ;
        }//for i
        
    }//for col
}	


template<unsigned int WarpSize, unsigned int JSIZE>
__global__ void ComputeAB_iter_SpMMv1(double *A, int *IPTR, int *JPTR, double *B, double *C, int N1MAX, int N2MAX, int nCol)
{
	__shared__ double BV_s[JSIZE];
	
	int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
  int offset = blockDim.x / WarpSize * gridDim.x;
  int warp_id = gid / WarpSize; //global warp index
  int lane = gid & (WarpSize-1);// index of threads in warp
  int tid = threadIdx.x / WarpSize; 
  
  int SEP, tj, ti, col;
  int i, j;
  double sum;
  
  int IBnd, JBnd, ABnd;
  
  for(col = warp_id; col < nCol; col += offset)
  {  
  	tj = JPTR[col];
    SEP = tid * WarpSize ;
    IBnd = col * N1MAX;
    JBnd = col * N2MAX;
    ABnd = col * N1MAX * N2MAX;
    
    //read dev_Jv into shared memory
  	for(j = lane; j < tj; j += WarpSize){
  		BV_s[SEP + j] = B[JBnd + j];
  	}

  	__syncthreads();
    
    ti = IPTR[col];
  	for(i = lane; i < ti; i += WarpSize){
  		sum = 0.0;
  		for(j = 0; j < tj; j++){
  			sum += A[ABnd + i * N2MAX + j] * BV_s[SEP + j];
  	  }
  	  
  	  C[IBnd + i] = sum ;
  	}
  	
  }	
}


template<unsigned int WarpSize>
__global__ void ComputeAB_iter_SpMMv1NoSharedM(double *A, int *IPTR, int *JPTR, double *B, double *C, int N1MAX, int N2MAX, int nCol)
{
	
	int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
  int offset = blockDim.x / WarpSize * gridDim.x;
  int warp_id = gid / WarpSize; //global warp index
  int lane = gid & (WarpSize-1);// index of threads in warp
  
  int tj, ti, col;
  int i, j;
  double sum;
  
  int IBnd, JBnd, ABnd;
  
  for(col = warp_id; col < nCol; col += offset)
  {  
  	tj = JPTR[col];
    IBnd = col * N1MAX;
    JBnd = col * N2MAX;
    ABnd = col * N1MAX * N2MAX;
    
    ti = IPTR[col];
  	for(i = lane; i < ti; i += WarpSize){
  		sum = 0.0;
  		for(j = 0; j < tj; j++){
  			sum += A[ABnd + i * N2MAX + j] * B[JBnd + j];
  	  }
  	  
  	  C[IBnd + i] = sum ;
  	}
  	
  }	
}

//---once compute
template<unsigned int WarpSize>
__global__ void computeJ_SpMMv1(int *MPtr, int *MIndex, double *MData,
                         int nCol, int *J, int *jPTR, double *JV, int N2MAX)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize-1);//index of threads in warp
    
    int col, col_s, col_e, j, bV ;
    
    for(col = warp_id; col < nCol; col += offset)
    {
        col_s = MPtr[col];
        col_e = MPtr[col+1];
        bV = col_e - col_s ;
        jPTR[col] = bV ;
        for(j = lane; j < bV ; j += WarpSize)
        {
            J[col*N2MAX+j] = MIndex[j+col_s]; 
            JV[col*N2MAX+j] = MData[j+col_s];
        }
    }//for col
  
} 

template<unsigned int WarpSize, unsigned int CounterSize>
__global__ void computeI_SpMMv1(int *APtr, int *AIndex, int nCol, 
                int *I, int *iPTR, int N1MAX, int *J, int *jPTR, int N2MAX)
{
  
    __shared__ int counter_s[CounterSize];
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize-1);// index of threads in warp
    int tid = threadIdx.x / WarpSize; 
    
    int col, fcol, fcol_s, fcol_e, bV, bV1;
    int ocol, othercol, othercol_s, othercol_e, j, j1 ;
    
    //int seg = SIZE_I_SHARED/CounterSize;
    
    for(col = warp_id; col < nCol; col += offset)
    {
        //read the columns
        fcol = J[col*N2MAX];
        fcol_s = APtr[fcol];
        fcol_e = APtr[fcol+1];
        bV = fcol_e - fcol_s ;
        
        counter_s[tid] = bV;

        __syncthreads();
        
        //for the first column, all data are written into I
        for(j = lane ; j < bV ; j += WarpSize)
        {
            I[col*N1MAX+j] = AIndex[j+fcol_s];
        }//for j
        
        __syncthreads();
        
        //for other columns
        for(othercol = 1; othercol < jPTR[col] ; othercol++){
            ocol = J[col*N2MAX+othercol];
            othercol_s = APtr[ocol];
            othercol_e = APtr[ocol+1];
            
            bV1 = othercol_e - othercol_s ;
            //if the element in other column does not exist in I, append it to I
            int counter = 0;
            for(j = 0 ; j < bV1 ; j += 1)
            {
                int idx = AIndex[j+ othercol_s] ;
                int flag = 1;
                for(j1 = 0 ; j1 < counter_s[tid] ; j1++){
                    if(idx ==  I[col*N1MAX+j1]){
                        flag = -1;
                        break;
                    }
                }
                
                if(flag == 1){
                    I[col*N1MAX+counter_s[tid] + counter] = idx;
                    counter++;
                }
            }//for j
            
            counter_s[tid] += counter;
            __syncthreads();
        
        }//for othercol
        
        iPTR[col] = counter_s[tid] ;
        
        
        //ODD_EVEN
        //if(lane==0){   
        for( int i = 1; i <=counter_s[tid]; i++)
            {
                if( i%2 == 1){
                    for(int j1 = lane; j1 < counter_s[tid]; j1+=WarpSize)
                    {
                        if(((j1+1)%2==1)&&(I[col*N1MAX+j1]> I[col*N1MAX+j1+1])&&(j1+1)<counter_s[tid])
                        {
                            bV1 = I[col*N1MAX+j1] ;
                            I[col*N1MAX+j1]=I[col*N1MAX+j1+1] ;
                            I[col*N1MAX+j1+1] = bV1 ; 
                        }
                    }
                    __syncthreads() ;
                }
                else{
                    for( int j2 = lane; j2 < counter_s[tid]; j2+=WarpSize)
                    {
                        if(((j2+1)%2==0)&&(I[col*N1MAX+j2]>I[col*N1MAX+j2+1])&&(j2+1)<counter_s[tid])
                        {
                            bV1 = I[col*N1MAX+j2] ;
                            I[col*N1MAX+j2] = I[col*N1MAX+j2+1] ;
                            I[col*N1MAX+j2+1] = bV1 ;
                        }
                    }
                    __syncthreads() ;
                }
            } 

                // }
    }//for col
}


template<unsigned int WarpSize>
__global__ void ComputeTildeACSR_SpMMv1(double *A1, double *AData, int *APtr, int *AIndex, int nCol, 
                        int *I, int *iPTR, int *J, int *jPTR, int N1MAX, int N2MAX)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index 
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize-1);//index of threads in warp
    
    int col,i,irow,j,jcol,jcol_b,jcol_e,jcol1;
    int AM, AN;
    double idata;
    
    for(col = warp_id; col < nCol; col += offset)
    {
        AM = iPTR[col];
        AN = jPTR[col];
        for(i = 0 ; i < AM ; i++)
        {
        irow = I[col*N1MAX+i];
        for(j = lane ; j < AN; j += WarpSize)
        {
            jcol = J[col*N2MAX+j];
            jcol_b = APtr[jcol];
            jcol_e = APtr[jcol+1];
            
            idata = 0.0;
            for(jcol1 = jcol_b ; jcol1 < jcol_e ; jcol1++)
            {
                if(AIndex[jcol1] == irow){
                    idata = AData[jcol1];
                    break;
                }
            }
            
            A1[col*N1MAX*N2MAX + i*N2MAX + j] = idata;
            
        }//for j
        
        __syncthreads() ;
        }//for i
        
    }//for col
}	

template<unsigned int WarpSize, unsigned int JSIZE>
__global__ void ComputeAB_SpMMv1(double *A, int *IPTR, int *JPTR, double *B, double *C, int N1MAX, int N2MAX, int nCol)
{
	__shared__ double BV_s[JSIZE];
	
	int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
  int offset = blockDim.x / WarpSize * gridDim.x;
  int warp_id = gid / WarpSize; //global warp index
  int lane = gid & (WarpSize-1);// index of threads in warp
  int tid = threadIdx.x / WarpSize; 
  
  int SEP, tj, ti, col;
  int i, j;
  double sum;
  
  int IBnd, JBnd, ABnd;
  
  for(col = warp_id; col < nCol; col += offset)
  {  
  	tj = JPTR[col];
    SEP = tid * WarpSize ;
    IBnd = col * N1MAX;
    JBnd = col * N2MAX;
    ABnd = col * N1MAX * N2MAX;
    
    //read dev_Jv into shared memory
  	for(j = lane; j < tj; j += WarpSize){
  		BV_s[SEP + j] = B[JBnd + j];
  	}

  	__syncthreads();
    
    ti = IPTR[col];
  	for(i = lane; i < ti; i += WarpSize){
  		sum = 0.0;
  		for(j = 0; j < tj; j++){
  			sum += A[ABnd + i * N2MAX + j] * BV_s[SEP + j];
  	  }
  	  
  	  C[IBnd + i] = sum ;
  	}
  	
  }	
}


template<unsigned int WarpSize>
__global__ void ComputeAB_SpMMv1NoSharedM(double *A, int *IPTR, int *JPTR, double *B, double *C, int N1MAX, int N2MAX, int nCol)
{
	
	int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
  int offset = blockDim.x / WarpSize * gridDim.x;
  int warp_id = gid / WarpSize; //global warp index
  int lane = gid & (WarpSize-1);// index of threads in warp
  int tid = threadIdx.x / WarpSize; 
  
  int tj, ti, col;
  int i, j;
  double sum;
  
  int IBnd, JBnd, ABnd;
  
  for(col = warp_id; col < nCol; col += offset)
  {  
  	tj = JPTR[col];
    IBnd = col * N1MAX;
    JBnd = col * N2MAX;
    ABnd = col * N1MAX * N2MAX;
    
    ti = IPTR[col];
  	for(i = lane; i < ti; i += WarpSize){
  		sum = 0.0;
  		for(j = 0; j < tj; j++){
  			sum += A[ABnd + i * N2MAX + j] * B[JBnd + j];
  	  }
  	  
  	  C[IBnd + i] = sum ;
  	}
  	
  }	
}

__global__ void summation_SpMMv1(int *csrPtr, int m){
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
__global__ void WriteABIntoC_SpMMv1(int *devTempIndex, double *devTempData, int *csrSparsityAPTR, int *csrSparsityAIndex, double *csrSparsityAData, int N2MAX, int n)
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
  		csrSparsityAData[col_s + i] = devTempData[COLJ + i];
  	}
  }
}

template<unsigned int N2SIZE>
__global__ void cuComputeN2MINwithSparityofA_SpMMv2(int *aPtr, int nCol, int *n2min){
	
	__shared__ int n2_s[N2SIZE];
	int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
  int offset = blockDim.x * gridDim.x;
  int tid = threadIdx.x;
 
	int col, col_s, col_e, tV; 
	int i;
  
  int value = 1e7; 
  for(col = gid; col < nCol; col += offset)
  {
     col_s = aPtr[col];
     col_e = aPtr[col+1];
     tV = col_e - col_s ;
     if( value > tV) value = tV;
    
  }//for col
  
  n2_s[tid] = value;
  
  __syncthreads();
      
  i =  N2SIZE/2;
  while( tid < i )
  {
    if(n2_s[tid] > n2_s[tid + i])  n2_s[tid] =  n2_s[tid + i];
    i = i/2;
	  __syncthreads();
  }
     
  if(tid == 0) n2min[blockIdx.x] = n2_s[0];
	
}

__global__ void computeNZUPPWithGuass_SpMMv2(int *cPtr, int nCol, int nonzeros, int *nzUPP, double alpha, int n2max, int n2min)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
  int offset = blockDim.x * gridDim.x;
  
  int col, col_s, col_e, bV;
  double c, x;
  double eqV = (double)nonzeros/nCol;
  
  for(col = gid; col < nCol; col += offset)
  {
     col_s = cPtr[col];
     col_e = cPtr[col+1];
     bV = col_e - col_s ;
     
     if(n2max == n2min){
     	 nzUPP[col] = bV;
     }else{
     	 c = n2max - eqV;
     	 if( c < (eqV - n2min) ) c = eqV - n2min;
     	 	
     	 x = (bV - eqV)/c;	
     	 nzUPP[col] = min((int)(alpha*bV*exp(-x*x)), nCol);
     	 //printf("ltes=%d\n", nzUPP[col]);
     }
     
  }//for col
  
} 

template<unsigned int WarpSize>
__global__ void SparistyGuass_SpMMv2(int *tempPTR, double *V, int *I, int *UPP, int N1MAX, int nCol)
{
	
	int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
  int offset = blockDim.x / WarpSize * gridDim.x;
  int warp_id = gid / WarpSize; //global warp index
  int lane = gid & (WarpSize-1);// index of threads in warp
  
  int tN, IBnd, col, bV1;
  double bV2;
  int i, j1, j2, idx;
  
  for(col = warp_id; col < nCol; col += offset)
  {
	  
    tN = tempPTR[col];
	  IBnd = col * N1MAX;
	  
	  if(tN <= UPP[col]) continue; 
	  	
	  for(i = lane ; i < tN-1; i += WarpSize){
	  	idx = I[IBnd + i];
	  	if( idx == col ){
	  		bV1 = I[IBnd + i];
	  		I[IBnd + i] = I[IBnd + tN -1];
	  		I[IBnd + tN -1] = bV1;
	  
	  		bV2 = V[IBnd + i];
	  		V[IBnd + i] = V[IBnd + tN -1];
	  		V[IBnd + tN -1] = bV2;
	  		break;
	  	}
	  }
	  
	  __syncthreads() ;
	  
		//ODD_EVEN
	  for( i = 1; i <= tN-1; i++)
	  {
			if( i%2 == 1){
				for(j1 = lane; j1 < tN-1; j1 += WarpSize)
				{
					if((j1*2+1)<tN-1&&(V[IBnd+j1*2] < V[IBnd+j1*2+1]))
					{
							bV1 = I[IBnd+j1*2] ;
							I[IBnd+j1*2]=I[IBnd+j1*2+1] ;
							I[IBnd+j1*2+1] = bV1 ; 
							
							bV2 = V[IBnd+j1*2] ;
							V[IBnd+j1*2]=V[IBnd+j1*2+1] ;
							V[IBnd+j1*2+1] = bV2 ; 
					}
				}
				__syncthreads() ;
			}
			else{
				for( j2 = lane; j2 < tN-1; j2 += WarpSize)
				{
					if((j2*2+2)<tN-1&&(V[IBnd+j2*2+1] < V[IBnd+j2*2+2]))
					{
							bV1= I[IBnd+j2*2+2] ;
							I[IBnd+j2*2+2] = I[IBnd+j2*2+1] ;
							I[IBnd+j2*2+1] = bV1 ;
							
							bV2= V[IBnd+j2*2+2] ;
							V[IBnd+j2*2+2] = V[IBnd+j2*2+1] ;
							V[IBnd+j2*2+1] = bV2 ;
					}
				}
				__syncthreads() ;
			}
		}
	  
	  I[IBnd + UPP[col] -1] = I[IBnd + tN -1];
	  V[IBnd + UPP[col] -1] = V[IBnd + tN -1];
		
		tempPTR[col] = UPP[col];
	  
	  tN = tempPTR[col];
	  IBnd = col * N1MAX;
		//ODD_EVEN
	  for( i = 1; i <= tN; i++)
	  {
			if( i%2 == 1){
				for(j1 = lane; j1 < tN; j1 += WarpSize)
				{
					if((j1*2+1)<tN&&(I[IBnd+j1*2] > I[IBnd+j1*2+1]))
					{
							bV1 = I[IBnd+j1*2] ;
							I[IBnd+j1*2]=I[IBnd+j1*2+1] ;
							I[IBnd+j1*2+1] = bV1 ; 
							
							bV2 = V[IBnd+j1*2] ;
							V[IBnd+j1*2]=V[IBnd+j1*2+1] ;
							V[IBnd+j1*2+1] = bV2 ; 
					}
				}
				__syncthreads() ;
			}
			else{
				for( j2 = lane; j2 < tN; j2 += WarpSize)
				{
					if((j2*2+2)<tN&&(I[IBnd+j2*2+1] > I[IBnd+j2*2+2]))
					{
							bV1= I[IBnd+j2*2+2] ;
							I[IBnd+j2*2+2] = I[IBnd+j2*2+1] ;
							I[IBnd+j2*2+1] = bV1 ;
							
							bV2= V[IBnd+j2*2+2] ;
							V[IBnd+j2*2+2] = V[IBnd+j2*2+1] ;
							V[IBnd+j2*2+1] = bV2 ;
					}
				}
				__syncthreads() ;
			}
		} 
		
	}//end col 
}