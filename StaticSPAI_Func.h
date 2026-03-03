#ifndef STATICSPAI_FUNC_H
#define STATICSPAI_FUNC_H

template<unsigned int N2SIZE>
__global__ void cuComputeN2MAXwithSparityofA_SSPAIv1(int *aPtr, int nCol, int *n2max){
	
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
__global__ void cuComputeN1withSparityofA_SSPAIv1(int *aPtr, int *aIndex, int *mPtr, int *mIndex, int *I, int n1max, int nCol, int *n1)
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
__global__ void cuComputeN1MAX_SSPAIv1(int *n1, int nCol, int *n1max){
		
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
__global__ void cuComputeN1withSparityofA2_SSPAIv1(int *aPtr, int *aIndex, int *mPtr, int *mIndex, int *I, int n1max, int nCol, int *n1){
	  
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
__global__ void computeJ_iter_SSPAIv1(int *MPtr, int *MIndex, 
                         int nCol, int *J, int *jPTR, int N2MAX, int sK)
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
        }
    }//for col
  
} 


template<unsigned int WarpSize, unsigned int CounterSize>
__global__ void computeI_iter_SSPAIv1(int *APtr, int *AIndex, int nCol, 
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
__global__ void ComputeTildeACSR_iter_SSPAIv1(double *A1, double *AData, int *APtr, int *AIndex, int nCol, 
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


template<unsigned int WarpSize, unsigned int SIZE_R_SHARED>
__global__ void QR_RShared_iter_SSPAIv1(double *Q, double *R, int *iPTR, int *jPTR, int N1MAX, int N2MAX, int nCol)
{
    __shared__ double R_s[SIZE_R_SHARED];
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize-1);//index of threads in warp
    
    int tid = threadIdx.x / WarpSize;
    
    int col, AM, AN;
    int i, j, k;
    double rii, tR;
    
    int segR = SIZE_R_SHARED * WarpSize / blockDim.x ;
    
    for(col = warp_id; col < nCol; col += offset)
    {
        //for each col corresponding to a tilde of A
        AM = iPTR[col];
        AN = jPTR[col];
        
        for(i = 0 ; i < AN; i++)
        {
        
            /*Compute R in parallel an put them into shared memory*/
            for(j = lane + i; j < AN; j += WarpSize)
            {
                tR = 0.0;
                for(k = 0 ; k < AM ; k++)
                {
                tR += Q[col*N1MAX*N2MAX + i + k*N2MAX] * Q[col*N1MAX*N2MAX + j + k*N2MAX];
                }//for k
                
                R_s[tid*segR+j-i] = tR ;
            }//for j
            
            __syncthreads();
            
            rii = sqrt(R_s[tid*segR]);
            //normalize column i of Q
            for(j = lane ; j < AM; j += WarpSize)
            {
                Q[col*N1MAX*N2MAX+j*N2MAX+i] /= rii ;
            }//for j
            
            __syncthreads();
            
            //compute projection factors
            for(j = lane + i; j < AN; j += WarpSize)
            {
                R_s[tid*segR+j-i] /= rii;
                R[col*N2MAX*N2MAX+i*N2MAX+j] = R_s[tid*segR+j-i] ;
            }//for j
            
            __syncthreads();
            
            for(j = lane + i + 1; j < AN; j += WarpSize)
            {
                for( k = 0 ; k < AM; k++){
                Q[col*N1MAX*N2MAX+k*N2MAX+j] -= R_s[tid*segR+j-i]*Q[col*N1MAX*N2MAX+k*N2MAX+i] ;
                }
            }
            
            __syncthreads();
        
        }//for i
        
    }//for col
    
}


template<unsigned int WarpSize>
__global__ void ComputeTildeE_iter_SSPAIv1(int *E, int *I, int *iPTR, int N1MAX, int nCol, int sK)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize-1);//index of threads in warp
    
    int col, i, AM, colK;
    
    for(col = warp_id; col < nCol; col += offset)
    {
        colK = sK + col;
        E[col] = -1;
        AM = iPTR[col];
        for(i = lane ; i < AM ; i += WarpSize)
        {
            if(I[col*N1MAX+i] == colK){
                E[col] = i;
                break;
            }
        }//for i 
    }//for col
}



template<unsigned int WarpSize>
__global__ void Sol_iter_SSPAIv1(double *Q, double *R, double *X, int *E, int *jPTR, int N1MAX, int N2MAX, int nCol)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize-1);//index of threads in warp
    
    int col, AN, i, j;
    
    //compute Q^TE
    for(col = warp_id; col < nCol; col += offset)
    {
        AN = jPTR[col];
        if(E[col] == -1)
        {
            for(i = lane ; i < AN ; i += WarpSize)	
            {
                X[col*N2MAX+i] = 0.0;
            }//for i
        }else{
            for(i = lane ; i < AN ; i += WarpSize)
            {
                X[col*N2MAX+i] = Q[col*N1MAX*N2MAX+E[col]*N2MAX+i];
            }//for i
        }
        
        __syncthreads();
        
        //solving the upper triangular system
        for(i = AN - 1; i >= 0 ; i--)
        {
            if(lane == 0){
                X[col*N2MAX+i] /= R[col*N2MAX*N2MAX+i*N2MAX+i];
            }
        
            __syncthreads();
        
            for(j = lane ; j < i; j += WarpSize)
            {
                X[col*N2MAX+j] -= R[col*N2MAX*N2MAX+j*N2MAX+i] * X[col*N2MAX+i];//csr
                //X[col*N2MAX+j] -= R[col*N2MAX*N2MAX+i*N2MAX+j] * X[col*N2MAX+i]; //csc
            }// for j
            __syncthreads();
        }//for i
        
    }//for col
  
}


template<unsigned int WarpSize>
__global__ void modifyData_iter_SSPAIv1(double *mData, int *mPtr, 
                       double *X, int *jPTR, int n2max, int nCol, int sK)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize-1);// index of threads in warp
    
    int col, j;
    int col_s, colK;
    
    for(col = warp_id; col < nCol; col += offset)
    {
        colK = col + sK;
        col_s = mPtr[colK];
        for(j = lane ; j < jPTR[col] ; j += WarpSize)
        {
            mData[col_s + j] = X[col * n2max + j];
        }
    }
}



//---once compute
template<unsigned int WarpSize>
__global__ void computeJ_SSPAIv1(int *MPtr, int *MIndex, 
                         int nCol, int *J, int *jPTR, int N2MAX)
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
        }
    }//for col
  
} 



template<unsigned int WarpSize, unsigned int CounterSize>
__global__ void computeI_SSPAIv1(int *APtr, int *AIndex, int nCol, 
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
__global__ void ComputeTildeACSR_SSPAIv1(double *A1, double *AData, int *APtr, int *AIndex, int nCol, 
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



template<unsigned int WarpSize, unsigned int SIZE_R_SHARED>
__global__ void QR_RShared_SSPAIv1(double *Q, double *R, int *iPTR, int *jPTR, int N1MAX, int N2MAX, int nCol)
{
    __shared__ double R_s[SIZE_R_SHARED];
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize-1);//index of threads in warp
    
    int tid = threadIdx.x / WarpSize;
    
    int col, AM, AN;
    int i, j, k;
    double rii, tR;
    
    int segR = SIZE_R_SHARED * WarpSize / blockDim.x ;
    
    for(col = warp_id; col < nCol; col += offset)
    {
        //for each col corresponding to a tilde of A
        AM = iPTR[col];
        AN = jPTR[col];
        
        for(i = 0 ; i < AN; i++)
        {
        
            /*Compute R in parallel an put them into shared memory*/
            for(j = lane + i; j < AN; j += WarpSize)
            {
                tR = 0.0;
                for(k = 0 ; k < AM ; k++)
                {
                    tR += Q[col*N1MAX*N2MAX + i + k*N2MAX] * Q[col*N1MAX*N2MAX + j + k*N2MAX];
                }//for k
                
                R_s[tid*segR+j-i] = tR ;
            }//for j
            
            __syncthreads();
            
            rii = sqrt(R_s[tid*segR]);
            //normalize column i of Q
            for(j = lane ; j < AM; j += WarpSize)
            {
                Q[col*N1MAX*N2MAX+j*N2MAX+i] /= rii ;
            }//for j
            
            __syncthreads();
            
            //compute projection factors
            for(j = lane + i; j < AN; j += WarpSize)
            {
                R_s[tid*segR+j-i] /= rii;
                R[col*N2MAX*N2MAX+i*N2MAX+j] = R_s[tid*segR+j-i] ;
            }//for j
            
            __syncthreads();
            
            for(j = lane + i + 1; j < AN; j += WarpSize)
            {
                for( k = 0 ; k < AM; k++){
                    Q[col*N1MAX*N2MAX+k*N2MAX+j] -= R_s[tid*segR+j-i]*Q[col*N1MAX*N2MAX+k*N2MAX+i] ;
                }
            }
            
            __syncthreads();
        
        }//for i
        
    }//for col
    
}



template<unsigned int WarpSize>
__global__ void ComputeTildeE_SSPAIv1(int *E, int *I, int *iPTR, int N1MAX, int nCol)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize-1);//index of threads in warp
    
    int col, i, AM;
    
    for(col = warp_id; col < nCol; col += offset)
    {
        E[col] = -1;
        AM = iPTR[col];
        for(i = lane ; i < AM ; i += WarpSize)
        {
            if(I[col*N1MAX+i] == col){
                E[col] = i;
                break;
            }
        }//for i 
    }//for col
}



template<unsigned int WarpSize>
__global__ void Sol_SSPAIv1(double *Q, double *R, double *X, int *E, int *jPTR, int N1MAX, int N2MAX, int nCol)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize-1);//index of threads in warp
    
    int col, AN, i, j;
    
    //compute Q^TE
    for(col = warp_id; col < nCol; col += offset)
    {
        AN = jPTR[col];
        if(E[col] == -1)
        {
            for(i = lane ; i < AN ; i += WarpSize)	
            {
                X[col*N2MAX+i] = 0.0;
            }//for i
        }else{
            for(i = lane ; i < AN ; i += WarpSize)
            {
                X[col*N2MAX+i] = Q[col*N1MAX*N2MAX+E[col]*N2MAX+i];
            }//for i
        }
        
        __syncthreads();
        
        //solving the upper triangular system
        for(i = AN - 1; i >= 0 ; i--)
        {
            if(lane == 0){
                X[col*N2MAX+i] /= R[col*N2MAX*N2MAX+i*N2MAX+i];
            }
        
            __syncthreads();
            
            for(j = lane ; j < i; j += WarpSize)
            {
                X[col*N2MAX+j] -= R[col*N2MAX*N2MAX+j*N2MAX+i] * X[col*N2MAX+i];//csr
                //X[col*N2MAX+j] -= R[col*N2MAX*N2MAX+i*N2MAX+j] * X[col*N2MAX+i]; //csc
            }// for j
            __syncthreads();
        }//for i
        
    }//for col
  
}


template<unsigned int WarpSize>
__global__ void modifyData_SSPAIv1(double *mData, int *mPtr, 
                       double *X, int *jPTR, int n2max, int nCol)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize-1);// index of threads in warp
    
    int col, j;
    int col_s;
    
    for(col = warp_id; col < nCol; col += offset)
    {
        col_s = mPtr[col];
        for(j = lane ; j < jPTR[col] ; j += WarpSize)
        {
            mData[col_s + j] = X[col * n2max + j];
        }
    }
}




#endif