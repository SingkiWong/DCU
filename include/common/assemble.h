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
 *  This .h file is used to initialize the static or dynamic preconditioners
** ---------------------------------------------------------------------------*/

#ifndef ASSEMBLE_H
#define ASSEMBLE_H

#include "dataType.h"

using namespace std;

/* ----------------------------------------------------------------------------
 * ++The method is used to transfer CSC into CSR format 
 * @parameters
 * @CSR_A:  the matrix with a compressed storage of rows (CSR)
 * @CSC_A:  the matrix with a compressed storage of columns (CSC)
** ---------------------------------------------------------------------------*/
void CSC2CSR(CSC_Matrix *CSC_A, CSR_Matrix *CSR_A, const unsigned int N)
{
	std::vector<std::vector<int> > tIndex(N);
	std::vector<std::vector<double> > tData(N);
	
	
	int nBegin, nEnd;
	for(int col = 0 ; col < CSC_A->nCol; col++){
	  nBegin = CSC_A->mPtr[col] ;
	  nEnd = CSC_A->mPtr[col+1] ;
	  
	  for(int j = nBegin ; j < nEnd ; j++){
	    tIndex[CSC_A->mIndex[j]].push_back(col) ;
	    tData[CSC_A->mIndex[j]].push_back(CSC_A->mData[j]) ;
	  }
	}
	
	CSR_A->nRow = N ;
	CSR_A->nCol = CSC_A->nCol ;
	CSR_A->n = N ;
	CSR_A->nonzeroes = CSC_A->nonzeroes ;
	

	printf( "nRow = %d \n", CSR_A->nRow ) ;
	
	CSR_A->mData = ( double* )malloc( sizeof( double ) * CSR_A->nonzeroes ) ;
	CSR_A->mIndex = ( int* )malloc( sizeof( int ) * CSR_A->nonzeroes ) ;
	CSR_A->mPtr = ( int* )malloc( sizeof( int ) * (CSR_A->nRow + 1) ) ;
	
	CSR_A->mPtr[0] = 0 ;
	int nonz = 0 ;
	for(int row = 0 ; row < CSR_A->nRow; row++){
	  for(int j = 0 ; j < tIndex[row].size(); j++){
	    CSR_A->mData[nonz] = tData[row][j] ;
	    CSR_A->mIndex[nonz] = tIndex[row][j] ;
	    nonz++;
	  }
	  CSR_A->mPtr[row+1] = nonz ;
	}
	
	
}

/* ----------------------------------------------------------------------------
 * ++The method is used to transfer CSC into CSR format 
 * @parameters
 * @CSR_M:  the matrix with a compressed storage of rows (CSR)
 * @CSC_M:  the matrix with a compressed storage of columns (CSC)
 * @X: solution
** ---------------------------------------------------------------------------*/
void assembleM(CSC_Matrix *CSC_M, CSR_Matrix *CSR_M, double *X, int N2MAX, const unsigned int N)
{
	std::vector<std::vector<int> > tIndex(N);
	std::vector<std::vector<double> > tData(N);
		
	
	int nBegin, nEnd;
	for(int col = 0 ; col < CSC_M->nCol; col++){
	  nBegin = CSC_M->mPtr[col] ;
	  nEnd = CSC_M->mPtr[col+1] ;
	  
	  for(int j = nBegin ; j < nEnd ; j++){
	    tIndex[CSC_M->mIndex[j]].push_back(col) ;
	    tData[CSC_M->mIndex[j]].push_back(X[col*N2MAX + j-nBegin]) ;
	  }
	}	
	
	CSR_M->nRow = N ;
	CSR_M->nCol = CSC_M->nCol ;
	CSR_M->n = N ;
	CSR_M->nonzeroes = CSC_M->nonzeroes ;
	

	printf( "nRow = %d \n", CSR_M->nRow ) ;
	
	CSR_M->mData = ( double* )malloc( sizeof( double ) * CSR_M->nonzeroes ) ;
	CSR_M->mIndex = ( int* )malloc( sizeof( int ) * CSR_M->nonzeroes ) ;
	CSR_M->mPtr = ( int* )malloc( sizeof( int ) * (CSR_M->nRow + 1) ) ;
	
	CSR_M->mPtr[0] = 0 ;
	int nonz = 0 ;
	for(int row = 0 ; row < CSR_M->nRow; row++){
	  for(int j = 0 ; j < tIndex[row].size(); j++){
	    CSR_M->mData[nonz] = tData[row][j] ;
	    CSR_M->mIndex[nonz] = tIndex[row][j] ;
	    nonz++;
	  }
	  CSR_M->mPtr[row+1] = nonz ;
	}
	
}


#endif