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

#ifndef INIT_H
#define INIT_H

#include "dataType.h"

using namespace std;

/* This method is used to compute N1_MAX
 * @CSC_M: A matrix with CSC format
 */
int computeN1MAX(CSC_Matrix *CSC_M){
	
	int n1max = 0;
	vector<int> p;
	
	for(int i = 0 ; i < CSC_M->n ; i++){
		int cn1 = 0;
		
		int c_s = CSC_M->mPtr[i];
		int c_e = CSC_M->mPtr[i+1];
		
		
		p.clear();
		for(int j = c_s ; j < c_e ; j++){
			int iCol = CSC_M->mIndex[c_s];
		  int iCol_s = CSC_M->mPtr[iCol] ;
	    int iCol_e = CSC_M->mPtr[iCol + 1] ;
	    if( j == c_s ){
	    	for(int 
	    }else{
	    	d
	    }
		}
		
	}
	
	return n1max
	
}

/* This method is used to compute N2_MAX
 * @CSC_M: A matrix with CSC format
 */
int computeN2MAX(CSC_Matrix *CSC_M){
	
	int n2max = 0 ;
   
  for(int col = 0 ; col < CSC_M->n ; col++){
    int tempn2 = CSC_M->mPtr[col+1] - CSC_M->mPtr[col];
    if( tempn2 > n2max ) n2max = tempn2;
  }
   
  return n2max ;
	
}


#endif