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

int isExist(vector<int > p, int index){
	int flag = 0;
	for(int i = 0 ; i < p.size(); i++){
		if(p[i] == index){
			flag = 1;
			break;
		} 
	}
	//printf("index=%d, flag=%d\n", index, flag);
	return flag;
}

/* This method is used to compute N1_MAX
 * @CSC_M: A matrix with CSC format
 */
int computeN1MAX(CSC_Matrix *CSC_A, CSC_Matrix *CSC_M){
	
	int n1max = 0;
	vector<int > p;
	
	for(int i = 0 ; i < CSC_M->n ; i++){
		int c_s = CSC_M->mPtr[i];
		int c_e = CSC_M->mPtr[i+1];
		
		
		p.clear();
		for(int j = c_s ; j < c_e ; j++){
			int iCol = CSC_M->mIndex[j];
		  	int iCol_s = CSC_A->mPtr[iCol] ;
	    	int iCol_e = CSC_A->mPtr[iCol + 1] ;
	    if( j == c_s ){
	    	for(int j1 = iCol_s; j1 < iCol_e; j1++){
	    		p.push_back(CSC_A->mIndex[j1]);
	    	}
	    }else{
	    	for(int j1 = iCol_s; j1 < iCol_e; j1++){
	    		//printf("%lf \n", CSC_A->mData[j1]);
	    		int flag = isExist(p, CSC_A->mIndex[j1]);
	    		if( flag == 0) p.push_back(CSC_A->mIndex[j1]);
	    	}
	    }//end if
		}
		//printf("i = %d, rows=%d\n", i, p.size());
		if(p.size() > n1max) n1max = p.size();
		
	}
	
	return n1max;
	
}

/* This method is used to compute N2_MAX
 * @CSC_M: A matrix with CSC format
 */
int computeN2MAX(CSC_Matrix *CSC_M){
	
	int n2max = 0 ;
   
  for(int col = 0 ; col < CSC_M->n ; col++){
    int tempn2 = CSC_M->mPtr[col+1] - CSC_M->mPtr[col];
    //printf("col=%d,nonz=%d\n",col,tempn2);
    if( tempn2 > n2max ) n2max = tempn2;
  }
   
  return n2max ;
	
}


#endif