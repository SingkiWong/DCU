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
 *  This .h file is used to define the data structure of matrix
** ---------------------------------------------------------------------------*/

#ifndef MATRIX_DATATYPE_H
#define MATRIX_DATATYPE_H

typedef struct {
	int *mIndex ;
	double *mData ;
	int *mPtr ;
	int nRow;
	int nCol;
	int n; //for square matrix
	int nonzeroes ;
} CSR_Matrix ;

typedef struct {
	int *mIndex ;
	double *mData ;
	int *mPtr ;
  int nRow;
  int nCol;
  int n; //for square matrix
	int nonzeroes ;
} CSC_Matrix ;

typedef struct {
	int nRow ;
	int nCol ;	
	int nDIG ;
	int nonzeroes ;
	int *mOffset ;
	double *mData ;
} DIA_Matrix ;

typedef struct {
	int nRow ;
	int nCol ;	
	int nonzeroes ;
	int *mIndex ;
	double *mData ;
} ELL_Matrix ;

typedef struct {
	int n ;
	int nonzeroes ;
	int *mI ;
	int *mJ ;
	double *mV ;
} COO_Matrix ;

typedef struct {
	double *vect ;
	int n   ;
} Vector ;

typedef struct {
	int n ;
	double *dgl ;
} DIAGS ;

#endif

