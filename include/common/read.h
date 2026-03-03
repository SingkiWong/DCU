/* -----------------------------------------------------------------------------
    The programming is licensed to you under the NJNU Consortium license:
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
 *  This .h file is used to read the sparse matrix from Florida Sparse Matrix 
 *  Collection
** ---------------------------------------------------------------------------*/

#include <string>
using namespace std;

#ifndef READFILE
#define READFILE

int computeNumOfValues(char *str)
{
	
	int spacePos = -1;
	int numOfValues = 0;
	for(int i = 0; i < strlen(str); i++){
		if(isspace(str[i])){
			if(i != spacePos && (i-1) != spacePos ){
				numOfValues++;
				spacePos = i;
			}
		}
	}

	return numOfValues;
}

/*
 * Read the non-symmetric matrix with no value and fill it with 1.0
 */
void readCSRNoValue(char *filename, int count, CSR_Matrix *CSR_A)
{
  char str[1024];
  FILE *fp;
  if((fp = fopen(filename,"r")) == NULL)
  {
		printf("can not open file\n");
		exit(0);
  }
  
  int sum = 0;
  while(!feof(fp))
  {
    char *status = fgets(str,1024, fp);
    sum++;
    
    if(sum > (count-2))
    {
      int matrixrow=0;
      int matrixcol=0;
      int nonzeroes=0;
      int rV = fscanf(fp,"%d %d %d\n", &matrixrow, &matrixcol, &nonzeroes);
      if( rV == EOF ){
      	printf("Error of reading data in readCSRNoValue\n");
      	exit(0); 
      }
      printf("row = %d, col = %d, nonzeroes = %d\n", matrixrow, matrixcol, nonzeroes);
      
      vector<vector<int> > mIndex(matrixrow);
      vector<vector<double> > mData(matrixrow);
      
      int i1, j1;
      for( int i = 0 ; i< nonzeroes; i++ )
      {
      	rV = fscanf(fp,"%d %d\n", &i1, &j1); //no value
      	if( rV == EOF ){
      		printf("Error of reading data in readCSRNoValue\n");
      		exit(0); 
        }
      	mIndex[i1-1].push_back(j1-1);
      	mData[i1-1].push_back(1.0);
      }
      
      CSR_A->n = matrixrow;
      CSR_A->nRow = matrixrow;
      CSR_A->nCol = matrixcol;
      CSR_A->nonzeroes =nonzeroes; 
      CSR_A->mData = ( double* )malloc( sizeof( double ) * CSR_A->nonzeroes ) ;
      CSR_A->mIndex = ( int* )malloc( sizeof( int ) * CSR_A->nonzeroes ) ;
      CSR_A->mPtr = ( int* )malloc( sizeof( int ) * (CSR_A->nRow + 1) );
      CSR_A->mPtr[0] = 0 ;
      int nonz = 0;
      for( int i = 0 ; i < CSR_A->nRow ; i++ )
      {
      	int size = mIndex[i].size();
      	for(int j = 0 ; j < size; j++){
      		CSR_A->mIndex[nonz + j] = mIndex[i][j];
      		CSR_A->mData[nonz + j] = mData[i][j];
      	}
      	nonz += size;
      	CSR_A->mPtr[i+1] = nonz;
      }
      //CSR_A->mPtr[matrixrow] = nonzeroes;
      
	    break;
    }
  }
  
  fclose(fp);
  
}

/*
 * Read the non-symmetric matrix with one value
 */
void readCSROneValue(char *filename, int count, CSR_Matrix *CSR_A)
{
  char str[1024];
  FILE *fp;
  if((fp = fopen(filename,"r")) == NULL)
  {
		printf("can not open file\n");
		exit(0);
  }
  
  int sum = 0;
  while(!feof(fp))
  {
    char *status = fgets(str,1024, fp);
    sum++;
    
    if(sum > (count-2))
    {
      int matrixrow=0;
      int matrixcol=0;
      int nonzeroes=0;
      int rV = fscanf(fp,"%d %d %d\n", &matrixrow, &matrixcol, &nonzeroes);
      if( rV == EOF ){
      	printf("Error of reading data in readCSROneValue\n");
      	exit(0); 
      }
      printf("before: row = %d, col = %d, nonzeroes = %d\n", matrixrow, matrixcol, nonzeroes);
      
      vector<vector<int> > mIndex(matrixrow);
      vector<vector<double> > mData(matrixrow);
      
      int i1, j1;
      double dt;
      int realnonzeroes = 0;
      for( int i = 0 ; i< nonzeroes; i++ )
      {
      	rV = fscanf(fp,"%d %d %lf\n", &i1, &j1, &dt); //one value
      	if( rV == EOF ){
      		printf("Error of reading data in readCSROneValue\n");
      		exit(0); 
      	}
        if( fabs(dt) == 0) continue;
      	mIndex[i1-1].push_back(j1-1);
      	mData[i1-1].push_back(dt);
      	realnonzeroes++;
      }
      
      nonzeroes = realnonzeroes;
      printf("after: row = %d, col = %d, nonzeroes = %d\n", matrixrow, matrixcol, nonzeroes);
      
      CSR_A->n = matrixrow;
      CSR_A->nRow = matrixrow;
      CSR_A->nCol = matrixcol;
      CSR_A->nonzeroes =nonzeroes; 
      CSR_A->mData = ( double* )malloc( sizeof( double ) * CSR_A->nonzeroes ) ;
      CSR_A->mIndex = ( int* )malloc( sizeof( int ) * CSR_A->nonzeroes ) ;
      CSR_A->mPtr = ( int* )malloc( sizeof( int ) * (CSR_A->nRow + 1) );
      CSR_A->mPtr[0] = 0 ;
      int nonz = 0;
      for( int i = 0 ; i < CSR_A->nRow ; i++ )
      {
      	int size = mIndex[i].size();
      	for(int j = 0 ; j < size; j++){
      		CSR_A->mIndex[nonz + j] = mIndex[i][j];
      		CSR_A->mData[nonz + j] = mData[i][j];
      	}
      	nonz += size;
      	CSR_A->mPtr[i+1] = nonz;
      }
      //CSR_A->mPtr[matrixrow] = nonzeroes;
      
	    break;
    }
  }
  
  fclose(fp);
}

/*
 * Read the non-symmetric matrix with two values
 */
void readCSRTwoValues(char *filename, int count, CSR_Matrix *CSR_A)
{
	char str[1024];
  FILE *fp;
  if((fp = fopen(filename,"r")) == NULL)
  {
		printf("can not open file\n");
		exit(0);
  }
  
  int sum = 0;
  while(!feof(fp))
  {
    char *status = fgets(str,1024, fp);
    sum++;
    
    if(sum > (count-2))
    {
      int matrixrow=0;
      int matrixcol=0;
      int nonzeroes=0;
      int rV = fscanf(fp,"%d %d %d\n", &matrixrow, &matrixcol, &nonzeroes);
      if( rV == EOF ){
      	printf("Error of reading data in readCSRTwoValues\n");
      	exit(0); 
      }
      printf("before: row = %d, col = %d, nonzeroes = %d\n", matrixrow, matrixcol, nonzeroes);
      
      vector<vector<int> > mIndex(matrixrow);
      vector<vector<double> > mData(matrixrow);
      
      int i1, j1;
      double dt, dt1;
      int realnonzeroes = 0;
      for( int i = 0 ; i< nonzeroes; i++ )
      {
      	rV = fscanf(fp,"%d %d %lf %lf\n", &i1, &j1, &dt, &dt1); //two values
      	if( rV == EOF ){
      		printf("Error of reading data in readCSRTwoValues\n");
      		exit(0); 
      	}
        if( fabs(dt) == 0) continue;
      	mIndex[i1-1].push_back(j1-1);
      	mData[i1-1].push_back(dt);
      	realnonzeroes++;
      }
      
      nonzeroes = realnonzeroes;
      printf("after: row = %d, col = %d, nonzeroes = %d\n", matrixrow, matrixcol, nonzeroes);
      
      CSR_A->n = matrixrow;
      CSR_A->nRow = matrixrow;
      CSR_A->nCol = matrixcol;
      CSR_A->nonzeroes =nonzeroes; 
      CSR_A->mData = ( double* )malloc( sizeof( double ) * CSR_A->nonzeroes ) ;
      CSR_A->mIndex = ( int* )malloc( sizeof( int ) * CSR_A->nonzeroes ) ;
      CSR_A->mPtr = ( int* )malloc( sizeof( int ) * (CSR_A->nRow + 1) );
      CSR_A->mPtr[0] = 0 ;
      int nonz = 0;
      for( int i = 0 ; i < CSR_A->nRow ; i++ )
      {
      	int size = mIndex[i].size();
      	for(int j = 0 ; j < size; j++){
      		CSR_A->mIndex[nonz + j] = mIndex[i][j];
      		CSR_A->mData[nonz + j] = mData[i][j];
      	}
      	nonz += size;
      	CSR_A->mPtr[i+1] = nonz;
      }
      //CSR_A->mPtr[matrixrow] = nonzeroes;
      
	    break;
    }
  }
  
  fclose(fp);
}

/*
 * Read the non-symmetric matrix with the following three type:
 * (1) no value
 * (2) one value
 * (3) two values
 */
void readCSR(char *filename, CSR_Matrix *CSR_A)
{
  FILE *fp;
  if((fp = fopen(filename,"r")) == NULL)
  {
     printf("can not open file\n");
	   exit(0);
  }
  
  char str[1024];
  int count = 0;
  //int space = 0;
  int numofvalues = 0;
  while(!feof(fp))
  {
  	char *status = fgets(str, 1024, fp);
  	count++;
  	if(str[0] != '%')
  	{
  	  status = fgets(str, 1024, fp);
  	  numofvalues = computeNumOfValues(str);
//  	  for(int i = 0; i < strlen(str); i++)
//  	  {
//  	  	if(isspace(str[i]))
//  	  	{
//          space++;
//  	  	}
//  	  }
      break;
  	}
  } 
  
  if(numofvalues==2)//no value 
  {
  	readCSRNoValue(filename, count, CSR_A);
  }
  else if(numofvalues==3)//one value
  {
  	readCSROneValue(filename, count, CSR_A);
  }
  else if(numofvalues==4)//two values
  {
  	readCSRTwoValues(filename,count,CSR_A);
  }else{
  	printf("Error in reading CSR !!!\n");
  	exit(0);
  }
  fclose(fp);
}


/*
 * Read the symmetric matrix with no value and fill the value with 1.0
 */
void readCSRfillingNoValue(char *filename,int count,CSR_Matrix *CSR_A)
{
   char str[1024];
   FILE *fp;
   if((fp = fopen(filename,"r")) == NULL)
   {
     printf("can not open file\n");
     exit(0);
   }
   
   int sum=0;
   int nonzeroes1=0;
   int matrixrow=0;
   int matrixcol=0;
   int nonzeroes=0;
   
   while(!feof(fp))
   {
	    char *status = fgets(str,1024, fp);
	    sum++;
	    if(sum>(count-2))
	    {
	      int rV = fscanf(fp,"%d %d %d\n", &matrixrow,&matrixcol,&nonzeroes); 
	      if( rV == EOF ){
      		printf("Error of reading data in readCSRfillingNoValue\n");
      		exit(0); 
      	}
	      printf("before:row = %d, col = %d, nonzeroes = %d\n", matrixrow, matrixcol, nonzeroes);
		    std::vector<std::vector<int> > tIndex(matrixrow);
	      std::vector<std::vector<double> > tData(matrixrow);  
		    int i1,j1;
		    while(!feof(fp))
		    {
	        rV = fscanf(fp,"%d %d\n", &i1, &j1);
	        if( rV == EOF ){
      			printf("Error of reading data in readCSRfillingNoValue\n");
      			exit(0); 
      		}
	        if(i1==j1)
	        {
	          tData[i1-1].push_back(1.0);
	          tIndex[i1-1].push_back(j1-1);
	          nonzeroes1++;
	        }
	        else{
	          tData[i1-1].push_back(1.0);
	          tIndex[i1-1].push_back(j1-1);
	          nonzeroes1++;
	          tData[j1-1].push_back(1.0);//another parts
	          tIndex[j1-1].push_back(i1-1);	
	          nonzeroes1++;
	        }
	      } 
	      
		    printf("after:row = %d, col = %d, nonzeroes = %d\n", matrixrow, matrixcol, nonzeroes1);
		   
		    CSR_A->n = matrixrow;
		    CSR_A->nRow = matrixrow;
        CSR_A->nCol = matrixcol;
		    CSR_A->nonzeroes =nonzeroes1; 
		    CSR_A->mData = ( double* )malloc( sizeof( double ) * CSR_A->nonzeroes ) ;
		    CSR_A->mIndex = ( int* )malloc( sizeof( int ) * CSR_A->nonzeroes ) ;
		    CSR_A->mPtr = ( int* )malloc( sizeof( int ) * (CSR_A->n + 1) );
		    CSR_A->mPtr[0] = 0 ;
		    int nonz = 0 ;
		    for(int row = 0 ; row < CSR_A->nRow; row++)
		    {
		    	 int size = tIndex[row].size();
					 for(int j = 0 ; j < size; j++)
					 {
					    CSR_A->mData[nonz + j] = tData[row][j] ;
					    CSR_A->mIndex[nonz + j] = tIndex[row][j] ;
					 }
					 nonz += size;
					 CSR_A->mPtr[row+1] = nonz ;
		    }
		    
		    //printf("MPTR = %d\n", CSR_A->mPtr[CSR_A->n]);
		    break;
      }
  
   }
   fclose(fp);
}


/*
 * Read the symmetric matrix with one value
 */
void readCSRfillingOneValue(char *filename, int count, CSR_Matrix *CSR_A)
{
	 char str[1024];
   FILE *fp;
   if((fp = fopen(filename,"r")) == NULL)
   {
     printf("can not open file\n");
     exit(0);
   }
   
   int sum=0;
   int nonzeroes1=0;
   int matrixrow=0;
   int matrixcol=0;
   int nonzeroes=0;
   
   while(!feof(fp))
   {
	    char *status = fgets(str,1024, fp);
	    sum++;
	    if(sum>(count-2))
	    {
	      int rV = fscanf(fp,"%d %d %d\n", &matrixrow,&matrixcol,&nonzeroes); 
	      if( rV == EOF ){
      			printf("Error of reading data in readCSRfillingOneValue\n");
      			exit(0); 
      	}
	      printf("before:row = %d, col = %d, nonzeroes = %d\n", matrixrow, matrixcol, nonzeroes);
		    std::vector<std::vector<int> > tIndex(matrixrow);
	      std::vector<std::vector<double> > tData(matrixrow);  
		    int i1,j1;
		    double dt;
		    while(!feof(fp))
		    {
	        rV = fscanf(fp,"%d %d %lf\n", &i1, &j1, &dt);
	        if( rV == EOF ){
      			printf("Error of reading data in readCSRfillingOneValue\n");
      			exit(0); 
      		}
	        if( fabs(dt) == 0) continue;
	        if(i1==j1)
	        {
	          tData[i1-1].push_back(dt);
	          tIndex[i1-1].push_back(j1-1);
	          nonzeroes1++;
	        }
	        else{
	          tData[i1-1].push_back(dt);
	          tIndex[i1-1].push_back(j1-1);
	          nonzeroes1++;
	          tData[j1-1].push_back(dt);//another parts
	          tIndex[j1-1].push_back(i1-1);	
	          nonzeroes1++;
	        }
	      } 
	      
		    printf("after:row = %d, col = %d, nonzeroes = %d\n", matrixrow, matrixcol, nonzeroes1);
		   
		    CSR_A->n = matrixrow;
		    CSR_A->nRow = matrixrow;
        CSR_A->nCol = matrixcol;
		    CSR_A->nonzeroes =nonzeroes1; 
		    CSR_A->mData = ( double* )malloc( sizeof( double ) * CSR_A->nonzeroes ) ;
		    CSR_A->mIndex = ( int* )malloc( sizeof( int ) * CSR_A->nonzeroes ) ;
		    CSR_A->mPtr = ( int* )malloc( sizeof( int ) * (CSR_A->n + 1) );
		    CSR_A->mPtr[0] = 0 ;
		    int nonz = 0 ;
		    for(int row = 0 ; row < CSR_A->nRow; row++)
		    {
		    	 int size = tIndex[row].size();
					 for(int j = 0 ; j < size; j++)
					 {
					    CSR_A->mData[nonz + j] = tData[row][j] ;
					    CSR_A->mIndex[nonz + j] = tIndex[row][j] ;
					 }
					 nonz += size;
					 CSR_A->mPtr[row+1] = nonz ;
		    }
		    
		    //printf("MPTR = %d\n", CSR_A->mPtr[CSR_A->n]);
		    break;
      }
  
   }
   fclose(fp);

}

/*
 * Read the symmetric matrix with two values
 */
void readCSRfillingTwoValues(char *filename, int count, CSR_Matrix *CSR_A)
{
  char str[1024];
   FILE *fp;
   if((fp = fopen(filename,"r")) == NULL)
   {
     printf("can not open file\n");
     exit(0);
   }
   
   int sum=0;
   int nonzeroes1=0;
   int matrixrow=0;
   int matrixcol=0;
   int nonzeroes=0;
   
   while(!feof(fp))
   {
	    char *status = fgets(str,1024, fp);
	    sum++;
	    if(sum>(count-2))
	    {
	      int rV = fscanf(fp,"%d %d %d\n", &matrixrow,&matrixcol,&nonzeroes); 
	      if( rV == EOF ){
      			printf("Error of reading data in readCSRfillingTwoValues\n");
      			exit(0); 
      	}
	      printf("before:row = %d, col = %d, nonzeroes = %d\n", matrixrow, matrixcol, nonzeroes);
		    std::vector<std::vector<int> > tIndex(matrixrow);
	      std::vector<std::vector<double> > tData(matrixrow);  
		    int i1,j1;
		    double dt,dt1;
		    while(!feof(fp))
		    {
	        rV = fscanf(fp,"%d %d %lf %lf\n", &i1, &j1, &dt, &dt1);
	        if( rV == EOF ){
      			printf("Error of reading data in readCSRfillingTwoValues\n");
      			exit(0); 
      		}
	        if( fabs(dt) == 0) continue;
	        if(i1==j1)
	        {
	          tData[i1-1].push_back(dt);
	          tIndex[i1-1].push_back(j1-1);
	          nonzeroes1++;
	        }
	        else{
	          tData[i1-1].push_back(dt);
	          tIndex[i1-1].push_back(j1-1);
	          nonzeroes1++;
	          tData[j1-1].push_back(dt);//another parts
	          tIndex[j1-1].push_back(i1-1);	
	          nonzeroes1++;
	        }
	      } 
	      
		    printf("after:row = %d, col = %d, nonzeroes = %d\n", matrixrow, matrixcol, nonzeroes1);
		   
		    CSR_A->n = matrixrow;
		    CSR_A->nRow = matrixrow;
        CSR_A->nCol = matrixcol;
		    CSR_A->nonzeroes =nonzeroes1; 
		    CSR_A->mData = ( double* )malloc( sizeof( double ) * CSR_A->nonzeroes ) ;
		    CSR_A->mIndex = ( int* )malloc( sizeof( int ) * CSR_A->nonzeroes ) ;
		    CSR_A->mPtr = ( int* )malloc( sizeof( int ) * (CSR_A->n + 1) );
		    CSR_A->mPtr[0] = 0 ;
		    int nonz = 0 ;
		    for(int row = 0 ; row < CSR_A->nRow; row++)
		    {
		    	 int size = tIndex[row].size();
					 for(int j = 0 ; j < size; j++)
					 {
					    CSR_A->mData[nonz + j] = tData[row][j] ;
					    CSR_A->mIndex[nonz + j] = tIndex[row][j] ;
					 }
					 nonz += size;
					 CSR_A->mPtr[row+1] = nonz ;
		    }
		    
		    //printf("MPTR = %d\n", CSR_A->mPtr[CSR_A->n]);
		    break;
      }
  
   }
   fclose(fp);
}

/*
 * Read the symmetric matrix with the following three type:
 * (1) no value
 * (2) one value
 * (3) two values
 */
void readCSRBYfilling(char *filename, CSR_Matrix *CSR_A)
{
  FILE *fp;
  
  if((fp = fopen(filename,"r")) == NULL)
  {
     printf("can not open file\n");
	 exit(0);
  }
  
  char str[1024];
  int count = 0; //record the row number of comments
  int numofvalues = 0;
  while(!feof(fp))
  {
  	char *status = fgets(str, 1024, fp);
  	count++;
  	if(str[0] != '%')
  	{
  	  status = fgets(str, 1024, fp);
  	  numofvalues = computeNumOfValues(str);
      break;
  	}
  } 
  
  if(numofvalues == 2) //no value
  {
  	readCSRfillingNoValue(filename, count, CSR_A);
  }
  else if(numofvalues==3) //one value
  {
  	readCSRfillingOneValue(filename, count, CSR_A);
  }
  else if(numofvalues==4) //two values
  {
  	readCSRfillingTwoValues(filename, count, CSR_A); 
  }else{
  	printf("Error in reading CSRfilling !!!\n");
  	exit(0);
  }
  fclose(fp);
}

/*
 * Read the matrix and store it by the CSR format
 */
void readMatrixToCSR(char *filename, CSR_Matrix *CSR_A)
{
  FILE *fp;
  if((fp = fopen(filename,"r")) == NULL)
  {
     printf("can not open file\n");
	   exit(0);
  }
  
  char str[1024];
  char *status = fgets(str, 1024, fp);
  
  string str1 = str;
  int loc = str1.find("symmetric");
  
  fclose(fp);
  
  if(loc != string::npos)
  {
  	readCSRBYfilling(filename,CSR_A); //symmetric
  }
  else{
    readCSR(filename,CSR_A); //non-symmetric
  }
}


/*----------------------------------------------------------------------*
 *                   CSC storage format                                 *
 *----------------------------------------------------------------------*/

/*
 * Read the non-symmetric matrix with no value and fill it with 1.0
 */
void readCSCNoValue(char *filename, int count, CSC_Matrix *CSC_A)
{
  char str[1024];
  FILE *fp;
  if((fp = fopen(filename,"r")) == NULL)
  {
		printf("can not open file\n");
		exit(0);
  }
  printf("count = %d\n", count);
  int sum = 0;
  while(!feof(fp))
  {
    char *status = fgets(str,1024, fp);
    sum++;
    
    if(sum > (count-2))
    {
      int matrixrow=0;
      int matrixcol=0;
      int nonzeroes=0;
      int rV = fscanf(fp,"%d %d %d\n", &matrixrow, &matrixcol, &nonzeroes);
      if( rV == EOF ){
      		printf("Error of reading data in readCSCNoValue\n");
      		exit(0); 
      }
      printf("row = %d, col = %d, nonzeroes = %d\n", matrixrow, matrixcol, nonzeroes);
      
      vector<vector<int> > mIndex(matrixrow);
      vector<vector<double> > mData(matrixrow);
      
      int i1, j1;
      for( int i = 0 ; i< nonzeroes; i++ )
      {
      	rV = fscanf(fp,"%d %d\n", &i1, &j1); //no value
      	if( rV == EOF ){
      		printf("Error of reading data in readCSCNoValue\n");
      		exit(0); 
      	}
      	mIndex[j1-1].push_back(i1-1);
      	mData[j1-1].push_back(1.0);
      }
      
      CSC_A->n = matrixcol;
      CSC_A->nRow = matrixrow;
      CSC_A->nCol = matrixcol;
      CSC_A->nonzeroes =nonzeroes; 
      CSC_A->mData = ( double* )malloc( sizeof( double ) * CSC_A->nonzeroes ) ;
      CSC_A->mIndex = ( int* )malloc( sizeof( int ) * CSC_A->nonzeroes ) ;
      CSC_A->mPtr = ( int* )malloc( sizeof( int ) * (CSC_A->nCol + 1) );
      CSC_A->mPtr[0] = 0 ;
      int nonz = 0;
      for( int i = 0 ; i < CSC_A->nCol ; i++ )
      {
      	int size = mIndex[i].size();
      	for(int j = 0 ; j < size; j++){
      		CSC_A->mIndex[nonz + j] = mIndex[i][j];
      		CSC_A->mData[nonz + j] = mData[i][j];
      	}
      	nonz += size;
      	CSC_A->mPtr[i+1] = nonz;
      }
      //CSR_A->mPtr[matrixrow] = nonzeroes;
      
	    break;
    }
  }
  
  fclose(fp);
  
}

/*
 * Read the non-symmetric matrix with one value
 */
void readCSCOneValue(char *filename, int count, CSC_Matrix *CSC_A)
{
  char str[1024];
  FILE *fp;
  if((fp = fopen(filename,"r")) == NULL)
  {
		printf("can not open file\n");
		exit(0);
  }
  
  int sum = 0;
  while(!feof(fp))
  {
    char *status = fgets(str,1024, fp);
    sum++;
    
    if(sum > (count-2))
    {
      int matrixrow=0;
      int matrixcol=0;
      int nonzeroes=0;
      int rV = fscanf(fp,"%d %d %d\n", &matrixrow, &matrixcol, &nonzeroes);
      if( rV == EOF ){
      	printf("Error of reading data in readCSCOneValue\n");
      	exit(0); 
      }
      printf("before: row = %d, col = %d, nonzeroes = %d\n", matrixrow, matrixcol, nonzeroes);
      
      vector<vector<int> > mIndex(matrixrow);
      vector<vector<double> > mData(matrixrow);
      
      int i1, j1;
      double dt;
      int realnonzeroes = 0;
      for( int i = 0 ; i< nonzeroes; i++ )
      {
      	rV = fscanf(fp,"%d %d %lf\n", &i1, &j1, &dt); //one value
      	if( rV == EOF ){
	      	printf("Error of reading data in readCSCOneValue\n");
	      	exit(0); 
	      }
      	if( fabs(dt) == 0) continue;
      	mIndex[j1-1].push_back(i1-1);
      	mData[j1-1].push_back(dt);
      	realnonzeroes++;
      }
      
      nonzeroes = realnonzeroes;
      printf("after: row = %d, col = %d, nonzeroes = %d\n", matrixrow, matrixcol, nonzeroes);
      
      CSC_A->n = matrixcol;
      CSC_A->nRow = matrixrow;
      CSC_A->nCol = matrixcol;
      CSC_A->nonzeroes =nonzeroes; 
      CSC_A->mData = ( double* )malloc( sizeof( double ) * CSC_A->nonzeroes ) ;
      CSC_A->mIndex = ( int* )malloc( sizeof( int ) * CSC_A->nonzeroes ) ;
      CSC_A->mPtr = ( int* )malloc( sizeof( int ) * (CSC_A->nCol + 1) );
      CSC_A->mPtr[0] = 0 ;
      int nonz = 0;
      for( int i = 0 ; i < CSC_A->nCol ; i++ )
      {
      	int size = mIndex[i].size();
      	for(int j = 0 ; j < size; j++){
      		CSC_A->mIndex[nonz + j] = mIndex[i][j];
      		CSC_A->mData[nonz + j] = mData[i][j];
      	}
      	nonz += size;
      	CSC_A->mPtr[i+1] = nonz;
      }
      //CSR_A->mPtr[matrixrow] = nonzeroes;
      
	    break;
    }
  }
  
  fclose(fp);
}

/*
 * Read the non-symmetric matrix with two values
 */
void readCSCTwoValues(char *filename, int count, CSC_Matrix *CSC_A)
{
	char str[1024];
  FILE *fp;
  if((fp = fopen(filename,"r")) == NULL)
  {
		printf("can not open file\n");
		exit(0);
  }
  
  int sum = 0;
  while(!feof(fp))
  {
    char *status = fgets(str,1024, fp);
    sum++;
    
    if(sum > (count-2))
    {
      int matrixrow=0;
      int matrixcol=0;
      int nonzeroes=0;
      int rV = fscanf(fp,"%d %d %d\n", &matrixrow, &matrixcol, &nonzeroes);
      if( rV == EOF ){
      	printf("Error of reading data in readCSCTwoValues\n");
      	exit(0); 
      }
      printf("before: row = %d, col = %d, nonzeroes = %d\n", matrixrow, matrixcol, nonzeroes);
      
      vector<vector<int> > mIndex(matrixrow);
      vector<vector<double> > mData(matrixrow);
      
      int i1, j1;
      double dt, dt1;
      int realnonzeroes = 0;
      for( int i = 0 ; i< nonzeroes; i++ )
      {
      	rV = fscanf(fp,"%d %d %lf %lf\n", &i1, &j1, &dt, &dt1); //two values
      	if( rV == EOF ){
	      	printf("Error of reading data in readCSCTwoValues\n");
	      	exit(0); 
	      }
      	if( fabs(dt) == 0) continue;
      	mIndex[j1-1].push_back(i1-1);
      	mData[j1-1].push_back(dt);
      	realnonzeroes++;
      }
      
      nonzeroes = realnonzeroes;
      printf("after: row = %d, col = %d, nonzeroes = %d\n", matrixrow, matrixcol, nonzeroes);
      
      CSC_A->n = matrixcol;
      CSC_A->nRow = matrixrow;
      CSC_A->nCol = matrixcol;
      CSC_A->nonzeroes =nonzeroes; 
      CSC_A->mData = ( double* )malloc( sizeof( double ) * CSC_A->nonzeroes ) ;
      CSC_A->mIndex = ( int* )malloc( sizeof( int ) * CSC_A->nonzeroes ) ;
      CSC_A->mPtr = ( int* )malloc( sizeof( int ) * (CSC_A->nCol + 1) );
      CSC_A->mPtr[0] = 0 ;
      int nonz = 0;
      for( int i = 0 ; i < CSC_A->nCol ; i++ )
      {
      	int size = mIndex[i].size();
      	for(int j = 0 ; j < size; j++){
      		CSC_A->mIndex[nonz + j] = mIndex[i][j];
      		CSC_A->mData[nonz + j] = mData[i][j];
      	}
      	nonz += size;
      	CSC_A->mPtr[i+1] = nonz;
      }
      //CSR_A->mPtr[matrixrow] = nonzeroes;
      
	    break;
    }
  }
  
  fclose(fp);
}

/*
 * Read the non-symmetric matrix with the following three type:
 * (1) no value
 * (2) one value
 * (3) two values
 */
void readCSC(char *filename, CSC_Matrix *CSC_A)//one value
{
 
  FILE *fp;
  if((fp = fopen(filename,"r")) == NULL)
  {
     printf("can not open file\n");
	 exit(0);
  }
  
  char str[1024];
  int count = 0;
  int numofvalues = 0;
  while(!feof(fp))
  {
  	char *status = fgets(str, 1024, fp);
  	count++;
  	if(str[0]!='%')
  	{
  	  status = fgets(str, 1024, fp);
  	  numofvalues = computeNumOfValues(str);
      break;
  	}
  } 
  
  if(numofvalues==2)//no value 
  {
  	readCSCNoValue(filename, count, CSC_A);
  }
  else if(numofvalues==3)//one value
  {
  	readCSCOneValue(filename, count, CSC_A);
  }
  else if(numofvalues==4)//two values
  {
  	readCSCTwoValues(filename,count,CSC_A);
  }else{
  	printf("Error in reading CSC !!!\n");
  	exit(0);
  }
  fclose(fp);
}

/*
 * Read the symmetric matrix with no value and fill the value with 1.0
 */
void readCSCfillingNoValue(char *filename,int count,CSC_Matrix *CSC_A)
{
   char str[1024];
   FILE *fp;
   if((fp = fopen(filename,"r")) == NULL)
   {
     printf("can not open file\n");
     exit(0);
   }
   
   int sum=0;
   int nonzeroes1=0;
   int matrixrow=0;
   int matrixcol=0;
   int nonzeroes=0;
   
   while(!feof(fp))
   {
	    char *status = fgets(str,1024, fp);
	    sum++;
	    if(sum>(count-2))
	    {
	      int rV = fscanf(fp,"%d %d %d\n", &matrixrow,&matrixcol,&nonzeroes); 
	      if( rV == EOF ){
	      	printf("Error of reading data in readCSCfillingNoValue\n");
	      	exit(0); 
	      }
	      printf("before:row = %d, col = %d, nonzeroes = %d\n", matrixrow, matrixcol, nonzeroes);
		    std::vector<std::vector<int> > tIndex(matrixrow);
	      std::vector<std::vector<double> > tData(matrixrow);  
		    int i1,j1;
		    while(!feof(fp))
		    {
	        rV = fscanf(fp,"%d %d\n", &i1, &j1);
	        if( rV == EOF ){
		      	printf("Error of reading data in readCSCfillingNoValue\n");
		      	exit(0); 
		      }
	        if(j1==i1)
	        {
	          tData[j1-1].push_back(1.0);
	          tIndex[j1-1].push_back(i1-1);
	          nonzeroes1++;
	        }
	        else{
	          tData[j1-1].push_back(1.0);
	          tIndex[j1-1].push_back(i1-1);
	          nonzeroes1++;
	          tData[i1-1].push_back(1.0);//another parts
	          tIndex[i1-1].push_back(j1-1);	
	          nonzeroes1++;
	        }
	      } 
	      
		    printf("after:row = %d, col = %d, nonzeroes = %d\n", matrixrow, matrixcol, nonzeroes1);
		   
		    CSC_A->n = matrixcol;
		    CSC_A->nRow = matrixrow;
        CSC_A->nCol = matrixcol;
		    CSC_A->nonzeroes =nonzeroes1; 
		    CSC_A->mData = ( double* )malloc( sizeof( double ) * CSC_A->nonzeroes ) ;
		    CSC_A->mIndex = ( int* )malloc( sizeof( int ) * CSC_A->nonzeroes ) ;
		    CSC_A->mPtr = ( int* )malloc( sizeof( int ) * (CSC_A->n + 1) );
		    CSC_A->mPtr[0] = 0 ;
		    int nonz = 0 ;
		    for(int col = 0 ; col < CSC_A->nCol; col++)
		    {
		    	 int size = tIndex[col].size();
					 for(int j = 0 ; j < size; j++)
					 {
					    CSC_A->mData[nonz + j] = tData[col][j] ;
					    CSC_A->mIndex[nonz + j] = tIndex[col][j] ;
					 }
					 nonz += size;
					 CSC_A->mPtr[col+1] = nonz ;
		    }
		    
		    break;
      }
  
   }
   fclose(fp);
}


/*
 * Read the symmetric matrix with one value
 */
void readCSCfillingOneValue(char *filename, int count, CSC_Matrix *CSC_A)
{
	 char str[1024];
   FILE *fp;
   if((fp = fopen(filename,"r")) == NULL)
   {
     printf("can not open file\n");
     exit(0);
   }
   
   int sum=0;
   int nonzeroes1=0;
   int matrixrow=0;
   int matrixcol=0;
   int nonzeroes=0;
   
   while(!feof(fp))
   {
	    char *status = fgets(str,1024, fp);
	    sum++;
	    if(sum>(count-2))
	    {
	      int rV = fscanf(fp,"%d %d %d\n", &matrixrow,&matrixcol,&nonzeroes); 
	      if( rV == EOF ){
	      	printf("Error of reading data in readCSCfillingOneValue\n");
	      	exit(0); 
	      }
	      printf("before:row = %d, col = %d, nonzeroes = %d\n", matrixrow, matrixcol, nonzeroes);
		    std::vector<std::vector<int> > tIndex(matrixrow);
	      std::vector<std::vector<double> > tData(matrixrow);  
		    int i1,j1;
		    double dt;
		    while(!feof(fp))
		    {
	        rV = fscanf(fp,"%d %d %lf\n", &i1, &j1, &dt);
	        if( rV == EOF ){
		      	printf("Error of reading data in readCSCfillingOneValue\n");
		      	exit(0); 
		      }
	        if( fabs(dt) == 0) continue;
	        if(j1==i1)
	        {
	          tData[j1-1].push_back(dt);
	          tIndex[j1-1].push_back(i1-1);
	          nonzeroes1++;
	        }
	        else{
	          tData[j1-1].push_back(dt);
	          tIndex[j1-1].push_back(i1-1);
	          nonzeroes1++;
	          tData[i1-1].push_back(dt);//another parts
	          tIndex[i1-1].push_back(j1-1);	
	          nonzeroes1++;
	        }
	      } 
	      
		    printf("after:row = %d, col = %d, nonzeroes = %d\n", matrixrow, matrixcol, nonzeroes1);
		   
		    CSC_A->n = matrixcol;
		    CSC_A->nRow = matrixrow;
        CSC_A->nCol = matrixcol;
		    CSC_A->nonzeroes =nonzeroes1; 
		    CSC_A->mData = ( double* )malloc( sizeof( double ) * CSC_A->nonzeroes ) ;
		    CSC_A->mIndex = ( int* )malloc( sizeof( int ) * CSC_A->nonzeroes ) ;
		    CSC_A->mPtr = ( int* )malloc( sizeof( int ) * (CSC_A->n + 1) );
		    CSC_A->mPtr[0] = 0 ;
		    int nonz = 0 ;
		    for(int col = 0 ; col < CSC_A->nCol; col++)
		    {
		    	 int size = tIndex[col].size();
					 for(int j = 0 ; j < size; j++)
					 {
					    CSC_A->mData[nonz + j] = tData[col][j] ;
					    CSC_A->mIndex[nonz + j] = tIndex[col][j] ;
					 }
					 nonz += size;
					 CSC_A->mPtr[col+1] = nonz ;
		    }
		    
		    break;
      }
  
   }
   fclose(fp);
}

/*
 * Read the symmetric matrix with two values
 */
void readCSCfillingTwoValues(char *filename, int count, CSC_Matrix *CSC_A)
{
  char str[1024];
   FILE *fp;
   if((fp = fopen(filename,"r")) == NULL)
   {
     printf("can not open file\n");
     exit(0);
   }
   
   int sum=0;
   int nonzeroes1=0;
   int matrixrow=0;
   int matrixcol=0;
   int nonzeroes=0;
   
   while(!feof(fp))
   {
	    char *status = fgets(str,1024, fp);
	    sum++;
	    if(sum>(count-2))
	    {
	      int rV = fscanf(fp,"%d %d %d\n", &matrixrow,&matrixcol,&nonzeroes); 
	      if( rV == EOF ){
	      	printf("Error of reading data in readCSCfillingTwoValues\n");
	      	exit(0); 
	      }
	      printf("before:row = %d, col = %d, nonzeroes = %d\n", matrixrow, matrixcol, nonzeroes);
		    std::vector<std::vector<int> > tIndex(matrixrow);
	      std::vector<std::vector<double> > tData(matrixrow);  
		    int i1,j1;
		    double dt, dt1;
		    while(!feof(fp))
		    {
	        rV = fscanf(fp,"%d %d %lf %lf\n", &i1, &j1, &dt, &dt1);
	        if( rV == EOF ){
		      	printf("Error of reading data in readCSCfillingTwoValues\n");
		      	exit(0); 
		      }
	        if( fabs(dt) == 0) continue;
	        if(j1==i1)
	        {
	          tData[j1-1].push_back(dt);
	          tIndex[j1-1].push_back(i1-1);
	          nonzeroes1++;
	        }
	        else{
	          tData[j1-1].push_back(dt);
	          tIndex[j1-1].push_back(i1-1);
	          nonzeroes1++;
	          tData[i1-1].push_back(dt);//another parts
	          tIndex[i1-1].push_back(j1-1);	
	          nonzeroes1++;
	        }
	      } 
	      
		    printf("after:row = %d, col = %d, nonzeroes = %d\n", matrixrow, matrixcol, nonzeroes1);
		   
		    CSC_A->n = matrixcol;
		    CSC_A->nRow = matrixrow;
        CSC_A->nCol = matrixcol;
		    CSC_A->nonzeroes =nonzeroes1; 
		    CSC_A->mData = ( double* )malloc( sizeof( double ) * CSC_A->nonzeroes ) ;
		    CSC_A->mIndex = ( int* )malloc( sizeof( int ) * CSC_A->nonzeroes ) ;
		    CSC_A->mPtr = ( int* )malloc( sizeof( int ) * (CSC_A->n + 1) );
		    CSC_A->mPtr[0] = 0 ;
		    int nonz = 0 ;
		    for(int col = 0 ; col < CSC_A->nCol; col++)
		    {
		    	 int size = tIndex[col].size();
					 for(int j = 0 ; j < size; j++)
					 {
					    CSC_A->mData[nonz + j] = tData[col][j] ;
					    CSC_A->mIndex[nonz + j] = tIndex[col][j] ;
					 }
					 nonz += size;
					 CSC_A->mPtr[col+1] = nonz ;
		    }
		    
		    break;
      }
  
   }
   fclose(fp);
}

/*
 * Read the symmetric matrix with the following three type:
 * (1) no value
 * (2) one value
 * (3) two values
 */
void readCSCBYfilling(char *filename, CSC_Matrix *CSC_A)
{
  FILE *fp;
  
  if((fp = fopen(filename,"r")) == NULL)
  {
     printf("can not open file\n");
	 exit(0);
  }
  
  char str[1024];
  int count = 0; //record the row number of comments
  int numofvalues = 0;
  while(!feof(fp))
  {
  	char *status = fgets(str, 1024, fp);
  	count++;
  	if(str[0] != '%')
  	{
  	  status = fgets(str, 1024, fp);
  	  numofvalues = computeNumOfValues(str);
      break;
  	}
  } 
  
  if(numofvalues == 2) //no value
  {
  	readCSCfillingNoValue(filename, count, CSC_A);
  }
  else if(numofvalues==3) //one value
  {
  	readCSCfillingOneValue(filename, count, CSC_A);
  }
  else if(numofvalues==4) //two values
  {
  	readCSCfillingTwoValues(filename, count, CSC_A); 
  }else{
  	printf("Error in reading CSCfilling !!!\n");
  	exit(0);
  }
  fclose(fp);
}
 
/*
 * Read the matrix and store it by the CSC format
 */
void readMatrixToCSC(char *filename, CSC_Matrix *CSC_A)
{
  FILE *fp;
  if((fp = fopen(filename,"r")) == NULL)
  {
     printf("can not open file\n");
	   exit(0);
  }
  
  char str[1024];
  char *status = fgets(str, 1024, fp);
  
  string str1 = str;
  int loc = str1.find("symmetric");
  
  
  fclose(fp);
  
  if(loc != string::npos)
  {
  	readCSCBYfilling(filename,CSC_A); //symmetric
  }
  else{
    readCSC(filename,CSC_A); //non-symmetric
  }
}
          
#endif
