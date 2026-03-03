#include <string>
using namespace std;

#ifndef WRITEMATRIX_H
#define WRITEMATRIX_H

void writeMatrixToCSC(char *filename, CSC_Matrix* A)
{
    FILE *fp;
    
    if((fp = fopen(filename,"wb")) == NULL)
    {
       printf("can not open file\n");
     exit(0);
    }

    int n = A->n;
    int nonzeroes = A->nonzeroes;
    
    char sentence[500] = "%%MatrixMarket matrix coordinate real general";
    fprintf(fp, "%s\n", sentence);

    fprintf(fp,"%d %d %d\n",n,n,nonzeroes);
  
    for (int nCol = 0; nCol < A->n; ++nCol)
    {
        int start = A->mPtr[nCol];
        int end = A -> mPtr[nCol+1];
        
        for (int i = start; i < end; ++i)
        {
          int row = A->mIndex[i]; //列坐标
          double value = A->mData[i];
          fprintf(fp,"%d %d %.16lf\n",row+1,nCol+1,value);
        }
       
    }

    fclose(fp);
}


void writeMatrixToCSR(char *filename, CSR_Matrix* A)
{
    FILE *fp;
    
    if((fp = fopen(filename,"wb")) == NULL)
    {
       printf("can not open file\n");
     exit(0);
    }

    int n = A->n;
    int nonzeroes = A->nonzeroes;
    
    fprintf(fp,"%d %d %d\n",n,n,nonzeroes);
  
    for (int nRow = 0; nRow < A->n; ++nRow)
    {
        int start = A->mPtr[nRow];
        int end = A -> mPtr[nRow+1];
        
        for (int i = start; i < end; ++i)
        {
          int col = A->mIndex[i]; //列坐标
          double value = A->mData[i];
          fprintf(fp,"%d %d %lf\n",nRow+1,col+1,value);
        }
       
    }

    fclose(fp);
}

#endif