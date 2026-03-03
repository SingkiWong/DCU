/* -----------------------------------------------------------------------------
    The programming is licensed to you under the NJNU Consortium license:
    Copyright (C)  2012  Jiaquan Gao 
	  E-mail: gaojiaquan@njnu.edu.cn

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
 *  This .h file is used to sort
** ---------------------------------------------------------------------------*/

#ifndef SORT_H
#define SORT_H

void Swap(int *a, int i, int j)
{
  int temp = a[i];
  a[i] = a[j];
  a[j] = temp;
}

//---for BubbleSort
void BubbleSort(int *a, int n)  
{
  for(int j = 0 ; j < n -1 ; j++)
  {
    for(int i = 0 ; i < n -1 - j; i++)
    {
      if(a[i] > a[i+1])  Swap(a, i, i + 1);
    }
  } 
}  

int Partition(int *a, int left, int right)  // 划分函数
{
  int pivot = a[right];               // 这里每次都选择最后一个元素作为基准
  int tail = left - 1;                // tail为小于基准的子数组最后一个元素的索引
  for (int i = left; i < right; i++)  // 遍历基准以外的其他元素
  {
      if (a[i] <= pivot)              // 把小于等于基准的元素放到前一个子数组末尾
      {
          //Swap(a, ++tail, i);
          ++tail;
          if( tail != i) Swap(a, tail, i);
      }
  }
  Swap(a, tail + 1, right);           // 最后把基准放到前一个子数组的后边，剩下的子数组既是大于基准的子数组
                                      // 该操作很有可能把后面元素的稳定性打乱，所以快速排序是不稳定的排序算法
  return tail + 1;                    // 返回基准的索引
}

//---for QuickSort
void QuickSort(int *a, int left, int right)
{
  if (left >= right)
    return;
  int pivot_index = Partition(a, left, right);  // 基准的索引
  QuickSort(a, left, pivot_index - 1);
  QuickSort(a, pivot_index + 1, right);
}

#endif

