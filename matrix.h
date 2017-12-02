#ifndef __X_MATRIX__
#define __X_MATRIX__

#include <stdio.h>
#include <stdlib.h>
#include "helper_cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"


typedef struct
{
	int numRows;
	int numColumns;
	int* elements;
} Matrix;

void createHostMatrix(Matrix *matrix, int numRows, int numColums, int size);
void createDeviceMatrix(Matrix *matrix, int numRows, int numColums, int size);
void createHostMatrixNoAllocation(Matrix *matrix, int numRows, int numColums, int size);
void showMatrix(Matrix A, char* name);
void verifyHostAllocation(Matrix h_A);


#endif