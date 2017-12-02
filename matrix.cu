#include "matrix.h"
#include "helper_cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"


void createHostMatrix(Matrix *matrix, int numRows, int numColums, int size)
{
	matrix->numRows = numRows;
	matrix->numColumns = numColums;
	matrix->elements = (int *)malloc(size);
}
void createDeviceMatrix(Matrix *matrix, int numRows, int numColums, int size)
{
	matrix->numRows = numRows;
	matrix->numColumns = numColums;
	matrix->elements = NULL;
	checkCudaErrors(cudaMalloc((void**)&(matrix->elements), size));
}
void createHostMatrixNoAllocation(Matrix *matrix, int numRows, int numColums, int size)
{
	matrix->numRows = numRows;
	matrix->numColumns = numColums;
	matrix->elements = NULL;
}
void showMatrix(Matrix A, char* name)
{
	//printf("Matrix A \n");
	printf("%s\n", name);
	for (int j = 0; j < A.numRows; j++)
	{
		for (int i = 0; i < A.numColumns; i++)
		{
			printf("%3d ", A.elements[j*A.numColumns + i]);
		}
		printf("\n");
	}
}
void verifyHostAllocation(Matrix h_A)
{
	if (h_A.elements == NULL)
	{
		fprintf(stderr, "Failed to allocate host matrix\n");
		exit(EXIT_FAILURE);
	}
}