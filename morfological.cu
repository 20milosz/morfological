#include "morfological.h"

#include "cuda_runtime_api.h"
#include "helper_cuda.h"//-I$(NVCUDASAMPLES_ROOT)/common/inc
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void copy(Matrix structuringElement)
{
	checkCudaErrors(cudaMemcpyToSymbol(structuringElements, structuringElement.elements, strucElDim*strucElDim * sizeof(int), 0, cudaMemcpyHostToDevice));
}

void createStructuringElement(Matrix structuringElement)
{
	int i;
	for (int column = 0; column < strucElDim; column++)
	{
		for (int row = 0; row < strucElDim; row++)
		{
			i = column + strucElDim*row;
			if (row == strucElDim / 2 || column == strucElDim / 2)
			{
				structuringElement.elements[i] = 1;
			}
			else
			{
				structuringElement.elements[i] = 0;
			}
		}
	}
}

__global__ void dilatation_cuda(Matrix A, Matrix result)
{
	

	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	//
	__shared__ int dilTile[(blockD+strucElDim-1)*(blockD+strucElDim-1)];
	


	int subMatrix[strucElDim*strucElDim];
	for (int i = 0; i < strucElDim*strucElDim; i++)
	{
		subMatrix[i] = 0;
	}
	int index;
	int CValue;

	index = row * A.numColumns + column;
	CValue = 0;

	for (int i = 0; i < strucElDim; i++)
	{
		for (int j = 0; j < strucElDim; j++)
		{
			if ((column - j >= strucElDim / 2) && (row - i >= strucElDim / 2) && (column + j <= A.numColumns - strucElDim / 2) && (row + i <= A.numRows - strucElDim / 2))
			{
				subMatrix[j + strucElDim * i] = A.elements[index + j - strucElDim / 2 + (i - strucElDim / 2)*A.numColumns];
			}
			else
			{
				subMatrix[j + strucElDim * i] = 0;
			}
		}
	}

	for (int i = 0; i < strucElDim*strucElDim; i++)
	{
		if (structuringElements[i] * subMatrix[i] == 1)
			CValue = 1;
	}
	result.elements[index] = CValue;
	
}


Matrix* dilatation(Matrix A, Matrix structuringElement)
{
	
	Matrix* result = (Matrix*)malloc(sizeof(Matrix));
	createHostMatrix(result, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(int));
	verifyHostAllocation(*result);
	Matrix d_A;
	Matrix d_structuringElement;
	Matrix d_result;
	createDeviceMatrix(&d_A, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(int));
	createDeviceMatrix(&d_result, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(int));
	checkCudaErrors(cudaMemcpy(d_A.elements, A.elements, A.numColumns*A.numRows * sizeof(int), cudaMemcpyHostToDevice));

	dim3 threads(blockD, blockD);
	dim3 grid(A.numColumns / blockD, A.numRows / blockD);
	dilatation_cuda <<<grid, threads >>> (d_A, d_result);
	checkCudaErrors(cudaMemcpy(result->elements, d_result.elements, A.numColumns*A.numRows * sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_A.elements));
	checkCudaErrors(cudaFree(d_result.elements));

	return result;
}

Matrix* erosion(Matrix A, Matrix structuringElement)
{
	Matrix* result = (Matrix*)malloc(sizeof(Matrix));
	createHostMatrix(result, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(int));
	verifyHostAllocation(*result);
	int subMatrix[strucElDim*strucElDim];
	for (int i = 0; i < strucElDim*strucElDim; i++)
	{
		subMatrix[i] = 1;
	}
	int index;
	int CValue;
	for (int row = 0; row < A.numRows; row++)
	{
		for (int column = 0; column < A.numColumns; column++)
		{
			index = row * A.numColumns + column;
			CValue = 1;
			for (int i = 0; i < strucElDim; i++)
			{
				for (int j = 0; j < strucElDim; j++)
				{
					if ((column - j >= strucElDim / 2) && (row - i >= strucElDim / 2) && (column + j <= A.numColumns - strucElDim / 2) && (row + i <= A.numRows - strucElDim / 2))
					{
						subMatrix[j + strucElDim * i] = A.elements[index + j - strucElDim / 2 + (i - strucElDim / 2)*A.numColumns];
					}
					else
					{
						subMatrix[j + strucElDim * i] = 1;
					}
				}
			}

			for (int i = 0; i < strucElDim*strucElDim; i++)
			{
				if (structuringElement.elements[i] == 1 && subMatrix[i] == 0)
					CValue = 0;
			}
			result->elements[index] = CValue;
		}
	}
	return result;
}

Matrix* complement(Matrix A, Matrix B)
{
	Matrix* result = (Matrix*)malloc(sizeof(Matrix));
	createHostMatrix(result, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(int));
	verifyHostAllocation(*result);
	int index;
	for (int row = 0; row < A.numRows; row++)
	{
		for (int column = 0; column < A.numColumns; column++)
		{
			index = row * A.numColumns + column;
			result->elements[index] = A.elements[index] * B.elements[index];
		}
	}
	return result;
}

Matrix* negation(Matrix A)
{
	Matrix* result = (Matrix*)malloc(sizeof(Matrix));
	createHostMatrix(result, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(int));
	verifyHostAllocation(*result);
	int index;
	for (int row = 0; row < A.numRows; row++)
	{
		for (int column = 0; column < A.numColumns; column++)
		{
			index = row * A.numColumns + column;
			result->elements[index] = (A.elements[index] ? 0 : 1);

		}
	}
	return result;
}



Matrix* opening(Matrix A, Matrix structuringElement)
{
	Matrix* result = (Matrix*)malloc(sizeof(Matrix));
	Matrix* resultErosion = (Matrix*)malloc(sizeof(Matrix));
	createHostMatrixNoAllocation(result, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(int));
	createHostMatrixNoAllocation(resultErosion, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(int));
	resultErosion = erosion(A, structuringElement);
	result = dilatation(*resultErosion, structuringElement);
	free(resultErosion->elements);
	free(resultErosion);
	return result;
}

Matrix* closing(Matrix A, Matrix structuringElement)
{
	Matrix* result = (Matrix*)malloc(sizeof(Matrix));
	Matrix* resultDilatation = (Matrix*)malloc(sizeof(Matrix));
	createHostMatrixNoAllocation(result, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(int));
	createHostMatrixNoAllocation(resultDilatation, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(int));
	resultDilatation = dilatation(A, structuringElement);
	result = erosion(*resultDilatation, structuringElement);
	free(resultDilatation->elements);
	free(resultDilatation);
	return result;
}
int checkIfEqual(Matrix A, Matrix B)
{
	int isEqual = 1;
	for (int i = 0; i < A.numRows*A.numColumns; i++)
	{
		if (A.elements[i] != B.elements[i])
		{
			isEqual = 0;
		}
	}
	return isEqual;
}

Matrix* reconstruction(Matrix mask, Matrix marker, Matrix structuringElement)
{

	Matrix* result = (Matrix*)malloc(sizeof(Matrix));
	Matrix* resultDil = (Matrix*)malloc(sizeof(Matrix));
	Matrix* marker1 = (Matrix*)malloc(sizeof(Matrix));
	Matrix* marker2 = (Matrix*)malloc(sizeof(Matrix));
	createHostMatrixNoAllocation(result, mask.numRows, mask.numColumns, mask.numColumns*mask.numRows * sizeof(int));
	createHostMatrixNoAllocation(resultDil, mask.numRows, mask.numColumns, mask.numColumns*mask.numRows * sizeof(int));
	createHostMatrixNoAllocation(marker1, mask.numRows, mask.numColumns, mask.numColumns*mask.numRows * sizeof(int));
	createHostMatrixNoAllocation(marker2, mask.numRows, mask.numColumns, mask.numColumns*mask.numRows * sizeof(int));
	marker1 = &marker;
	resultDil = dilatation(*marker1, structuringElement);
	marker2 = complement(*resultDil, mask);


	marker1 = marker2;
	free(resultDil->elements);
	free(resultDil);
	resultDil = dilatation(*marker1, structuringElement);
	marker2 = complement(*resultDil, mask);


	while (!checkIfEqual(*marker1, *marker2))
	{
		free(marker1->elements);
		free(marker1);
		marker1 = marker2;
		free(resultDil->elements);
		free(resultDil);
		resultDil = dilatation(*marker1, structuringElement);
		marker2 = complement(*resultDil, mask);
	}
	free(marker1->elements);
	free(marker1);

	return marker2;
}


Matrix* openingByReconstruction(Matrix A, Matrix structuringElement)
{

	Matrix* resultEr = (Matrix*)malloc(sizeof(Matrix));
	Matrix* result = (Matrix*)malloc(sizeof(Matrix));
	createHostMatrixNoAllocation(resultEr, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(int));
	createHostMatrixNoAllocation(result, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(int));
	resultEr = erosion(A, structuringElement);
	result = reconstruction(A, *resultEr, structuringElement);
	free(resultEr->elements);
	free(resultEr);
	return result;
}


