#include "morfological.h"

#include "cuda_runtime_api.h"
#include "helper_cuda.h"//-I$(NVCUDASAMPLES_ROOT)/common/inc
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_timer.h"


void copy(Matrix structuringElement)
{
	checkCudaErrors(cudaMemcpyToSymbol(structuringElements, structuringElement.elements, strucElDim*strucElDim * sizeof(uint8_t), 0, cudaMemcpyHostToDevice));
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
				structuringElement.elements[i] = (uint8_t)1;
			}
			else
			{
				structuringElement.elements[i] = (uint8_t)0;
			}
		}
	}
}
Matrix* negation(Matrix A)
{
	Matrix* result = (Matrix*)malloc(sizeof(Matrix));
	createHostMatrix(result, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(uint8_t));
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


__global__ void dilatation_cuda(Matrix A, Matrix result)
{
	int column = threadIdx.x+strucElDim/2;
	int row = threadIdx.y+strucElDim/2;

	__shared__ uint8_t dilTile[(blockD+strucElDim-1)*(blockD+strucElDim-1)];
	
	dilTile[threadIdx.x + blockDim.x*threadIdx.y] = A.elements[threadIdx.x + A.numColumns*threadIdx.y + blockIdx.x*blockD + A.numColumns*blockD*blockIdx.y];
	__syncthreads();

	if (column < blockDim.x-strucElDim/2 && row < blockDim.y-strucElDim/2)
	{
		uint8_t subMatrix[strucElDim*strucElDim];
		for (int i = 0; i < strucElDim*strucElDim; i++)
		{
			subMatrix[i] = 0;
		}
		int index;
		uint8_t CValue;

		index = row * blockDim.x + column;
		CValue = 0;

		for (int i = -(strucElDim/2); i < strucElDim/2; i++)
		{
			for (int j = -(strucElDim/2); j < strucElDim/2; j++)
			{			
				subMatrix[j+strucElDim/2 + strucElDim * (i+strucElDim/2)] = dilTile[index + j + i*blockDim.x];
			}
		}

		for (int i = 0; i < strucElDim*strucElDim; i++)
		{
			if (structuringElements[i] * subMatrix[i] == 1)
				CValue = 1;
		}


		result.elements[threadIdx.x + strucElDim/2 + A.numColumns*(threadIdx.y+strucElDim/2) + blockIdx.x*blockD + A.numColumns*blockD*blockIdx.y] = CValue;
		

	}
	__syncthreads();

}

Matrix* dilatation(Matrix A, Matrix structuringElement)
{
	StopWatchInterface *timer = NULL;

	Matrix* result = (Matrix*)malloc(sizeof(Matrix));
	createHostMatrix(result, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(uint8_t));
	verifyHostAllocation(*result);
	Matrix d_A;
	Matrix d_structuringElement;
	Matrix d_result;
	createDeviceMatrix(&d_A, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(uint8_t));
	createDeviceMatrix(&d_structuringElement, strucElDim, strucElDim, strucElDim*strucElDim* sizeof(uint8_t));
	createDeviceMatrix(&d_result, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(uint8_t));
	checkCudaErrors(cudaMemcpy(d_A.elements, A.elements, A.numColumns*A.numRows * sizeof(uint8_t), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_structuringElement.elements, structuringElement.elements, strucElDim*strucElDim * sizeof(uint8_t), cudaMemcpyHostToDevice));

	dim3 threads1(blockD + strucElDim - 1, blockD + strucElDim - 1);
	dim3 grid1(A.numColumns / blockD, A.numRows / blockD);

	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);


	sdkStartTimer(&timer);
	dilatation_cuda <<<grid1, threads1 >>> (d_A, d_result);
	sdkStopTimer(&timer);
	float czas = sdkGetTimerValue(&timer);
	printf("Processing time: %f ms\n", czas);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(result->elements, d_result.elements, A.numColumns*A.numRows * sizeof(uint8_t), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_A.elements));
	checkCudaErrors(cudaFree(d_structuringElement.elements));
	checkCudaErrors(cudaFree(d_result.elements));


	sdkDeleteTimer(&timer);

	return result;
}


