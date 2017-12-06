#include "morfological.h"

#include "cuda_runtime_api.h"
#include "helper_cuda.h"//-I$(NVCUDASAMPLES_ROOT)/common/inc
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

__global__ void erosion_cuda(Matrix A, Matrix result)
{
	int column = threadIdx.x + strucElDim / 2;
	int row = threadIdx.y + strucElDim / 2;

	__shared__ uint8_t dilTile[(blockD + strucElDim - 1)*(blockD + strucElDim - 1)];

	dilTile[threadIdx.x + blockDim.x*threadIdx.y] = A.elements[threadIdx.x + A.numColumns*threadIdx.y + blockIdx.x*blockD + A.numColumns*blockD*blockIdx.y];
	__syncthreads();

	if (column < blockDim.x - strucElDim / 2 && row < blockDim.y - strucElDim / 2)
	{
		uint8_t subMatrix[strucElDim*strucElDim];
		int index;
		uint8_t CValue;

		index = row * blockDim.x + column;
		CValue = 1;
		
		for (int i = -(strucElDim / 2); i <= strucElDim / 2; i++)
		{
			for (int j = -(strucElDim / 2); j <= strucElDim / 2; j++)
			{
				subMatrix[j + strucElDim / 2 + strucElDim * (i + strucElDim / 2)] = dilTile[index + j + i*blockDim.x];
			}
		}
		for (int i = 0; i < strucElDim*strucElDim; i++)
		{
			if (structuringElements[i] == 1 && subMatrix[i] == 0)
				CValue = 0;
		}
		result.elements[threadIdx.x + strucElDim / 2 + A.numColumns*(threadIdx.y + strucElDim / 2) + blockIdx.x*blockD + A.numColumns*blockD*blockIdx.y] = CValue;
	}
	__syncthreads();
}

__global__ void dilatation_cuda(Matrix A, Matrix result)
{
	int column = threadIdx.x + strucElDim / 2;
	int row = threadIdx.y + strucElDim / 2;

	__shared__ uint8_t dilTile[(blockD + strucElDim - 1)*(blockD + strucElDim - 1)];

	dilTile[threadIdx.x + blockDim.x*threadIdx.y] = A.elements[threadIdx.x + A.numColumns*threadIdx.y + blockIdx.x*blockD + A.numColumns*blockD*blockIdx.y];
	__syncthreads();

	if (column < blockDim.x - strucElDim / 2 && row < blockDim.y - strucElDim / 2)
	{
		uint8_t subMatrix[strucElDim*strucElDim];
		int index;
		uint8_t CValue;

		index = row * blockDim.x + column;
		CValue = 0;

		for (int i = -(strucElDim / 2); i <= strucElDim / 2; i++)
		{
			for (int j = -(strucElDim / 2); j <= strucElDim / 2; j++)
			{
				subMatrix[j + strucElDim / 2 + strucElDim * (i + strucElDim / 2)] = dilTile[index + j + i*blockDim.x];
			}
		}
		for (int i = 0; i < strucElDim*strucElDim; i++)
		{
			if (structuringElements[i] * subMatrix[i] == 1)
				CValue = 1;
		}
		result.elements[threadIdx.x + strucElDim / 2 + A.numColumns*(threadIdx.y + strucElDim / 2) + blockIdx.x*blockD + A.numColumns*blockD*blockIdx.y] = CValue;
	}
	__syncthreads();
}


Matrix* dilatation(Matrix A)
{
	Matrix* result = (Matrix*)malloc(sizeof(Matrix));
	createHostMatrix(result, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(uint8_t));
	verifyHostAllocation(*result);
	Matrix d_A;
	Matrix d_result;
	createDeviceMatrix(&d_A, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(uint8_t));
	createDeviceMatrix(&d_result, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(uint8_t));
	checkCudaErrors(cudaMemcpy(d_A.elements, A.elements, A.numColumns*A.numRows * sizeof(uint8_t), cudaMemcpyHostToDevice));
	dim3 threads1(blockD + strucElDim - 1, blockD + strucElDim - 1);
	dim3 grid1(A.numColumns / blockD, A.numRows / blockD);
	dilatation_cuda <<<grid1, threads1 >>> (d_A, d_result);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(result->elements, d_result.elements, A.numColumns*A.numRows * sizeof(uint8_t), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_A.elements));
	checkCudaErrors(cudaFree(d_result.elements));
	return result;
}

Matrix* erosion(Matrix A)
{
	Matrix* result = (Matrix*)malloc(sizeof(Matrix));
	createHostMatrix(result, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(uint8_t));
	verifyHostAllocation(*result);
	Matrix d_A;
	Matrix d_result;
	createDeviceMatrix(&d_A, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(uint8_t));
	createDeviceMatrix(&d_result, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(uint8_t));
	checkCudaErrors(cudaMemcpy(d_A.elements, A.elements, A.numColumns*A.numRows * sizeof(uint8_t), cudaMemcpyHostToDevice));
	dim3 threads1(blockD + strucElDim - 1, blockD + strucElDim - 1);
	dim3 grid1(A.numColumns / blockD, A.numRows / blockD);
	erosion_cuda << <grid1, threads1 >> > (d_A, d_result);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(result->elements, d_result.elements, A.numColumns*A.numRows * sizeof(uint8_t), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_A.elements));
	checkCudaErrors(cudaFree(d_result.elements));
	return result;
}
__global__ void complement_cuda(Matrix A, Matrix B, Matrix result)
{
	int column = threadIdx.x + blockIdx.x*blockDim.x;
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	if (column < A.numColumns && row < A.numRows)
	{
		int index = row * A.numColumns + column;
		result.elements[index] = A.elements[index] * B.elements[index];

	}

}

Matrix* complement(Matrix A, Matrix B)
{
	Matrix* result = (Matrix*)malloc(sizeof(Matrix));
	createHostMatrix(result, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(uint8_t));
	verifyHostAllocation(*result);
	Matrix d_A;
	Matrix d_B;
	Matrix d_result;
	createDeviceMatrix(&d_A, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(uint8_t));
	createDeviceMatrix(&d_B, B.numRows, B.numColumns, B.numColumns*B.numRows * sizeof(uint8_t));
	createDeviceMatrix(&d_result, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(uint8_t));
	checkCudaErrors(cudaMemcpy(d_A.elements, A.elements, A.numColumns*A.numRows * sizeof(uint8_t), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B.elements, B.elements, B.numColumns*B.numRows * sizeof(uint8_t), cudaMemcpyHostToDevice));

	dim3 threads(blockD + strucElDim - 1, blockD + strucElDim - 1);
	dim3 grid((A.numColumns + threads.x-1) / threads.x, (A.numRows+threads.y-1) / threads.y);
	complement_cuda << <grid, threads >> > (d_A, d_B, d_result);
	checkCudaErrors(cudaMemcpy(result->elements, d_result.elements, A.numColumns*A.numRows * sizeof(uint8_t), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_A.elements));
	checkCudaErrors(cudaFree(d_B.elements));
	checkCudaErrors(cudaFree(d_result.elements));

	return result;
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

Matrix* opening(Matrix A)
{
	Matrix* result = (Matrix*)malloc(sizeof(Matrix));
	Matrix* resultErosion = (Matrix*)malloc(sizeof(Matrix));
	createHostMatrixNoAllocation(result, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(uint8_t));
	createHostMatrixNoAllocation(resultErosion, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(uint8_t));
	resultErosion = erosion(A);
	result = dilatation(*resultErosion);
	free(resultErosion->elements);
	free(resultErosion);
	return result;
}

Matrix* closing(Matrix A)
{
	Matrix* result = (Matrix*)malloc(sizeof(Matrix));
	Matrix* resultDilatation = (Matrix*)malloc(sizeof(Matrix));
	createHostMatrixNoAllocation(result, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(uint8_t));
	createHostMatrixNoAllocation(resultDilatation, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(uint8_t));
	resultDilatation = dilatation(A);
	result = erosion(*resultDilatation);
	free(resultDilatation->elements);
	free(resultDilatation);
	return result;
}
/*
__global__ int checkIfEqual_cuda(Matrix A, Matrix B)
{
	int column = threadIdx.x + strucElDim / 2;
	int row = threadIdx.y + strucElDim / 2;

	__shared__ uint8_t dilTile[(blockD + strucElDim - 1)*(blockD + strucElDim - 1)];
	if (A.elements[threadIdx.x + A.numColumns*threadIdx.y + blockIdx.x*blockD + A.numColumns*blockD*blockIdx.y] != B.elements[threadIdx.x + A.numColumns*threadIdx.y + blockIdx.x*blockD + A.numColumns*blockD*blockIdx.y])
		dilTile[threadIdx.x + blockDim.x*threadIdx.y] = 1;
	else
		dilTile[threadIdx.x + blockDim.x*threadIdx.y] = 0;
	__syncthreads();
	int tablica[1024];

	if (column < blockDim.x - strucElDim / 2 && row < blockDim.y - strucElDim / 2)
	{
		index = row * blockDim.x + column;
		dilTile[index]=
	}
	__syncthreads();
	return 1;
}
*/


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

Matrix* reconstruction_cuda(Matrix mask, Matrix marker)
{
	Matrix* result = (Matrix*)malloc(sizeof(Matrix));
	createHostMatrix(result, mask.numRows, mask.numColumns, mask.numColumns*mask.numRows * sizeof(uint8_t));

	Matrix d_mask;
	Matrix d_marker1;
	Matrix d_marker2;
	Matrix d_resultDil;
	createDeviceMatrix(&d_mask, mask.numRows, mask.numColumns, mask.numColumns*mask.numRows * sizeof(uint8_t));
	createDeviceMatrix(&d_marker1, mask.numRows, mask.numColumns, mask.numColumns*mask.numRows * sizeof(uint8_t));
	createDeviceMatrix(&d_marker2, mask.numRows, mask.numColumns, mask.numColumns*mask.numRows * sizeof(uint8_t));
	createDeviceMatrix(&d_resultDil, mask.numRows, mask.numColumns, mask.numColumns*mask.numRows * sizeof(uint8_t));
	checkCudaErrors(cudaMemcpy(d_mask.elements, mask.elements, mask.numColumns*mask.numRows * sizeof(uint8_t), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_marker1.elements, marker.elements, mask.numColumns*mask.numRows * sizeof(uint8_t), cudaMemcpyHostToDevice));
	dim3 threadsDil(blockD + strucElDim - 1, blockD + strucElDim - 1);
	dim3 gridDil(mask.numColumns / blockD, mask.numRows / blockD);
	dim3 threadsComp(blockD + strucElDim - 1, blockD + strucElDim - 1);
	dim3 gridComp((mask.numColumns + threadsComp.x - 1) / threadsComp.x, (mask.numRows + threadsComp.y - 1) / threadsComp.y);
	
	int isEqual=1;
	for (int i = 0; i < 10; i++)
	//while(!checkIfEqual(d_marker1, d_marker2))
	{
		dilatation_cuda <<< gridDil, threadsDil >>> (d_marker1, d_resultDil);
		complement_cuda <<< gridComp, threadsComp >>> (d_resultDil, d_mask, d_marker2);
		dilatation_cuda <<< gridDil, threadsDil >>> (d_marker2, d_resultDil);
		complement_cuda <<< gridComp, threadsComp >>> (d_resultDil, d_mask, d_marker1);
	}
	
	checkCudaErrors(cudaMemcpy(result->elements, d_marker1.elements, d_marker1.numColumns*d_marker1.numRows * sizeof(uint8_t), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_mask.elements));
	checkCudaErrors(cudaFree(d_marker1.elements));
	checkCudaErrors(cudaFree(d_marker2.elements));
	checkCudaErrors(cudaFree(d_resultDil.elements));
	return result;
}

Matrix* reconstruction(Matrix mask, Matrix marker)
{

	Matrix* result = (Matrix*)malloc(sizeof(Matrix));
	Matrix* resultDil = (Matrix*)malloc(sizeof(Matrix));
	Matrix* marker1 = (Matrix*)malloc(sizeof(Matrix));
	Matrix* marker2 = (Matrix*)malloc(sizeof(Matrix));
	createHostMatrixNoAllocation(result, mask.numRows, mask.numColumns, mask.numColumns*mask.numRows * sizeof(uint8_t));
	createHostMatrixNoAllocation(resultDil, mask.numRows, mask.numColumns, mask.numColumns*mask.numRows * sizeof(uint8_t));
	createHostMatrixNoAllocation(marker1, mask.numRows, mask.numColumns, mask.numColumns*mask.numRows * sizeof(uint8_t));
	createHostMatrixNoAllocation(marker2, mask.numRows, mask.numColumns, mask.numColumns*mask.numRows * sizeof(uint8_t));
	marker1 = &marker;
	resultDil = dilatation(*marker1);
	marker2 = complement(*resultDil, mask);


	marker1 = marker2;
	free(resultDil->elements);
	free(resultDil);
	resultDil = dilatation(*marker1);
	marker2 = complement(*resultDil, mask);


	while (!checkIfEqual(*marker1, *marker2))
	{
		free(marker1->elements);
		free(marker1);
		marker1 = marker2;
		free(resultDil->elements);
		free(resultDil);
		resultDil = dilatation(*marker1);
		marker2 = complement(*resultDil, mask);
	}
	free(marker1->elements);
	free(marker1);

	return marker2;
}


Matrix* openingByReconstruction(Matrix A)
{

	Matrix* resultEr = (Matrix*)malloc(sizeof(Matrix));
	Matrix* result = (Matrix*)malloc(sizeof(Matrix));
	createHostMatrixNoAllocation(resultEr, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(uint8_t));
	createHostMatrixNoAllocation(result, A.numRows, A.numColumns, A.numColumns*A.numRows * sizeof(uint8_t));
	resultEr = erosion(A);
	result = reconstruction_cuda(A, *resultEr);
	free(resultEr->elements);
	free(resultEr);
	return result;
}