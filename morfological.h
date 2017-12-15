#ifndef __X_MORFOLOGICAL__
#define __X_MORFOLOGICAL__

#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include "helper_cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#define strucElDim 3
#define blockD (32-strucElDim+1)

__constant__ static uint8_t structuringElements[strucElDim*strucElDim];


__global__ void erosion_cuda(Matrix A, Matrix result);
__global__ void dilatation_cuda(Matrix A, Matrix result);
__global__ void dilatation_complement_cuda(Matrix A, Matrix B, Matrix result);
__global__ void checkIfEqual_cuda(Matrix A, Matrix B, unsigned int *maximum, int *mutex);
__global__ void complement_cuda(Matrix A, Matrix B, Matrix result);
void copy(Matrix structuringElement);
void createStructuringElement(Matrix structuringElement);
Matrix* dilatation(Matrix A);
Matrix* erosion(Matrix A);
Matrix* complement(Matrix A, Matrix B);
Matrix* negation(Matrix A);
Matrix* opening(Matrix A);
Matrix* closing(Matrix A);
int checkIfEqual(Matrix A, Matrix B);
Matrix* reconstruction_cuda(Matrix mask, Matrix marker);
Matrix* reconstruction(Matrix mask, Matrix marker);
Matrix* openingByReconstruction(Matrix mask);


#endif