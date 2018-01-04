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
#define blockD 30

__constant__ static uint8_t structuringElements[strucElDim*strucElDim];


void createStructuringElement(Matrix structuringElement);
void copy(Matrix structuringElement);
Matrix* dilatation(Matrix A);
Matrix* erosion(Matrix A);
Matrix* complement(Matrix A, Matrix B);
Matrix* negation(Matrix A);
Matrix* opening(Matrix A);
Matrix* closing(Matrix A);
int checkIfEqual(Matrix A, Matrix B);
Matrix* reconstruction(Matrix mask, Matrix marker);
Matrix* openingByReconstruction(Matrix A);


#endif