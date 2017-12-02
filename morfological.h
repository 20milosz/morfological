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
#define blockD 15

__constant__ static int structuringElements[strucElDim*strucElDim];


void createStructuringElement(Matrix structuringElement);
void copy(Matrix structuringElement);
Matrix* dilatation(Matrix A, Matrix structuringElement);
Matrix* erosion(Matrix A, Matrix structuringElement);
Matrix* complement
(Matrix A, Matrix B);
Matrix* negation(Matrix A);
Matrix* opening(Matrix A, Matrix structuringElement);
Matrix* closing(Matrix A, Matrix structuringElement);
int checkIfEqual(Matrix A, Matrix B);
Matrix* reconstruction(Matrix mask, Matrix marker, Matrix structuringElement);
Matrix* openingByReconstruction(Matrix A, Matrix structuringElement);


#endif