#include "bitmap.h"
#include "morfological.h"
#include "helper_timer.h"
#include "helper_cuda.h"

//__constant__ static int structuringElement[strucElDim*strucElDim];

int main()
{
	checkCudaErrors(cudaSetDevice(0));
	StopWatchInterface *timer = NULL;

	bitmap* bmp;
	Matrix bImage;
	bitmap res_bmp;
	int8_t status;
	int i = 0;
	Matrix* resultNegation;
	Matrix* result;
	Matrix* resultMorfOp;
	Matrix h_structuringElement;

	// generacja elementu strukturalnego
	createHostMatrix(&h_structuringElement, strucElDim, strucElDim, strucElDim *strucElDim * sizeof(uint8_t));
	createStructuringElement(h_structuringElement);
	showMatrix(h_structuringElement, "structuring element");
	copy(h_structuringElement);

	// wczytanie obrazu
	bmp = readBitmap("fingerprint_noise_duzy.bmp");
	status = convertBitmapToBinaryImage(bmp, &bImage);
	if (status == -1) {
		printf("Error while converting bitmap to binary image.\n");
		WINPAUSE;
		exit(0);
	}

	// wstepnie przygotowanie obrazu
	// wykonane operacje morfologiczne w danym przypdaku maja sens je¿eli pracuje siê na obrazie zanegowanym
	resultNegation = negation(bImage);


	resultMorfOp = dilatation(*resultNegation);
	result = negation(*resultMorfOp);
	convertBinaryImageTOBitmapUsingHeader(result, bmp->hp, &res_bmp);
	writeBitmap(&res_bmp, "dylatacja.bmp");
	free(res_bmp.image);
	free(resultMorfOp->elements);
	free(resultMorfOp);
	free(result->elements);
	free(result);


	resultMorfOp = erosion(*resultNegation);
	result = negation(*resultMorfOp);
	convertBinaryImageTOBitmapUsingHeader(result, bmp->hp, &res_bmp);
	writeBitmap(&res_bmp, "erozja.bmp");
	free(res_bmp.image);
	free(resultMorfOp->elements);
	free(resultMorfOp);
	free(result->elements);
	free(result);

	float czas = 1000000000;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);
	for (int i = 0; i < 100; i++)
	{
		sdkStartTimer(&timer);
		resultMorfOp = openingByReconstruction(*resultNegation);
		sdkStopTimer(&timer);
		free(resultMorfOp->elements);
		free(resultMorfOp);
		if (sdkGetTimerValue(&timer) < czas)
			czas = sdkGetTimerValue(&timer);
	}
	printf("Processing time: %f ms\n", czas);
	sdkDeleteTimer(&timer);
	result = negation(*resultMorfOp);
	convertBinaryImageTOBitmapUsingHeader(result, bmp->hp, &res_bmp);
	writeBitmap(&res_bmp, "otwarciePrzezRekonstrukcje.bmp");
	free(res_bmp.image);
	free(result->elements);
	free(result);


	resultMorfOp = opening(*resultNegation);
	result = negation(*resultMorfOp);
	convertBinaryImageTOBitmapUsingHeader(result, bmp->hp, &res_bmp);
	writeBitmap(&res_bmp, "otwarcie.bmp");
	free(res_bmp.image);
	free(resultMorfOp->elements);
	free(resultMorfOp);
	free(result->elements);
	free(result);


	free(resultNegation->elements);
	free(resultNegation);
	freeBitmap(bmp);
	free(h_structuringElement.elements);

	WINPAUSE;
	return 0;
}
