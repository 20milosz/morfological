#include "bitmap.h"
#include "morfological.h"
#include "helper_timer.h"

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

	Matrix* result0;
	Matrix* result;
	Matrix* result2;
	Matrix* result3;
	Matrix h_structuringElement;

	createHostMatrix(&h_structuringElement, strucElDim, strucElDim, strucElDim *strucElDim * sizeof(uint8_t));
	verifyHostAllocation(h_structuringElement);
	createStructuringElement(h_structuringElement);
	showMatrix(h_structuringElement, "structuring element");
	copy(h_structuringElement);


	bmp = readBitmap("fingerprint_noise_duzy.bmp");
//	bmp = readBitmap("binaryzacja_gen_duzy.bmp");
	status = convertBitmapToBinaryImage(bmp, &bImage);
	if (status == -1) {
		printf("Error while converting bitmap to binary image.\n");
		WINPAUSE;
		exit(0);
	}

	result0 = negation(bImage);
	result2 = dilatation(*result0, h_structuringElement);
	result3 = negation(*result2);
	convertBinaryImageTOBitmapUsingHeader(result2, bmp->hp, &res_bmp);
	//convertBinaryImageTOBitmapUsingHeader(&bImage, bmp->hp, &res_bmp);
	writeBitmap(&res_bmp, "dylatacja.bmp");

	float czas = 1000000000;

	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	for (int i = 0; i < 1; i++)
	{
		sdkStartTimer(&timer);
		result = openingByReconstruction(*result0, h_structuringElement);
		sdkStopTimer(&timer);
		if (sdkGetTimerValue(&timer) < czas)
			czas = sdkGetTimerValue(&timer);
	}


	printf("Processing time: %f ms\n", czas);
	sdkDeleteTimer(&timer);

	//result2 = closing(*result, h_structuringElement);
	result3 = negation(*result);
	convertBinaryImageTOBitmapUsingHeader(result3, bmp->hp, &res_bmp);
	//convertBinaryImageTOBitmapUsingHeader(&bImage, bmp->hp, &res_bmp);
	writeBitmap(&res_bmp, "opening_gen_duzy.bmp");

	result = openingByReconstruction(*result0, h_structuringElement);
	//result2 = closing(*result, h_structuringElement);
	result3 = negation(*result);
	convertBinaryImageTOBitmapUsingHeader(result3, bmp->hp, &res_bmp);
	//convertBinaryImageTOBitmapUsingHeader(&bImage, bmp->hp, &res_bmp);
	writeBitmap(&res_bmp, "openingByReconstruction_gen_duzy.bmp");


	freeBitmap(bmp);

	free(result0->elements);
	free(result->elements);
	//free(result2->elements);
	free(result3->elements);
	free(h_structuringElement.elements);
	free(result0);
	free(result);
	//free(result2);
	free(result3);


	WINPAUSE;
	return 0;
}
