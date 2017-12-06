#include "bitmap.h"
#include "morfological.h"
#include "helper_timer.h"

int main()
{

	checkCudaErrors(cudaSetDevice(0));

	bitmap* bmp;
	Matrix bImage;
	bitmap res_bmp;

	int8_t status;
	int i = 0;

	Matrix* result0;
	Matrix* result2;
	Matrix* result3;
	Matrix h_structuringElement;

	createHostMatrix(&h_structuringElement, strucElDim, strucElDim, strucElDim *strucElDim * sizeof(uint8_t));
	verifyHostAllocation(h_structuringElement);
	createStructuringElement(h_structuringElement);
	showMatrix(h_structuringElement, "structuring element");
	copy(h_structuringElement);


	bmp = readBitmap("fingerprint_noise_duzy.bmp");

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
	writeBitmap(&res_bmp, "dylatacja.bmp");



	freeBitmap(bmp);

	free(result0->elements);
	free(result2->elements);
	free(result3->elements);
	free(h_structuringElement.elements);
	free(result0);
	free(result2);
	free(result3);

	WINPAUSE;
	return 0;
}
