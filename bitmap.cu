#include "bitmap.h"



bitmap* readBitmap(char* file) {
	bitmap* bmp;
	int32_t i, j;
	FILE *streamIn;
	fopen_s(&streamIn, file, "rb");
	if (streamIn == (FILE *)0) {
		printf("File opening error ocurred. Exiting program.\n");
		WINPAUSE;
		exit(0);
	}
	bmp = (bitmap*)malloc(sizeof(bitmap));
	if (bmp == NULL) {
		exit(0);
	}
	bmp->hp = (bitmap_header*)malloc(sizeof(bitmap_header));
	fread(bmp->hp, sizeof(bitmap_header), 1, streamIn);
	if (bmp->hp->bitsperpixel != 24) {
		printf("24 bits bitmaps support only.\n");
		WINPAUSE;
		exit(0);
	}
	if (bmp->hp->compression != 0) {
		printf("No compression support only.\n");
		WINPAUSE;
		exit(0);
	}
	fseek(streamIn, sizeof(char)*bmp->hp->fileheader.dataoffset, SEEK_SET);
	int padding = (4 - ((bmp->hp->width * 3) % 4)) % 4;

	bmp->image = (bitmap_rgb*)malloc(sizeof(bitmap_rgb)*(bmp->hp->width)*(bmp->hp->height));
	if (bmp->image == NULL) {
		exit(0);
	}
	for (j = bmp->hp->height - 1; j >= 0; j--) {    // foreach pixel row
		for (i = 0; i <bmp->hp->width; i++) {    // foreach pixel col
			bmp->image[j*bmp->hp->width + i].b = getc(streamIn);  // use BMP 24bit with no alpha channel
			bmp->image[j*bmp->hp->width + i].g = getc(streamIn);  // BMP uses BGR but we want RGB, grab byte-by-byte
			bmp->image[j*bmp->hp->width + i].r = getc(streamIn);  // reverse-order array indexing fixes RGB issue...
																  //printf("pixel %d : [%d,%d,%d]\n", i + 1, bmp->image[j*bmp->hp->width + i].r, bmp->image[j*bmp->hp->width + i].g, bmp->image[j*bmp->hp->width + i].b);
		}
		for (i = 0; i<padding; i++) getc(streamIn);
	}

	fclose(streamIn);
	return bmp;
}


int8_t writeBitmap(bitmap* bmp, char* file) {
	FILE *out;
	int32_t i, j;
	uint8_t n, padding_data = 0;
	fopen_s(&out, file, "wb");
	if (out == NULL) {
		return -1;
	}
	n = fwrite(bmp->hp, sizeof(char), sizeof(bitmap_header), out);
	if (n<1) {
		return -1;
	}
	fseek(out, sizeof(char)*(bmp->hp->fileheader.dataoffset), SEEK_SET);
	int padding = (4 - ((bmp->hp->width * 3) % 4)) % 4;
	for (j = bmp->hp->height - 1; j >= 0; j--) {    // foreach pixel row
		for (i = 0; i <bmp->hp->width; i++) {    // foreach pixel col

			n = fwrite(&bmp->image[j*bmp->hp->width + i].b, sizeof(uint8_t), 1, out);
			if (n<1) { return -1; }
			n = fwrite(&bmp->image[j*bmp->hp->width + i].g, sizeof(uint8_t), 1, out);
			if (n<1) { return -1; }
			n = fwrite(&bmp->image[j*bmp->hp->width + i].r, sizeof(uint8_t), 1, out);
			if (n<1) { return -1; }
		}
		for (i = 0; i < padding; i++) {
			n = fwrite(&padding_data, sizeof(uint8_t), 1, out);
			if (n<1) { return -1; }
		}
	}
	fclose(out);

}

void freeBitmap(bitmap *bmp) {
	free(bmp->hp);
	free(bmp->image);
}
/*
int8_t convertBitmapToBinaryImage(bitmap* bitmap_in, Matrix* mat_out) {
	//int16_t status = initializeMatrix(mat_out, bitmap_in->hp->height, bitmap_in->hp->width);
	createHostMatrix(mat_out, bitmap_in->hp->height, bitmap_in->hp->width, bitmap_in->hp->height*bitmap_in->hp->width * sizeof(int));
	verifyHostAllocation(*mat_out);

	int i = 0;
	for (i = 0; i < (bitmap_in->hp->width)*(bitmap_in->hp->height); i++) {
		//		if ((bitmap_in->image[i].r+ bitmap_in->image[i].b+ bitmap_in->image[i].g)/3 < 170) {
		//	if ((bitmap_in->image[i].r/3 + bitmap_in->image[i].b/3 + bitmap_in->image[i].g/3) < 128) {
		if (bitmap_in->image[i].r < 170) {
			mat_out->elements[i] = 1;
		}
		else {
			mat_out->elements[i] = 0;
		}
	}
	return 0;
}*/

int8_t convertBitmapToBinaryImage(bitmap* bitmap_in, Matrix* mat_out) {
	//int16_t status = initializeMatrix(mat_out, bitmap_in->hp->height, bitmap_in->hp->width);
	int height = bitmap_in->hp->height;
	int width = bitmap_in->hp->width;
	height = (height + blockD - 1) / blockD;
	width = (width + blockD - 1) / blockD;
	height = height*blockD;
	width = width*blockD;
	height = height + strucElDim - 1;
	width = width + strucElDim - 1;
	createHostMatrix(mat_out, height, width, height*width * sizeof(int));
	verifyHostAllocation(*mat_out);

	for (int i = 0; i < width*height; i++)
	{
		mat_out->elements[i] = 0;
	}

	int i = 0;
	int j = 0;
	int idxbitmap = 0;
	int idxmatrix = 0;
		for (i = 0; i < bitmap_in->hp->width; i++)
		{
			for (j = 0; j < bitmap_in->hp->height; j++)
			{
				idxbitmap = i + bitmap_in->hp->width*j;
				idxmatrix = i+1 + width*(j+1);
				if (bitmap_in->image[idxbitmap].r < 170) {
				mat_out->elements[idxmatrix] = 1;
			}
			else 
			{
				mat_out->elements[idxmatrix] = 0;
			}

		}

	}
		bitmap_in->hp->width = width;
		bitmap_in->hp->height = height;
		bitmap_in->hp->bitmapsize=width*height;
		printf("wysokosc%d\n", height);

	return 0;
}


int8_t convertBinaryImageTOBitmapUsingHeader(Matrix* mat_in, bitmap_header* bh_in, bitmap* bitmap_out) {
	bitmap_out->hp = bh_in;
	bitmap_out->image = (bitmap_rgb*)malloc(sizeof(bitmap_rgb)*(mat_in->numColumns)*(mat_in->numRows));
	if (bitmap_out->image == NULL) {
		return -1;
	}
	int i = 0;
	for (i = 0; i < (mat_in->numColumns)*(mat_in->numRows); i++) {
		if (mat_in->elements[i] == 1) {
			bitmap_out->image[i].b = bitmap_out->image[i].g = bitmap_out->image[i].r = 0;
		}
		else {
			bitmap_out->image[i].b = bitmap_out->image[i].g = bitmap_out->image[i].r = 255;
		}
	}
	return 0;
}
