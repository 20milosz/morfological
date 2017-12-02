#ifndef __X_BITMAP__
#define __X_BITMAP__

#include "matrix.h"
#include <inttypes.h>
#include "morfological.h"

#ifdef _WIN32
#define WINPAUSE system("pause")
#endif

#pragma pack(push,1)
/* Windows 3.x bitmap file header */
typedef struct {
	char         filetype[2];   /* magic - always 'B' 'M' */
	unsigned int filesize;
	short        reserved1;
	short        reserved2;
	unsigned int dataoffset;    /* offset in bytes to actual bitmap data */
} file_header;

/* Windows 3.x bitmap full header, including file header */
typedef struct {
	file_header  fileheader;
	unsigned int headersize;
	int          width;
	int          height;
	short        planes;
	short        bitsperpixel;  /* we only support the value 24 here */
	unsigned int compression;   /* we do not support compression */
	unsigned int bitmapsize;
	int          horizontalres;
	int          verticalres;
	unsigned int numcolors;
	unsigned int importantcolors;
} bitmap_header;

typedef struct {
	uint8_t r;
	uint8_t g;
	uint8_t b;
} bitmap_rgb;

typedef struct {
	bitmap_header* hp;
	bitmap_rgb* image;
} bitmap;
#pragma pack(pop)



bitmap* readBitmap(char* file);
int8_t writeBitmap(bitmap* bmp, char* file);
void freeBitmap(bitmap *bmp);
int8_t convertBitmapToBinaryImage(bitmap* bitmap_in, Matrix* mat_out);
int8_t convertBinaryImageTOBitmapUsingHeader(Matrix* mat_in, bitmap_header* bh_in, bitmap* bitmap_out);



#endif