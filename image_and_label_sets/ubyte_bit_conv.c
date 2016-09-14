#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define FILEPATH_LEARNING_SET "train-images-idx3-ubyte"
#define FILEPATH_TEST_SET "t10k-images-idx3-ubyte"
#define FILEPATH_BITFILE "bit_mnist_images"

#define IMG_X 28
#define IMG_Y 28

#define THRESHOLD 40

typedef unsigned char byte_t;

void printImg(int* img) {
	for (int x = 0; x < IMG_X; x++) {
		for (int y = 0; y < IMG_Y; y++) {
			fprintf(stdout, "%d ", img[x * IMG_X + y]);
		}
		fprintf(stdout, "\n");
	}
}

void convert_and_write(FILE* fin, FILE* fout) {
	unsigned int pixel;
	byte_t byte = 0;
	while (!feof(fin)) {
		byte = 0;
		for (int i = 0; i < 8; i++) {
			pixel = getc(fin);
			if (pixel >= THRESHOLD) {
				pixel = 1;
			} else {
				pixel = 0;
			}

			byte |= (pixel << (7-i));
		}
		putc(byte, fout);
	}
}

void read_next_bitfile_image(FILE* bitfile, int* img) {
	/* Bytes per image is 784 / 8 = 98 */
	byte_t temp = 0;
	for (int i = 0; i < 98; i++) {
		temp = getc(bitfile);
		for (int offset = 0; offset < 8; offset++) {
			img[i * 8 + offset] = (temp & (1 << (7 - offset))) >> (7 - offset);
		}
	}
}

int do_conversion(FILE* bitfile) {
		FILE* li = fopen(FILEPATH_LEARNING_SET, "rb");
	
		if (!li) {
			return -1;
		}
		/* Skip header bytes */
		fseek(li, 32*4, SEEK_SET);

		convert_and_write(li, bitfile);

		fclose(li);

		FILE* ti = fopen(FILEPATH_TEST_SET, "rb");
		if (!ti) {
			return -1;
		}
		/* Skip header bytes */
		fseek(ti, 32*4, SEEK_SET);

		convert_and_write(ti, bitfile);

		fclose(ti);
		return 0;
}

int main(int argc, char** argv) {

	int test = 0;

	for (int i = 0; i < argc; i++) {
		if (strcmp(argv[i], "-test") == 0) {
			test = 1;
		}
	}

	FILE* bitfile;

	if (test == 0) {
		bitfile = fopen(FILEPATH_BITFILE, "wb");
		if (!bitfile) {
			fprintf(stderr, "%s\n", "Could not create bitfile. Exiting.");
			return -1;
		}
		int err = do_conversion(bitfile);
		if (err < 0) {
			fclose(bitfile);
			fprintf(stderr, "%s\n", "An error occurred during file I/O. Exiting.");
			return -1;
		}
	}
	
	if (test == 1) {
		bitfile = fopen(FILEPATH_BITFILE, "rb");
		if (!bitfile) {
			fprintf(stderr, "%s\n", "Could not open bitfile. Exiting.");
			return -1;
		}

		int img[IMG_X*IMG_Y];

		for (int i = 0; i < 5; i++) {
			read_next_bitfile_image(bitfile, img);
			printImg(img);
		}
	}

	fclose(bitfile);

	return 0;
}