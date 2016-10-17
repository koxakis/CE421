// This will apply the sobel filter and return the PSNR between the golden sobel and the produced sobel
// sobelized image
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>

/*
Loop interchange
Loop unrolling
Function Inlining
Loop fusion
Common subexpresion elimination
*/

//Common subexpresion elimination 5th stage
#define SIZE	4096
#define SIZE_2	4096*4096
#define INPUT_FILE	"input.grey"
#define OUTPUT_FILE	"output_sobel.grey"
#define GOLDEN_FILE	"golden.grey"

/* The horizontal and vertical operators to be used in the sobel filter */
char horiz_operator[3][3] = {{-1, 0, 1},
                             {-2, 0, 2},
                             {-1, 0, 1}};
char vert_operator[3][3] = {{1, 2, 1},
                            {0, 0, 0},
                            {-1, -2, -1}};

double sobel(unsigned char *input, unsigned char *output, unsigned char *golden);
int convolution2D(int posy, int posx, const unsigned char *input, char operator[][3]);

/* The arrays holding the input image, the output image and the output used *
 * as golden standard. The luminosity (intensity) of each pixel in the      *
 * grayscale image is represented by a value between 0 and 255 (an unsigned *
 * character). The arrays (and the files) contain these values in row-major *
 * order (element after element within each row and row after row. 			*/
unsigned char input[SIZE_2], output[SIZE_2], golden[SIZE_2];

/* The main computational function of the program. The input, output and *
 * golden arguments are pointers to the arrays used to store the input   *
 * image, the output produced by the algorithm and the output used as    *
 * golden standard for the comparisons.									 */
double sobel(unsigned char *input, unsigned char *output, unsigned char *golden)
{
	double PSNR = 0, t;
	int i, j;
	unsigned int p;
	int res;
	struct timespec  tv1, tv2;
	FILE *f_in, *f_out, *f_golden;

	/* The first and last row of the output array, as well as the first  *
     * and last element of each column are not going to be filled by the *
     * algorithm, therefore make sure to initialize them with 0s.		 */
	memset(output, 0, SIZE*sizeof(unsigned char));
	memset(&output[SIZE*(SIZE-1)], 0, SIZE*sizeof(unsigned char));
	for (i = 1; i < SIZE-1; i++) {
		output[i*SIZE] = 0;
		output[i*SIZE + SIZE - 1] = 0;
	}

	/* Open the input, output, golden files, read the input and golden    *
     * and store them to the corresponding arrays.						  */
	f_in = fopen(INPUT_FILE, "r");
	if (f_in == NULL) {
		printf("File " INPUT_FILE " not found\n");
		exit(1);
	}

	f_out = fopen(OUTPUT_FILE, "wb");
	if (f_out == NULL) {
		printf("File " OUTPUT_FILE " could not be created\n");
		fclose(f_in);
		exit(1);
	}

	f_golden = fopen(GOLDEN_FILE, "r");
	if (f_golden == NULL) {
		printf("File " GOLDEN_FILE " not found\n");
		fclose(f_in);
		fclose(f_out);
		exit(1);
	}

	fread(input, sizeof(unsigned char), SIZE_2, f_in);
	fread(golden, sizeof(unsigned char), SIZE_2, f_golden);
	fclose(f_in);
	fclose(f_golden);

	/* This is the main computation. Get the starting time. */
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
	/* For each pixel of the output image */
	//Loop interchange 1st stage

	for (i=1; i<SIZE-1; i+=1 ) {
		for (j=1; j<SIZE-1; j+=1) {
			/* Apply the sobel filter and calculate the magnitude *
			 * of the derivative.								  */
			//function inlining 3rd stage
			//loop fusion 4th stage
			int k, z, horiz_res, vert_res;
 			horiz_res = 0;
 			vert_res = 0;
 			//Horizontal oporations
 			//Vertical oporations
			/* Implement a 2D convolution of the matrix with the operator */
			/* posy and posx correspond to the vertical and horizontal disposition of the *
			 * pixel we process in the original image, input is the input image and       *
			 * operator the operator we apply (horizontal or vertical). The function ret. *
			 * value is the convolution of the operator with the neighboring pixels of the*
			 * pixel we process.														  */
 			for (k = -1; k <= 1; k++) {
 				for (z = -1; z <= 1; z+=3) {
 					horiz_res += input[(i + k)*SIZE + j + z] * horiz_operator[k+1][z+1];
					vert_res += input[(i + k)*SIZE + j + z] * vert_operator[k+1][z+1];

 					horiz_res += input[(i + k)*SIZE + j + (z+1)] * horiz_operator[k+1][z+2];
					vert_res += input[(i + k)*SIZE + j + (z+1)] * vert_operator[k+1][z+2];

 					horiz_res += input[(i + k)*SIZE + j + (z+2)] * horiz_operator[k+1][z+3];
					vert_res += input[(i + k)*SIZE + j + (z+2)] * vert_operator[k+1][z+3];
 				}
 			}
 			p = ( horiz_res*horiz_res) + ( vert_res*vert_res);
 			res = (int)sqrt(p);
			/* If the resulting value is greater than 255, clip it *
			 * to 255.											   */
			if (res > 255)
				output[i*SIZE + j] = 255;
			else
				output[i*SIZE + j] = (unsigned char)res;
		}
	}

	/* Now run through the output and the golden output to calculate *
	 * the MSE and then the PSNR.									 */
	//Loop unrolling 2nd stage
	for (i=1; i<SIZE-1; i++) {
		for ( j=1; j<SIZE-1; j+=3 ) {
			t = ((output[i*SIZE+j] - golden[i*SIZE+j]) * (output[i*SIZE+j] - golden[i*SIZE+j]));
			PSNR += t;

			t = ((output[i*SIZE+(j+1)] - golden[i*SIZE+(j+1)]) * (output[i*SIZE+(j+1)] - golden[i*SIZE+(j+1)]));
			PSNR += t;

			t =((output[i*SIZE+(j+2)] - golden[i*SIZE+(j+2)])*(output[i*SIZE+(j+2)] - golden[i*SIZE+(j+2)]));
			PSNR += t;
		}
	}

	PSNR /= (double)(SIZE_2);
	PSNR = 10*log10(65536/PSNR);

	/* This is the end of the main computation. Take the end time,  *
	 * calculate the duration of the computation and report it. 	*/
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

	printf ("Total time = %10g seconds\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(double) (tv2.tv_sec - tv1.tv_sec));


	/* Write the output file */
	fwrite(output, sizeof(unsigned char), SIZE_2, f_out);
	fclose(f_out);

	return PSNR;
}


int main(int argc, char* argv[])
{
	double PSNR;
	PSNR = sobel(input, output, golden);
	printf("PSNR of original Sobel and computed Sobel image: %g\n", PSNR);

	return 0;
}
