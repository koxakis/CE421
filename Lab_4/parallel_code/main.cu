#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

#define cudaCheckError() {                                                                       \
        cudaError_t e=cudaGetLastError();                                                        \
        if(e!=cudaSuccess) {                                                                     \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    }



__global__ void histogramGPU ( int * d_hist_out, unsigned char * d_img_in, int img_size, int nbr_bin) {
	/* code */
}

__global__ void histogram_equalizationGPU ( unsigned char * d_img_out, unsigned char * d_img_in,
											int * d_hist_in, int img_size, int nbr_bin) {
	/* code */
}

int main(int argc, char *argv[]){
	// Host Variables
    PGM_IMG h_img_in;
	unsigned int timer = 0;
    PGM_IMG h_img_out_buf;

    int hist[256];

	// Device Variables
	int *d_hist_in,
		*d_img_in;

	unsigned char *d_output;


	if (argc != 3) {
		printf("Run with input file name and output file name as arguments\n");
		exit(1);
	}
    h_img_in = read_pgm(argv[1]);

	h_img_out_buf.w = h_img_in.w;
	h_img_out_buf.h = h_img_in.h;

	printf("Allocating host memory...\n");
	//Host memory allocation
	h_img_out_buf.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
	if ( h_img_out_buf == NULL){
		fprintf(stderr, "Failed to allocate Host memory!\n");
        exit(EXIT_FAILURE);
	}

	//Device memory allocation
	printf("Allocating Device arrays...\n");
	d_hist_out = NULL;
	cudaMalloc((void **)&d_hist_out, 256 * sizeof(int));
	cudaCheckError();

	d_img_in = NULL;
	cudaMalloc((void **)&d_img_in, h_img_in.h * h_img_in.w * sizeof(int));
	cudaCheckError();

	d_output = NULL;
	cudaMalloc((void **)&d_output, result.w * result.h * sizeof(unsigned char));
	cudaCheckError();

	// Main Function call
	// Data transfer to Device
	cudaMemcpy(d_img_in, h_img_in, h_img_in.h * h_img_in.w * sizeof(int), cudaMemcpyHostToDevice);
	cudaCheckError();

	printf("Starting GPU processing...\n");
	// Redirect to contrast-enhancement.c
	//Kernel invocations
	histogramGPU<<<grid, threads>>>(d_hist_out, d_img_in, h_img_in.h * h_img_in.w, 256);
	cudaCheckError();

	cudaDeviceSynchronize();
	cudaCheckError();

	histogram_equalizationGPU<<<grid, threads>>>(d_output, d_img_in, d_hist_out, h_img_in.h * h_img_in.w, 256);
	cudaCheckError();

	cudaDeviceSynchronize();
	cudaCheckError();

	//Return Stuff to h_img_out_buf
	cudaMemcpy(h_img_out_buf, d_output, h_img_in.h * h_img_in.w * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaCheckError();
	// I/O Stuff
    write_pgm(h_img_out_buf, argv[2]);
	free_pgm(h_img_in);
    free_pgm(h_img_out_buf);

	return 0;
}

// I/O Stuff implementation
PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];


    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }

    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);


    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));


    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);
    fclose(in_file);

    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    free(img.img);
}
