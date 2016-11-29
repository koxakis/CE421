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

__global__ void histogramGPU ( int * d_hist_out, int * d_hist_in, unsigned char * d_img_in, int img_size, int nbr_bin) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int i = col + (threadIdx.y * img_size) + (blockIdx.y * blockDim.y) * img_size;

	d_hist_out[i] = 0 ;
    //for ( i = 0; i < img_size; i ++){
    d_hist_out[d_img_in[i]] ++;
    //}
}

__global__ void histogram_equalizationGPU ( unsigned char * d_img_out, unsigned char * d_img_in,
											int * d_hist_in, int img_size, int nbr_bin, int * d_lut) {

	//int *lut = (int *)malloc(sizeof(int)*nbr_bin);
	//__shared__ int lut[nbr_bin];
    int cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int i = col + (threadIdx.y * img_size) + (blockIdx.y * blockDim.y) * img_size;
    while(min == 0){
        min = d_hist_in[i++];
    }
    d = img_size - min;
    //for(int k = 0; k < nbr_bin; k ++){
        cdf += d_hist_in[i];
        d_lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(d_lut[i] < 0){
            d_lut[i] = 0;
        }


    //}

    /* Get the result image */
    //for(int k = 0; k < img_size; k ++){
        if(d_lut[d_img_in[i]] > 255){
            d_img_out[i] = 255;
        }
        else{
            d_img_out[i] = (unsigned char)d_lut[d_img_in[i]];
        }

    //}
}



int main(int argc, char *argv[]){
	// Host Variables
    PGM_IMG h_img_in;
	unsigned int timer = 0;
    PGM_IMG h_img_out_buf;

    int hist[256];

	// Device Variables
	int *d_hist_out, *d_lut, *d_hist_in;
	unsigned char
		*d_img_in,
		*d_img_out;


	if (argc != 3) {
		printf("Run with input file name and output file name as arguments\n");
		exit(1);
	}
    h_img_in = read_pgm(argv[1]);

	h_img_out_buf.w = h_img_in.w;
	h_img_out_buf.h = h_img_in.h;

	for (int i = 0; i < 256; i ++){
        hist[i] = 0;
    }

	printf("Allocating host memory...\n");
	//Host memory allocation
	h_img_out_buf.img = (unsigned char *)malloc(h_img_out_buf.w * h_img_out_buf.h * sizeof(unsigned char));

	//Device memory allocation
	printf("Allocating Device arrays...\n");
	d_hist_in = NULL;
	cudaMalloc((void **)&d_hist_in, 256 * sizeof(int));
	cudaCheckError();

	d_lut = NULL;
	cudaMalloc((void **)&d_lut, 256 * sizeof(int));
	cudaCheckError();

	d_hist_out = NULL;
	cudaMalloc((void **)&d_hist_out, 256 * sizeof(int));
	cudaCheckError();

	d_img_in = NULL;
	cudaMalloc((void **)&d_img_in, h_img_in.h * h_img_in.w * sizeof(unsigned char));
	cudaCheckError();

	d_img_out = NULL;
	cudaMalloc((void **)&d_img_out, h_img_out_buf.w * h_img_out_buf.h * sizeof(unsigned char));
	cudaCheckError();

	// Main Function call
	// Data transfer to Device
	cudaMemcpy(d_img_in, h_img_in.img, h_img_in.h * h_img_in.w * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaCheckError();

	cudaMemcpy(d_hist_in, hist, 256 * sizeof(int), cudaMemcpyHostToDevice);

	printf("Starting GPU processing...\n");
	//Kernel invocations
	int threadsPerBlock = 32;
	dim3 threads(threadsPerBlock, threadsPerBlock);

	int blocksPerGridx =  h_img_in.h/threads.x;
	int blocksPerGridy =  h_img_in.w/threads.y;
	dim3 grid(blocksPerGridy, blocksPerGridx);

	printf("CUDA kernel launch with %dx%d blocks of %dx%d threads\n", blocksPerGridy, blocksPerGridx, threadsPerBlock, threadsPerBlock);
	histogramGPU<<<1, 16>>>(d_hist_out, d_img_in, h_img_in.h * h_img_in.w, 256);
	cudaCheckError();

	cudaDeviceSynchronize();
	cudaCheckError();

	printf("CUDA kernel launch with %dx%d blocks of %dx%d threads\n", blocksPerGridy, blocksPerGridx, threadsPerBlock, threadsPerBlock);
	histogram_equalizationGPU<<<grid, threads>>>(d_img_out, d_img_in, d_hist_out, h_img_in.h * h_img_in.w, 256, d_lut);
	cudaCheckError();

	cudaDeviceSynchronize();
	cudaCheckError();

	//Return Stuff to h_img_out_buf
	cudaMemcpy(h_img_out_buf.img, d_img_out, h_img_in.h * h_img_in.w * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaCheckError();

	cudaDeviceReset();
	// I/O Stuff
    write_pgm(h_img_out_buf, argv[2]);
	free_pgm(h_img_in);
    free_pgm(h_img_out_buf);

	cudaFree(d_hist_out);
	cudaFree(d_img_in);
	cudaFree(d_img_out);
	cudaCheckError();

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
