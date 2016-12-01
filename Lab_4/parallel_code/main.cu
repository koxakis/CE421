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

// Global kernel invocation stuff
int threadsPerBlock ;

int blocksPerGridx ;
int blocksPerGridy ;

__global__ void histogram_equalizationGPU ( unsigned char * d_img_out, unsigned char * d_img_in,
											int * d_hist_in, int img_size, int nbr_bin, int * d_lut, int threads_number) {

	int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;

	int satrt, end;

	if (thread_pos >= img_size ){
		return;
	}

	satrt = (( img_size/threads_number) * thread_pos);
	if (threads_number == 1) {
		end = (img_size/threads_number);
	}else{
		end = ((img_size/threads_number) * (thread_pos + 1));
	}
	for (int i = 0; i < end	; i++) {
		if (d_lut[d_img_in[i]] > 255) {
			d_img_out[i] = 255;
		}else{
			d_img_out[i] = (unsigned char) d_lut[d_img_in[i]];
		}
	}


}



int main(int argc, char *argv[]){
	// Host Variables
    PGM_IMG h_img_in;
	unsigned int timer = 0;
    PGM_IMG h_img_out_buf;
	int cdf = 0, min = 0, d, i = 0;

	int * h_hist_buffer, *h_lut;

	// Device Variables
	int *d_lut, *d_hist_in;
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

	printf("Allocating host memory...\n");
	//Host memory allocation
	h_img_out_buf.img = (unsigned char *)malloc(h_img_out_buf.w * h_img_out_buf.h * sizeof(unsigned char));
	h_hist_buffer = (int*)malloc(256 * sizeof(int));

	//Device memory allocation
	printf("Allocating Device arrays...\n");

	cudaMalloc(&d_img_in, h_img_in.w * h_img_in.h * sizeof(unsigned char) );
	cudaCheckError();

	cudaMalloc(&d_img_out, h_img_in.w * h_img_in.h * sizeof(unsigned char) );
	cudaCheckError();

	cudaMalloc(&d_hist_in, 256 * sizeof(int));
	cudaCheckError();

	cudaMalloc(&d_lut, 256 * sizeof(int));
	cudaCheckError();

	// Main Function call
	// Data transfer to Device

	//cudaMemcpy(&d_img_out, h_img_out_buf.img, h_img_in.w * h_img_in.h * sizeof(unsigned char), cudaMemcpyHostToDevice );
	//cudaCheckError();

	cudaMemcpy(d_img_in, h_img_in.img, h_img_in.w * h_img_in.h * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaCheckError();

	printf("Starting CPU processing...\n");

	for (int i = 0; i < 256; i++) {
		h_hist_buffer[i] = 0;
	}

	for (int i = 0; i < h_img_in.w * h_img_in.h; i++) {
		h_hist_buffer[h_img_in.img[i]] ++;
	}

	while (min == 0) {
		min = h_hist_buffer[i++];
	}
	d = (h_img_in.w * h_img_in.h) - min;
	for (int i = 0; i < 256; i++) {
		cdf += h_hist_buffer[i];
		h_lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
		if (h_lut[i] < 0) {
			h_lut[i];
		}

	}
	// Data transfer to Device
	cudaMemcpy(d_hist_in, h_hist_buffer, 256 * sizeof(int), cudaMemcpyHostToDevice);
	cudaCheckError();

	cudaMemcpy(d_lut, h_lut, 256 * sizeof(int), cudaMemcpyHostToDevice);
	cudaCheckError();

	printf("Starting GPU processing...\n");
	//Kernel invocations

	threadsPerBlock = 32;
	dim3 threads(threadsPerBlock, threadsPerBlock);

	blocksPerGridx =  h_img_in.h/threads.x;
	blocksPerGridy =  h_img_in.w/threads.y;

	dim3 grid(blocksPerGridy, blocksPerGridx);

	printf("CUDA kernel launch with %dx%d blocks of %dx%d threads\n", blocksPerGridy, blocksPerGridx, threadsPerBlock, threadsPerBlock);
	histogram_equalizationGPU<<<grid, threads>>>(d_img_out, d_img_in, d_hist_in,
		 											h_img_in.h * h_img_in.w, 256, d_lut,
													(threads.x * threads.y)* (blocksPerGridx * blocksPerGridy));
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

	cudaFree(d_hist_in);
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