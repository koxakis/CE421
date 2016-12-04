#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include "gputimer.h"
#include <time.h>

#define cudaCheckError() {                                                                       \
        cudaError_t e=cudaGetLastError();                                                        \
        if(e!=cudaSuccess) {                                                                     \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    }
//Posible cdf culc
__global__ void histogram_equalizationGPU ( unsigned char * d_img_out, unsigned char * d_img_in,
											int img_size, int nbr_bin, int * d_lut, int threads_number) {

	int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (d_lut[d_img_in[thread_pos]] > (nbr_bin - 1)) {
		d_img_out[thread_pos] = (nbr_bin - 1);
	}else {
		d_img_out[thread_pos] = (unsigned char)d_lut[d_img_in[thread_pos]];
	}

}

int main(int argc, char *argv[]){
	// Host Variables
	clock_t start, end, start2, end2;

	start = clock();
    PGM_IMG h_img_in, h_img_out_buf;
	int cdf = 0, min = 0, d, i = 0;

	int * h_hist_buffer, * h_lut;

	double overal_CPU_time, overal_time;

	GpuTimer timer;
	double overal_GPU_time = 0, overal_data_transfer_time = 0, overal_data_allocation_time = 0;

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
	h_lut = (int*)malloc(256 * sizeof(int));

	//Device memory allocation
	printf("Allocating Device arrays...\n");

	d_img_in = NULL;
	cudaMalloc((void **)&d_img_in, h_img_in.w * h_img_in.h * sizeof(unsigned char) );
	cudaCheckError();

	d_img_out = NULL;
	cudaMalloc((void **)&d_img_out, h_img_in.w * h_img_in.h * sizeof(unsigned char) );
	cudaCheckError();

	d_hist_in = NULL;
	cudaMalloc((void **)&d_hist_in, 256 * sizeof(int));
	cudaCheckError();

	d_lut = NULL;
	cudaMalloc((void **)&d_lut, 256 * sizeof(int));
	cudaCheckError();


	// Main Function call

	printf("Starting CPU processing...\n");

	start2 = clock();
	for (int i = 0; i < 256; i++) {
		h_hist_buffer[i] = 0;
	}

	for (int i = 0; i < h_img_in.w * h_img_in.h; i++) {
		h_hist_buffer[h_img_in.img[i]] ++;
	}
	// Can be further GPUed
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
	end2 = clock();
	// Data transfer to Device
	timer.Start();
	cudaMemcpy(d_img_in, h_img_in.img, h_img_in.w * h_img_in.h * sizeof(unsigned char), cudaMemcpyHostToDevice);
	timer.Stop();
	overal_data_transfer_time += timer.Elapsed();
	cudaCheckError();

	timer.Start();
	cudaMemcpy(d_hist_in, h_hist_buffer, 256 * sizeof(int), cudaMemcpyHostToDevice);
	timer.Stop();
	overal_data_transfer_time += timer.Elapsed();
	cudaCheckError();

	timer.Start();
	cudaMemcpy(d_lut, h_lut, 256 * sizeof(int), cudaMemcpyHostToDevice);
	timer.Stop();
	overal_data_transfer_time += timer.Elapsed();
	cudaCheckError();

	printf("Starting GPU processing...\n");
	//Kernel invocations

	int blocksPerGrid = ((h_img_in.w * h_img_in.h)/1024);
	int threads_number = 1024;

	//printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threads_number);
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threads_number);
	timer.Start();
	histogram_equalizationGPU<<<blocksPerGrid, threads_number>>>(d_img_out, d_img_in ,
		 											h_img_in.h * h_img_in.w, 256, d_lut,
													(threads_number)* (blocksPerGrid));
	timer.Stop();
	overal_GPU_time += timer.Elapsed();
	cudaCheckError();

	cudaDeviceSynchronize();
	cudaCheckError();

	//Return Stuff to h_img_out_buf
	timer.Start();
	cudaMemcpy(h_img_out_buf.img, d_img_out, h_img_in.h * h_img_in.w * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	timer.Stop();
	overal_data_transfer_time += timer.Elapsed();
	cudaCheckError();

	cudaDeviceReset();
	// I/O Stuff
    write_pgm(h_img_out_buf, argv[2]);
	free_pgm(h_img_in);
    free_pgm(h_img_out_buf);

	timer.Start();
	cudaFree(d_hist_in);
	timer.Stop();
	overal_data_allocation_time += timer.Elapsed();
	cudaCheckError();

	timer.Start();
	cudaFree(d_img_in);
	timer.Stop();
	overal_data_allocation_time += timer.Elapsed();
	cudaCheckError();

	timer.Start();
	cudaFree(d_img_out);
	timer.Stop();
	overal_data_allocation_time += timer.Elapsed();
	cudaCheckError();

	printf("\nTime elapsed on GPU( computation) = %g ms\n", overal_GPU_time);

	printf("\nTime elapsed on GPU( memory transfers) = %g ms", overal_data_transfer_time);

	printf("\nTime elapsed on GPU( memory transfers) = %g ms", overal_data_allocation_time);

	printf("\nTime elapsed on GPU( overal) = %g ms\n", overal_GPU_time + overal_data_transfer_time + overal_data_allocation_time);

	end = clock();
	overal_time = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC ;
	overal_CPU_time = (double)(end2 - start2) * 1000.0 / CLOCKS_PER_SEC ;

	printf("Overal program time %g \n", overal_CPU_time);

	printf("Overal program time %g \n", overal_time);


	cudaDeviceReset();
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
