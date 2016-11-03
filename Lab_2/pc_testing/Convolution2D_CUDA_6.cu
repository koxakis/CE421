/*
* This sample implements a separable convolution
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include "gputimer.h"
#include <time.h>

unsigned int filter_radius;
GpuTimer timer;
float overal_time = 0;
clock_t start, end;
double overal_CPU_time;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.00005

#define cudaCheckError() {                                                                       \
        cudaError_t e=cudaGetLastError();                                                        \
        if(e!=cudaSuccess) {                                                                     \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    }

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(double *h_Dst, double *h_Src, double *h_Filter,int imageW, int imageH, int filterR) {

	int x, y, k;

	for (y = 0; y < imageH; y++) {
		for (x = 0; x < imageW; x++) {
			double sum = 0;

			for (k = -filterR; k <= filterR; k++) {
				int d = x + k;

				if (d >= 0 && d < imageW) {
					sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
				}

				h_Dst[y * imageW + x] = sum;
			}
		}
	}

}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(double *h_Dst, double *h_Src, double *h_Filter,int imageW, int imageH, int filterR) {

	int x, y, k;

	for (y = 0; y < imageH; y++) {
		for (x = 0; x < imageW; x++) {
			double sum = 0;

			for (k = -filterR; k <= filterR; k++) {
				int d = y + k;

				if (d >= 0 && d < imageH) {
					sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
				}

				h_Dst[y * imageW + x] = sum;
			}
		}
	}

}

////////////////////////////////////////////////////////////////////////////////
// Device code
////////////////////////////////////////////////////////////////////////////////

__global__ void
convolutionRowDevice(double *d_Dst, double *d_Src, double *d_Filter,int imageW, int imageH, int filterR)
{
	int k;

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	double sum = 0;

	for (k = -filterR; k <= filterR; k++) {
		int d = row + k;

		if (d >= 0 && d < imageW) {
			//sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
			sum += d_Src[col * imageW + d] * d_Filter[filterR - k];
		}
		//h_Dst[y * imageW + x] = sum;
		d_Dst[col * imageW + row] = sum;
	}

}


__global__ void
convolutionColumnDevice(double *d_Dst, double *d_Src, double *d_Filter,int imageW, int imageH, int filterR)
{
	int k;

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	double sum = 0;

	for (k = -filterR; k <= filterR; k++) {
		int d = col + k;

		if (d >= 0 && d < imageH) {
			//sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
			sum += d_Src[d * imageW + row] * d_Filter[filterR -k];
		}
		//h_Dst[y * imageW + x] = sum;
		d_Dst[col * imageW + row] = sum;
	}

}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

	double
	*h_Filter,
	*h_Input,
	*h_Buffer,
	*h_OutputCPU,
	*h_OutputGPU;

	double
	*d_Filter,
	*d_Input,
	*d_Buffer,
	*d_OutputD;

	int imageW;
	int imageH;
	unsigned int N;
	unsigned int i;

	// Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
	// dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
	// Gia aplothta thewroume tetragwnikes eikones.

	if ( argc != 3){
		printf("Missmach in argument input \n");
		printf("1st argument: Image Size \n 2nd argument: Filter Radius \n");
		return 0;
	}

	filter_radius = atoi(argv[1]);

	N = atoi(argv[2]);
	imageH = N;
	imageW = N;

	if ( N < FILTER_LENGTH || N%2 != 0 ){
		printf ( "Wrong image size \n");
		printf ( "It should be greater than %d and a power of 2 \n", FILTER_LENGTH);
		return 0;
	}


	printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
	printf("Allocating host arrays...\n");
	// Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
	// Host mallocs

	h_Filter    = (double *)malloc(FILTER_LENGTH * sizeof(double));
	h_Input     = (double *)malloc(imageW * imageH * sizeof(double));
	h_Buffer    = (double *)malloc(imageW * imageH * sizeof(double));
	h_OutputCPU = (double *)malloc(imageW * imageH * sizeof(double));
	h_OutputGPU = (double *)malloc(imageW * imageH * sizeof(double));

	if ( h_Filter == NULL || h_Input == NULL || h_Buffer == NULL || h_OutputCPU == NULL || h_OutputGPU == NULL) {
		fprintf(stderr, "Failed to allocate Host matrices!\n");
        exit(EXIT_FAILURE);
	}

	printf("Allocating Device arrays...\n");
	// Device mallocs
	d_Filter = NULL;
	cudaMalloc((void **)&d_Filter, FILTER_LENGTH * sizeof(double));
	cudaCheckError();

	d_Input = NULL;
	cudaMalloc((void **)&d_Input, imageW * imageH * sizeof(double));
	cudaCheckError();

	d_Buffer = NULL;
	cudaMalloc((void **)&d_Buffer, imageW * imageH * sizeof(double));
	cudaCheckError();

	d_OutputD = NULL;
	cudaMalloc((void **)&d_OutputD, imageW * imageH * sizeof(double));
	cudaCheckError();

	// to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
	// arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
	// to convolution kai arxikopoieitai kai auth tuxaia.
	printf("Initializing Host arrays...\n");
	srand(200);

	for (i = 0; i < FILTER_LENGTH; i++) {
		h_Filter[i] = (double)(rand() % 16);
	}
	for (i = 0; i < imageW * imageH; i++) {
		h_Input[i] = (double)rand() / ((double)RAND_MAX / 255) + (double)rand() / (double)RAND_MAX;
	}

	printf("Initializing Device arrays...\n");
	// Transfer Data to Device
	timer.Start();
	cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(double), cudaMemcpyHostToDevice);
	timer.Stop();
	overal_time = overal_time + timer.Elapsed();
	cudaCheckError();

	timer.Start();
	cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(double), cudaMemcpyHostToDevice);
	timer.Stop();
	overal_time = overal_time + timer.Elapsed();
	cudaCheckError();

	// To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
	printf("CPU computation...\n");

	start = clock();
	convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
	convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
	end = clock();

	// Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
	// pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas

	printf("GPU computation...\n");

	// Kernel paramiters prep
	int threadsPerBlock;
	if (N >= 32){
		threadsPerBlock = 32;
	}else{
		threadsPerBlock = N;
	}
	dim3 threads(threadsPerBlock, threadsPerBlock);

	int blocksPerGrid;
	if ( N>=32){
		blocksPerGrid =  N/threads.x;
	}else{
		blocksPerGrid = 1;
	}
	dim3 grid(blocksPerGrid,blocksPerGrid);

	// convolution by rows device
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid*blocksPerGrid, threadsPerBlock*threadsPerBlock);

	timer.Start();
	convolutionRowDevice<<<grid, threads>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);
	timer.Stop();
	overal_time = overal_time + timer.Elapsed();
	cudaCheckError();

	cudaDeviceSynchronize();
	cudaCheckError();

	// convolution by columns device
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid*blocksPerGrid, threadsPerBlock*threadsPerBlock);

	timer.Start();
	convolutionColumnDevice<<<grid, threads>>>(d_OutputD, d_Buffer, d_Filter, imageW, imageH, filter_radius);
	timer.Stop();
	overal_time = overal_time + timer.Elapsed();
	cudaCheckError();

	cudaDeviceSynchronize();
	cudaCheckError();

	// Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");

	timer.Start();
    cudaMemcpy(h_OutputGPU, d_OutputD, imageW * imageH * sizeof(double), cudaMemcpyDeviceToHost);
	timer.Stop();
	overal_time = overal_time + timer.Elapsed();

	cudaCheckError();

	printf("\nComparing the outputs\n");
    double max_diff=0, temp;

    for (unsigned i = 0; i < imageW * imageH; i++)
    {
    	temp = ABS(h_OutputCPU[i] - h_OutputGPU[i]);
		if (max_diff < temp) {
			max_diff = temp;
		}

		if ( max_diff > accuracy){
			printf("The accuracy is not good enough\n" );
			break;
		}
    }

    printf("Max diff: %g\n\n", max_diff);
	printf("Time elapsed = %g ms\n", overal_time);

	overal_CPU_time = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC ;
	printf ("Time elapsed on CPU = %g ms\n", overal_CPU_time);


	// free all the allocated memory
	free(h_OutputCPU);
	free(h_Buffer);
	free(h_Input);
	free(h_Filter);

	cudaFree(d_OutputD);
	cudaCheckError();

	cudaFree(d_Buffer);
	cudaCheckError();

	cudaFree(d_Input);
	cudaCheckError();

	cudaFree(d_Filter);
	cudaCheckError();

	// Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
	cudaDeviceReset();


	return 0;
}
