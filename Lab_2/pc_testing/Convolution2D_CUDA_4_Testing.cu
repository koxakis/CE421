/*
* This sample implements a separable convolution
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.00005

#define FLOAT
//#define DOUBLE

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
void convolutionRowCPU(int *h_Dst, int *h_Src, int *h_Filter,int imageW, int imageH, int filterR) {

	int x, y, k;

	for (y = FILTER_LENGTH; y < imageH; y++) {
		for (x = FILTER_LENGTH; x < imageW; x++) {
			int sum = 0;

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
void convolutionColumnCPU(int *h_Dst, int *h_Src, int *h_Filter,int imageW, int imageH, int filterR) {

	int x, y, k;

	for (y = FILTER_LENGTH; y < imageH; y++) {
		for (x = FILTER_LENGTH; x < imageW; x++) {
			int sum = 0;

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
convolutionRowDevice(int *d_Dst, int *d_Src, int *d_Filter,int imageW, int imageH, int filterR)
{
	int k;

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	int sum = 0;

	for (k = -filterR; k <= filterR; k++) {
		int d = row + k;

		sum += d_Src[col * imageW + d] * d_Filter[filterR - k];

		d_Dst[col * imageW + row] = sum;
	}

}

__global__ void
convolutionColumnDevice(int *d_Dst, int *d_Src, int *d_Filter,int imageW, int imageH, int filterR)
{
	int k;

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	int sum = 0;

	for (k = -filterR; k <= filterR; k++) {
		int d = col + k;

		//if (d>=0) {
			sum += d_Src[col * imageW + d] * d_Filter[filterR - k];
		//}
		d_Dst[col * imageW + row] = sum;
	}

}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

#ifdef FLOAT
	int
	*h_Filter,
	*h_Input,
	*h_Buffer,
	*h_OutputCPU,
	*h_OutputGPU;

	int
	*d_Filter,
	*d_Input,
	*d_Buffer,
	*d_OutputD;
#endif
#ifdef DOUBLE
#endif

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

#ifdef FLOAT
	h_Filter    = (int *)malloc(FILTER_LENGTH * sizeof(int));
	h_Input     = (int *)malloc((imageW + (filter_radius * 2)) * (imageH + (filter_radius * 2)) * sizeof(int));
	h_Buffer    = (int *)malloc((imageW + (filter_radius * 2)) * (imageH + (filter_radius * 2)) * sizeof(int));
	h_OutputCPU = (int *)malloc(imageW * imageH * sizeof(int));
	h_OutputGPU = (int *)malloc(imageW * imageH * sizeof(int));


	if ( h_Filter == NULL || h_Input == NULL || h_Buffer == NULL || h_OutputCPU == NULL || h_OutputGPU == NULL) {
		fprintf(stderr, "Failed to allocate Host matrices!\n");
        exit(EXIT_FAILURE);
	}

	printf("Allocating Device arrays...\n");
	// Device mallocs
	d_Filter = NULL;
	cudaMalloc((void **)&d_Filter, FILTER_LENGTH * sizeof(int));
	cudaCheckError();

	d_Input = NULL;
	cudaMalloc((void **)&d_Input, ((imageW + (filter_radius * 2)) * (imageH + (filter_radius * 2))) * sizeof(int));
	cudaCheckError();

	d_Buffer = NULL;
	cudaMalloc((void **)&d_Buffer, ((imageW + (filter_radius * 2)) * (imageH + (filter_radius * 2))) * sizeof(int));
	cudaCheckError();

	d_OutputD = NULL;
	cudaMalloc((void **)&d_OutputD, imageW * imageH * sizeof(int));
	cudaCheckError();

	// to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
	// arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
	// to convolution kai arxikopoieitai kai auth tuxaia.
	printf("Initializing Host arrays...\n");
	srand(200);

	//fil the table normally or fill at the start ??
	for (i = 0; i < FILTER_LENGTH; i++) {
		h_Filter[i] = (int)(rand() % 16);
	}
	for (i = FILTER_LENGTH * ((imageW + (filter_radius * 2))) + filter_radius; i < imageW * imageH; i++) {
		h_Input[i] = (int)rand() / ((int)RAND_MAX / 255) + (int)rand() / (int)RAND_MAX;
	}

	printf("Initializing Device arrays...\n");
	// Transfer Data to Device
	cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(int), cudaMemcpyHostToDevice);
	cudaCheckError();

	cudaMemcpy(d_Input, h_Input, (imageW + (filter_radius * 2)) * (imageH + (filter_radius * 2)) * sizeof(int), cudaMemcpyHostToDevice);
	cudaCheckError();

#endif

#ifdef DOUBLE
#endif

	// To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
	printf("CPU computation...\n");

	convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
	convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles


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

	convolutionRowDevice<<<grid, threads>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);
	cudaCheckError();

	cudaDeviceSynchronize();
	cudaCheckError();

	// convolution by columns device
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid*blocksPerGrid, threadsPerBlock*threadsPerBlock);

	convolutionColumnDevice<<<grid, threads>>>(d_OutputD, d_Buffer, d_Filter, imageW, imageH, filter_radius);
	cudaCheckError();

	cudaDeviceSynchronize();
	cudaCheckError();

	// Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
#ifdef FLOAT
    cudaMemcpy(h_OutputGPU, d_OutputD, imageW * imageH * sizeof(int), cudaMemcpyDeviceToHost);
#endif
#ifdef DOUBLE
#endif
	cudaCheckError();

	for (unsigned i = 0; i < imageH * imageW; i++) {
		if ( h_OutputCPU[i] != h_OutputGPU[i]){
			printf("Algorithm not correct \nCheck if algorithm is correct or it is just inaccured" );
			break;
		}
	}

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

    printf("Max diff: %lf\n\n", max_diff);

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
