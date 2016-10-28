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



////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter,int imageW, int imageH, int filterR) {

	int x, y, k;

	for (y = 0; y < imageH; y++) {
		for (x = 0; x < imageW; x++) {
			float sum = 0;

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
void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,int imageW, int imageH, int filterR) {

	int x, y, k;

	for (y = 0; y < imageH; y++) {
		for (x = 0; x < imageW; x++) {
			float sum = 0;

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
convolutionRowDevice(float *d_Dst, float *d_Src, float *d_Filter,int imageW, int imageH, int filterR)
{
	//printf("Hello world from the convolutionRowDevice! block=%d, thread=%d\n", blockIdx.x, threadIdx.x);
	int x, y, k;
	int x_pos = blockDim.x * blockIdx.x + threadIdx.x;
	int y_pos = blockDim.y * blockIdx.y + threadIdx.y;

	for (y = 0; y < imageH; y++) {
		for (x = 0; x < imageW; x++) {
			float sum = 0;

			for (k = -filterR; k <= filterR; k++) {
				int d = x_pos + k;

				if (d >= 0 && d < imageW) {
					//sum += d_Src[y * imageW + d] * d_Filter[filterR - k];
					//sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
					sum += d_Src[filterR - k] * d_Src[y_pos + d];
					printf(" %d %d\n", filterR - k, y_pos + d);
				}

				//d_Dst[y * imageW + x] = sum;
				d_Dst[y * x_pos] = sum;
				printf(" %d\n", y * x_pos);
			}
		}
	}
}

__global__ void
convolutionColumnDevice(float *d_Dst, float *d_Src, float *d_Filter,int imageW, int imageH, int filterR)
{
	int x, y, k;
	int x_pos = blockDim.x * blockIdx.x + threadIdx.x;
	int y_pos = blockDim.y * blockIdx.y + threadIdx.y;

	for (y = 0; y < imageH; y++) {
		for (x = 0; x < imageW; x++) {
			float sum = 0;

			for (k = -filterR; k <= filterR; k++) {
				int d = y + k;

				if (d >= 0 && d < imageH) {
					//sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
					//sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
					sum += d_Src[filterR - k] * d_Src[y_pos + d];
				}

				//h_Dst[y * imageW + x] = sum;
				d_Dst[y * x_pos] = sum;
			}
		}
	}


}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

	float
	*h_Filter,
	*h_Input,
	*h_Buffer,
	*h_OutputCPU,
	*h_OutputGPU;

	float
	*d_Filter,
	*d_Input,
	*d_Buffer,
	*d_OutputD;

	int imageW;
	int imageH;
	unsigned int N;
	unsigned int i;

	cudaError_t err = cudaSuccess;

	// Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
	// dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
	// Gia aplothta thewroume tetragwnikes eikones.

	if ( argc != 3){
		printf("Missmach in argument input \n");
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
	printf("Allocating and initializing host arrays...\n");
	// Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
	// Host mallocs
	h_Filter    = (float *)malloc(FILTER_LENGTH * sizeof(float));
	h_Input     = (float *)malloc(imageW * imageH * sizeof(float));
	h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
	h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
	h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float));

	if ( h_Filter == NULL || h_Input == NULL || h_Buffer == NULL || h_OutputCPU == NULL || h_OutputGPU == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
	}

	// Device mallocs
	d_Filter = NULL;
	err = cudaMalloc((void **)&d_Filter, FILTER_LENGTH * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device Filter (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	d_Input = NULL;
	err = cudaMalloc((void **)&d_Input, imageW * imageH * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device Input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	d_Buffer = NULL;
	err = cudaMalloc((void **)&d_Buffer, imageW * imageH * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device Buffer (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	d_OutputD = NULL;
	err = cudaMalloc((void **)&d_OutputD, imageW * imageH * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device Output (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	// to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
	// arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
	// to convolution kai arxikopoieitai kai auth tuxaia.

	srand(200);

	for (i = 0; i < FILTER_LENGTH; i++) {
		h_Filter[i] = (float)(rand() % 16);
	}
	for (i = 0; i < imageW * imageH; i++) {
		h_Input[i] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
	}

	// Transfer Data to Device
	err = cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy filter from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy input from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	// To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
	printf("CPU computation...\n");

	convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
	convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles


	// Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
	// pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas

	printf("GPU computation...\n");
	// Error code to check return values for CUDA calls

	// Kernel paramiters prep
	int threadsPerBlock = 4;
	int blocksPerGrid = 1;
	dim3 threads(threadsPerBlock, threadsPerBlock);
	dim3 grid(1,1);
	//int blocksPerGrid =(imageH + threadsPerBlock - 1) / threadsPerBlock;

	// convolution by rows device
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid*blocksPerGrid, threadsPerBlock*threadsPerBlock);
	//convolutionRowDevice<<<blocksPerGrid, threadsPerBlock>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);
	convolutionRowDevice<<<grid, threads>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);
	err = cudaGetLastError();

	cudaDeviceSynchronize();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch convolutionRowDevice kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// convolution by columns device
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid*blocksPerGrid, threadsPerBlock*threadsPerBlock);
	//convolutionColumnDevice<<<blocksPerGrid, threadsPerBlock>>>(d_OutputD, d_Buffer, d_Filter, imageW, imageH, filter_radius);
	convolutionColumnDevice<<<grid, threads>>>(d_OutputD, d_Buffer, d_Filter, imageW, imageH, filter_radius);
	err = cudaGetLastError();

	cudaDeviceSynchronize();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch convolutionColumnDevice kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_OutputGPU, d_OutputD, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy result from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	printf(" Comparing the outputs\n");
    double sum = 0, delta = 0;

    for (unsigned i = 0; i < imageW * imageH; i++)
    {
        delta += (h_OutputGPU[i] - h_OutputCPU[i]) * (h_OutputGPU[i] - h_OutputCPU[i]);
        sum   += h_OutputCPU[i] * h_OutputCPU[i];
		if ( delta > accuracy){
			printf("The accuracy is not good enough\n" );
			break;
		}
    }
	double L2norm = sqrt(delta / sum);
    printf(" Relative L2 norm: %E\n\n", L2norm);


	// free all the allocated memory
	free(h_OutputCPU);
	free(h_Buffer);
	free(h_Input);
	free(h_Filter);

	err = cudaFree(d_OutputD);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device output (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaFree(d_Buffer);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device buffer (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_Input);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device input (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_Filter);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device filter (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
	cudaDeviceReset();


	return 0;
}
