#Add any executable you want to be created here.
EXECUTABLES	= Convolution2D_CUDA_Gird Convolution2D_CUDA_Tiled Convolution2D_CUDA_Tiled_test

#This is the compiler to use
CC = nvcc

#These are the flags passed to the compiler. Change accordingly
CFLAGS = -arch sm_30 -lineinfo --ptxas-options=-v -Xcompiler -Wall -O4 -g

#These are the flags passed to the linker. Nothing in our case
LDFLAGS = -lm

# make all will create all executables
Convolution2D_CUDA_Gird: Convolution2D_CUDA_Gird.cu

%: %.cu
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

Convolution2D_CUDA_Tiled: Convolution2D_CUDA_Tiled.cu

%: %.cu
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

Convolution2D_CUDA_Tiled_test: Convolution2D_CUDA_Tiled_test.cu

%: %.cu
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)


# make clean will remove all executables, jpg files and the
# output of previous executions.
clean:
	rm -f $(EXECUTABLES)
