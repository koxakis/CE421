#Add any executable you want to be created here.
EXECUTABLES	= Convolution2D_CUDA_4 Convolution2D_CUDA

#This is the compiler to use
CC = nvcc

#These are the flags passed to the compiler. Change accordingly
CFLAGS = -arch sm_30 -G -lineinfo -Xcompiler -Wall -O4 -g


# make all will create all executables
Convolution2D_CUDA.c: Convolution2D_CUDA

# This is the rule to create any executable from the corresponding .c
# file with the same name.
%: %.cu
	$(CC) $(CFLAGS) $< -o $@

Convolution2D_CUDA_4.c: Convolution2D_CUDA_4

%: %.cu
	$(CC) $(CFLAGS) $< -o $@


# make clean will remove all executables, jpg files and the
# output of previous executions.
clean:
	rm -f $(EXECUTABLES)
