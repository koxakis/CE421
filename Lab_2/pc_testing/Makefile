#Use this compiler
CC = nvcc

#Use this compiler flags
CFLAGS = -arch sm_30 -G -lineinfo -Xcompiler -Wall -O4 -g

TARGET = Convolution2D_CUDA_4

#Build target example
all: Convolution2D_CUDA Convolution2D_CUDA_4
	
Convolution2D_CUDA: Convolution2D_CUDA.c
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).c

Convolution2D_CUDA_4: Convolution2D_CUDA_4.c
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).c

clean:
	rm -f test1 a.out $(TARGET)
