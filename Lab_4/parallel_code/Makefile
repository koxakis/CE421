# Makefile. If you change it, remember than in makefiles multiple spaces
# ARE NOT EQUIVALENT to tabs. The line after a rule starts with a tab!

EXECUTABLES	= main

CC = nvcc
CFLAGS = -arch sm_30 -lineinfo -Xcompiler -g -O4

DEPS = hist-equ.h
OBJ = contrast-enhancement.o histogram-equalization.o
LDFLAGS = -lm -I.


main: main.cu $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(EXECUTABLES) $(OBJ)
