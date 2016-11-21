# Makefile. If you change it, remember than in makefiles multiple spaces
# ARE NOT EQUIVALENT to tabs. The line after a rule starts with a tab!

EXECUTABLES	= main

CC = gcc
CFLAGS = -Wall -g -std=c99

DEPS = hist-equ.h
OBJ = contrast-enhancement.o histogram-equalization.o
LDFLAGS = -lm -I.


%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(LDFLAGS)

main: main.c $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(EXECUTABLES) $(OBJ)

# make image will create the output_sobel.jpg from the output_sobel.grey.
# Remember to change this rule if you change the name of the output file.
image: output_sobel.grey
	convert -depth 8 -size 4096x4096 GRAY:output_sobel.grey output_sobel.jpg
