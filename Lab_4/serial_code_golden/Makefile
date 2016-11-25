# Makefile. If you change it, remember than in makefiles multiple spaces
# ARE NOT EQUIVALENT to tabs. The line after a rule starts with a tab!

EXECUTABLES	= main

CC = icc
CFLAGS = -Wall -g -std=c99

DEPS = hist-equ.h
OBJ = contrast-enhancement.o histogram-equalization.o
LDFLAGS = -lm -I.


%.o: %.c $(DEPS)
	$(CC) -ipo $(CFLAGS) -c -o $@ $< $(LDFLAGS)

main: main.c $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(EXECUTABLES) $(OBJ)
