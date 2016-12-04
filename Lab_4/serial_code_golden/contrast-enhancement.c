#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include <time.h>

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result;
    int hist[256];

	double cpu_time = 0;
	clock_t start, end;

    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

	// Redirect to histogram-equalization.c
	start = clock();
    histogram(hist, img_in.img, img_in.h * img_in.w, 256);
    histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256);
	end = clock();

	cpu_time = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC ;
	printf("CPU time: %g \n", cpu_time);

	//Return resulting image
    return result;
}
