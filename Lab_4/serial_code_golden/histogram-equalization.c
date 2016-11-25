#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

//Build the main histogram
void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;

	// Initialize all intencity values to 0
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

	// Calculate the # of pixels for each intencity values
    for ( i = 0; i < img_size; i ++){
        hist_out[img_in[i]] ++;
    }
}

//Equalize histogram
void histogram_equalization(unsigned char * img_out, unsigned char * img_in,
                            int * hist_in, int img_size, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
	// Some Loop
    while(min == 0){
        min = hist_in[i++];
    }
	// Some oporation
    d = img_size - min;
	// Some other Loop
	//Data dependancy maybe 
    for(i = 0; i < nbr_bin; i ++){
		// Some cumulitive addition
        cdf += hist_in[i];
		// Breaks most images
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
		//locations based somthing
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
		//if somthing then somthing
        if(lut[i] < 0){
            lut[i] = 0;
        }


    }

    /* Get the result image */
	// Hiden paralelism ??? Maybe
    for(i = 0; i < img_size; i ++){
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }

    }
}
