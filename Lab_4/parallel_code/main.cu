#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

//void run_cpu_gray_test(PGM_IMG img_in, char *out_filename);

void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    for ( i = 0; i < img_size; i ++){
        hist_out[img_in[i]] ++;
    }
}

__global__ void histogramGPU ( int * d_hist_out, unsigned char * d_img_in, int img_size, int nbr_bin) {
	/* code */
}

__global__ void histogram_equalizationGPU ( unsigned char * d_img_out, unsigned char * d_img_in,
											int * d_hist_in, int img_size, int nbr_bin) {
	/* code */
}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in,
                            int * hist_in, int img_size, int nbr_bin){

	int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist_in[i++];
    }
    d = img_size - min;
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }


    }

    /* Get the result image */
    for(i = 0; i < img_size; i ++){
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }

    }
}
int main(int argc, char *argv[]){
	// Host Variables
    PGM_IMG img_ibuf_g;
	unsigned int timer = 0;
    PGM_IMG img_obuf;

	//from contrast enhancement
	PGM_IMG result;
    int hist[256];

	result.w = img_in.w;
	result.h = img_in.h;
	result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

	if (argc != 3) {
		printf("Run with input file name and output file name as arguments\n");
		exit(1);
	}

    printf("Running contrast enhancement for gray-scale images.\n");
    img_ibuf_g = read_pgm(argv[1]);
	// Main Function call
	// Data transfer to Device

	printf("Starting GPU processing...\n");
	// Redirect to contrast-enhancement.c
	histogram(hist, img_in.img, img_in.h * img_in.w, 256);
    histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256);

	// I/O Stuff
    write_pgm(img_ibuf_g, argv[2]);
    //free_pgm(img_ibuf_g);

    free_pgm(img_ibuf_g);

	return 0;
}

// I/O Stuff implementation
PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];


    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }

    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);


    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));


    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);
    fclose(in_file);

    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    free(img.img);
}
