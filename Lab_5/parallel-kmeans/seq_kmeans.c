/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         seq_kmeans.c  (sequential version)                        */
/*   Description:  Implementation of simple k-means clustering algorithm     */
/*                 This program takes an array of N data objects, each with  */
/*                 M coordinates and performs a k-means clustering given a   */
/*                 user-provided value of the number of clusters (K). The    */
/*                 clustering results are saved in 2 arrays:                 */
/*                 1. a returned array of size [K][N] indicating the center  */
/*                    coordinates of K clusters                              */
/*                 2. membership[N] stores the cluster center ids, each      */
/*                    corresponding to the cluster a data object is assigned */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department, Northwestern University                        */
/*            email: wkliao@ece.northwestern.edu                             */
/*                                                                           */
/*   Copyright (C) 2005, Northwestern University                             */
/*   See COPYRIGHT notice in top-level directory.                            */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "kmeans.h"


/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__inline static
float euclid_dist_2(int    numdims,  /* no. dimensions */
                    float *coord1,   /* [numdims] */
                    float *coord2)   /* [numdims] */
{
    int i;
    float ans=0.0;
	float l_ans=0.0;
	// Can be paralelized needs sync
	#pragma omp parallel for reduction (+:ans)
		for (i=0; i<numdims; i++)
			ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);


    return(ans);
}

/*----< find_nearest_cluster() >---------------------------------------------*/
__inline static
int find_nearest_cluster(int     numClusters, /* no. clusters */
                         int     numCoords,   /* no. coordinates */
                         float  *object,      /* [numCoords] */
                         float **clusters)    /* [numClusters][numCoords] */
{
    int   index, i, k, prv_return;
    float dist, min_dist;
	float ans=0.0;
    /* find the cluster id that has min distance to object */
    index    = 0;
	prv_return = 0;

	float *coord1 = object;
	float *coord2 = clusters[0];

	#pragma omp parallel default(shared)
	//private(index, min_dist, k, dist, i)
	{
		min_dist = euclid_dist_2(numCoords, object, clusters[0]);

		for (i=1; i<numClusters; i++) {
	        /* no need square root */
			dist = euclid_dist_2(numCoords, object, clusters[i]);
			//#pragma omp critical
	        if (dist < min_dist) { /* find the min and its array index */
	            min_dist = dist;
	            index    = i;
	        }
	    }

	}

    return(index);
}

/*----< seq_kmeans() >-------------------------------------------------------*/
/* return an array of cluster centers of size [numClusters][numCoords]       */
int par_kmeans(float **objects,      /* in: [numObjs][numCoords] */
               int     numCoords,    /* no. features */
               int     numObjs,      /* no. objects */
               int     numClusters,  /* no. clusters */
               float   threshold,    /* % objects change membership */
               int    *membership,   /* out: [numObjs] */
               float **clusters)     /* out: [numClusters][numCoords] */

{
    int      i, j, index, loop=0;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    float    delta;          /* % of objects change their clusters */
    float  **newClusters;    /* [numClusters][numCoords] */
	float *temp;

    /* initialize membership[] */
	//can be paralelized
	#pragma omp parallel for
    for (i=0; i<numObjs; i++) membership[i] = -1;

    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    newClusters    = (float**) malloc(numClusters *            sizeof(float*));
    assert(newClusters != NULL);
    newClusters[0] = (float*)  calloc(numClusters * numCoords, sizeof(float));
    assert(newClusters[0] != NULL);

	//Can be paralelized - data depentancy
	//#pragma omp parallel for
		for (i=1; i<numClusters; i++){
			newClusters[i] = newClusters[i-1] + numCoords;
		}


	//paralelized with tasks make plans

		do {
	        delta = 0.0;
			//#pragma omp for
			#pragma omp parallel private(i, j, index) \
								firstprivate( numObjs, numClusters, numCoords)
			{
	        for (i=0; i<numObjs; i++) {
	            /* find the array index of nestest cluster center */
	            index = find_nearest_cluster(numClusters, numCoords, objects[i],
	                                         clusters);

	            /* if membership changes, increase delta by 1 */
	            if (membership[i] != index) delta += 1.0;

	            /* assign the membership to object i */
	            membership[i] = index;

	            /* update new cluster center : sum of objects located within */
				#pragma omp atomic
	            newClusterSize[index]++;
				//#pragma omp for
	            for (j=0; j<numCoords; j++)
					#pragma omp atomic
	                newClusters[index][j] += objects[i][j];
	        }
			}

	        /* average the sum and replace old cluster center with newClusters */
			//Maybe paralell
	        for (i=0; i<numClusters; i++) {
	            for (j=0; j<numCoords; j++) {
	                if (newClusterSize[i] > 0)
	                    clusters[i][j] = newClusters[i][j] / newClusterSize[i];
	                newClusters[i][j] = 0.0;   /* set back to 0 */
	            }
	            newClusterSize[i] = 0;   /* set back to 0 */
	        }

	        delta /= numObjs;
	    } while (delta > threshold && loop++ < 500);

	//}


    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return 1;
}
