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

/*----< seq_kmeans() >-------------------------------------------------------*/
/* return an array of cluster centers of size [numClusters][numCoords]       */
int seq_kmeans(float **objects,      /* in: [numObjs][numCoords] */
               int     numCoords,    /* no. features */
               int     numObjs,      /* no. objects */
               int     numClusters,  /* no. clusters */
               float   threshold,    /* % objects change membership */
               int    *membership,   /* out: [numObjs] */
               float **clusters)     /* out: [numClusters][numCoords] */

{
    int      i, j, index, loop=0, k, v, l;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
	float  dist, min_dist = 0.0;
	float ans = 0.0;
    float    delta;          /* % of objects change their clusters */
    float  **newClusters;    /* [numClusters][numCoords] */

    /* initialize membership[] */
	#pragma omp parallel for
    for (i=0; i<numObjs; i++) membership[i] = -1;

    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    newClusters    = (float**) malloc(numClusters *            sizeof(float*));
    assert(newClusters != NULL);
    newClusters[0] = (float*)  calloc(numClusters * numCoords, sizeof(float));
    assert(newClusters[0] != NULL);

    for (i=1; i<numClusters; i++)
        newClusters[i] = newClusters[i-1] + numCoords;

    do {
        delta = 0.0;
		#pragma omp parallel for \
				private (i, j, l, index) \
				firstprivate(numObjs, numClusters, numCoords) \
				schedule (dynamic) \
				reduction (+:delta)
        for (i=0; i<numObjs; i++) {
            /* find the array index of nestest cluster center */
			/* find the cluster id that has min distance to object */
		    index    = 0;
			//#pragma omp for reduction (+:min_dist)
			for (l=0, min_dist = 0.0; l<numCoords; l++)
		        min_dist += (objects[i][l]-clusters[0][l]) * (objects[i][l]-clusters[0][l]);
		    //min_dist = euclid_dist_2(numCoords, object, clusters[0]);

		    for (l=1; l<numClusters; l++) {
				//#pragma omp for reduction (+:ans)
				for (k=0, ans = 0.0; k<numCoords; k++)
			        ans += (objects[i][k]-clusters[l][k]) * (objects[i][k]-clusters[l][k]);
		        //dist = euclid_dist_2(numCoords, object, clusters[i]);
				dist = ans;
		        /* no need square root */
		        if (dist < min_dist) { /* find the min and its array index */
		            min_dist = dist;
		            index    = l;
		        }
		    }

            //index = find_nearest_cluster(numClusters, numCoords, objects[i],
            //                             clusters);

            /* if membership changes, increase delta by 1 */
            if (membership[i] != index) delta += 1.0;

            /* assign the membership to object i */
            membership[i] = index;

            /* update new cluster center : sum of objects located within */
			#pragma omp atomic
            newClusterSize[index]++;
            for (j=0; j<numCoords; j++)
				#pragma omp atomic
                newClusters[index][j] += objects[i][j];
        }

        /* average the sum and replace old cluster center with newClusters */
		#pragma omp master
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


    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return 1;
}
